import os
import re
import pickle

import numpy as np
import sentencepiece as spm
import pandas as pd
from copy import deepcopy
from nltk import download
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from nltk.corpus import stopwords
from scipy.stats import spearmanr, pearsonr, alpha
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from enum import Enum
from typing import Optional, Any


class ModelType(Enum):
    RIDGE_REGRESSION_MODEL = 0,
    XGBOOST_REGRESSION_MODEL = 1,


download('stopwords')

DEBUG_MODE = os.environ.get("DEBUG_MODE", True)
SCRIPT_ENVIRONMENT = os.environ.get("SCRIPT_ENVIRONMENT", "development")  # or kaggle
WORD_TOKENIZATION = os.environ.get("WORD_TOKENIZATION", "sentencepiece")  # or sentencepiece

MODEL_TYPE = ModelType.RIDGE_REGRESSION_MODEL if int(
    os.environ.get("MODEL_TYPE", ModelType.XGBOOST_REGRESSION_MODEL.value[0])
) == 1 else ModelType.RIDGE_REGRESSION_MODEL
print(f'Initialized with the model: {MODEL_TYPE}')


class EnumDatasetType(Enum):
    TRAIN = "train.csv",
    VALIDATION = "val.csv",
    TEST = "test.csv",


def get_dataset_based_on_environment(
        environment: EnumDatasetType,
        dataset_path: Optional[str] = os.path.join(os.getcwd(), "data"),
        debugging_mode: Optional[bool] = DEBUG_MODE
):
    environment_dataset_path = f"{dataset_path}/{environment.value[0]}"
    if debugging_mode:
        print(f"[get_dataset_based_on_environment] Read dataset from {environment_dataset_path}...")
    return pd.read_csv(environment_dataset_path)


def preprocess_dataset(content: str):
    global set_stopwords

    regex_replace_contents = [r'[^\w\s]', r'\d+']
    for regex_replace_content in regex_replace_contents:
        content = re.sub(regex_replace_content, '', content, re.IGNORECASE)

    content = content.lower()
    remove_content_stopwords = [word for word in content.split() if word not in set_stopwords]

    return " ".join(remove_content_stopwords)


def feature_extraction_word2vec(sentence: str):
    global word2vec

    embeddings = [word2vec.wv[word] for word in sentence if word in word2vec.wv]
    return np.mean(embeddings, axis=0) if len(embeddings) > 0 else np.zeros(word2vec.vector_size)


if __name__ == "__main__":
    set_stopwords = set(stopwords.words('english'))
    train_dataframe = get_dataset_based_on_environment(EnumDatasetType.TRAIN, debugging_mode=True)
    validation_dataframe = get_dataset_based_on_environment(EnumDatasetType.VALIDATION, debugging_mode=True)
    test_dataframe = get_dataset_based_on_environment(EnumDatasetType.TEST, debugging_mode=True)

    train_dataframe['processed_text'] = train_dataframe['text'].apply(lambda text: preprocess_dataset(text))
    validation_dataframe['processed_text'] = validation_dataframe['text'].apply(lambda text: preprocess_dataset(text))
    test_dataframe['processed_text'] = test_dataframe['text'].apply(lambda text: preprocess_dataset(text))

    processed_texts = pd.concat([
        train_dataframe['processed_text'],
        validation_dataframe['processed_text'],
        test_dataframe['processed_text']
    ], axis=0)

    train_sentences, validation_sentences, test_sentences, processed_sentences = [], [], [], []
    if WORD_TOKENIZATION == "normal_split":
        train_sentences = train_dataframe['processed_text'].apply(lambda sentence: sentence.split()).tolist()
        validation_sentences = validation_dataframe['processed_text'].apply(lambda sentence: sentence.split()).tolist()
        test_sentences = test_dataframe['processed_text'].apply(lambda sentence: sentence.split()).tolist()

        processed_sentences = processed_texts.apply(lambda sentence: sentence.split()).tolist()
    elif WORD_TOKENIZATION == "sentencepiece":
        if not os.path.exists(os.path.join('data', 'sentences.txt')):
            processed_sentences = processed_texts.values.tolist()
            with open(os.path.join('data', 'sentences.txt'), 'w') as sentence_file:
                sentence_file.write('\n'.join(processed_sentences))
        if not os.path.exists(os.path.join('./', 'spm_vsc_model.model')):
            spm.SentencePieceTrainer.train(
                input="data/sentences.txt",
                model_prefix="spm_vsc_model",
                vocab_size=8000,
                character_coverage=1.0,
                model_type='bpe'
            )
        sentencepiece_piece_tokenizer = spm.SentencePieceProcessor(model_file="spm_vsc_model.model")
        train_sentences = (train_dataframe['processed_text']
                           .apply(lambda sentence: sentencepiece_piece_tokenizer.EncodeAsPieces(sentence))
                           .tolist())
        validation_sentences = (validation_dataframe['processed_text']
                                .apply(lambda sentence: sentencepiece_piece_tokenizer.EncodeAsPieces(sentence))
                                .tolist())
        test_sentences = (test_dataframe['processed_text']
                          .apply(lambda sentence: sentencepiece_piece_tokenizer.EncodeAsPieces(sentence))
                          .tolist())
        processed_sentences = (processed_texts
                               .apply(lambda sentence: sentencepiece_piece_tokenizer.EncodeAsPieces(sentence))
                               .tolist())

    word2vec_params = dict(
        sentences=processed_sentences,
        vector_size=175,
        workers=12,
        window=5,
        min_count=1,
        epochs=55
    )
    word2vec = Word2Vec(**word2vec_params)

    X, y = dict(train=Any, validation=Any, test=Any), dict(train=Any, validation=Any, test=Any)
    if SCRIPT_ENVIRONMENT == "development":
        X['train'] = np.array([feature_extraction_word2vec(sentence) for sentence in train_sentences])
        X['validation'] = np.array(
            [feature_extraction_word2vec(sentence) for sentence in validation_sentences])
    else:
        train_sentences = (train_sentences + validation_sentences)
        X['train'] = np.array([feature_extraction_word2vec(sentence) for sentence in train_sentences])
    X['test'] = np.array([feature_extraction_word2vec(sentence) for sentence in test_sentences])

    if SCRIPT_ENVIRONMENT == "development":
        y['train'] = train_dataframe['score'].values
        y['validation'] = validation_dataframe['score'].values
    else:
        y['train'] = pd.concat([train_dataframe['score'], validation_dataframe['score']], axis=0).values

    standard_scaler = StandardScaler()
    X['train'] = standard_scaler.fit_transform(X['train'])
    if SCRIPT_ENVIRONMENT == "development":
        X['validation'] = standard_scaler.transform(X['validation'])
    X['test'] = standard_scaler.transform(X['test'])

    model = None
    if MODEL_TYPE == ModelType.XGBOOST_REGRESSION_MODEL:
        model = XGBRegressor(
            random_state=63145,
            objective="reg:squarederror",
            verbosity=1,
            num_parallel_tree=3,
            multi_strategy="multi_output_tree",
            colsample_bytree=0.8,
            learning_rate=0.015,
            max_depth=9,
            n_estimators=570,
            subsample=0.8,
        )
        model.fit(X['train'], y['train'])
    elif MODEL_TYPE == ModelType.RIDGE_REGRESSION_MODEL:
        alphas = np.arange(0.0, 20.0, 0.05)

        maximum_spearman_correlation, new_model = -np.inf, None
        validation_mae, validation_mse, validation_spearman, validation_pearson = [], [], [], []
        for alpha in alphas:
            model = Ridge(random_state=63145, alpha=alpha)
            model.fit(X['train'], y['train'])

            if SCRIPT_ENVIRONMENT == "development":
                validation_predict_scores = model.predict(X['validation'])

                mse = mean_squared_error(y['validation'], validation_predict_scores)
                mae = mean_absolute_error(y['validation'], validation_predict_scores)
                spearman_corr = spearmanr(y['validation'], validation_predict_scores).correlation
                pearson_corr = pearsonr(y['validation'], validation_predict_scores).correlation

                print(f'[Ridge Regression] For alpha = {alpha} we have, '
                      f'MSE: {mse:.7f}, MAE: {mae:.7f}, S: {spearman_corr:.7f}, P: {pearson_corr:.7f}.')

                validation_mae.append(mae)
                validation_mse.append(mse)
                validation_spearman.append(spearman_corr)
                validation_pearson.append(pearson_corr)

                if spearman_corr > maximum_spearman_correlation:
                    maximum_spearman_correlation = spearman_corr
                    new_model = deepcopy(model)

                    print(f"[Ridge Regression] New model was saved during the training with the parameter alpha = {alpha}")

        if SCRIPT_ENVIRONMENT == "development":
            plt.subplot(2, 2, 1)
            plt.plot(alphas, validation_mae, marker='o')
            plt.title('Mean Absolute Error (MAE)')
            plt.xlabel('Alpha')
            plt.ylabel('MAE')

            # Plot MSE
            plt.subplot(2, 2, 2)
            plt.plot(alphas, validation_mse, marker='o')
            plt.title('Mean Squared Error (MSE)')
            plt.xlabel('Alpha')
            plt.ylabel('MSE')

            # Plot Spearman Correlation
            plt.subplot(2, 2, 3)
            plt.plot(alphas, validation_spearman, marker='o')
            plt.title('Spearman Correlation')
            plt.xlabel('Alpha')
            plt.ylabel('Spearman Correlation')

            # Plot Pearson Correlation
            plt.subplot(2, 2, 4)
            plt.plot(alphas, validation_pearson, marker='o')
            plt.title('Pearson Correlation')
            plt.xlabel('Alpha')
            plt.ylabel('Pearson Correlation')

            plt.tight_layout()
            plt.savefig("validation_metrics_ridge.png", dpi=300)
            plt.show()

        model = new_model
        if model is None or new_model is None:
            if os.path.exists(os.path.join('./', "ridge_regression_best_model.pkl")):
                with open('ridge_regression_best_model.pkl', 'rb') as file:
                    model = pickle.load(file)
            else:
                raise Exception("Model is None, please set the script environment to development"
                                " and train the model first.")
        with open('ridge_regression_best_model.pkl', 'wb') as file:
            pickle.dump(new_model, file)
    else:
        raise Exception("Model couldn't be loaded.")

    if SCRIPT_ENVIRONMENT == "kaggle":
        test_scores = model.predict(X['test'])

        submission = pd.DataFrame({'id': test_dataframe['id'].values, 'score': test_scores})
        submission.to_csv('submission.csv', index=False)
