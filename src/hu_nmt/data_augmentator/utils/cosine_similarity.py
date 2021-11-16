import pathlib
import re
import shutil

import fasttext
import fasttext.util
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_model(lang: str):
    # make sure there is a resource folder
    current_file_directory = pathlib.Path(__file__).parent.resolve()
    resource_directory = current_file_directory.parent / 'resources'
    resource_directory.mkdir(parents=True, exist_ok=True)

    # download model file if needed
    language_model_file = resource_directory / ('cc.' + lang + '.300.bin')
    if not language_model_file.is_file():
        # download model
        fasttext.util.download_model(lang, if_exists='ignore')

        # move model to resource folder
        current_working_directory = pathlib.Path().resolve()
        downloaded_model_file = current_working_directory / ('cc.' + lang + '.300.bin')
        shutil.copy(downloaded_model_file, language_model_file)
        
        # remove downloaded files
        downloaded_model_file.unlink()
        downloaded_compressed_model_file = current_working_directory / ('cc.' + lang + '.300.bin.gz')
        downloaded_compressed_model_file.unlink()

    # load the model and return it
    return fasttext.load_model(str(language_model_file))

def get_word_average_sentence_vector(sentence, ft_model):
    words = re.sub("[^\w]", " ",  sentence).split()
    word_vectors = np.array([ft_model.get_word_vector(word) for word in words])
    return word_vectors.sum(axis=(0,)) / len(words)

def get_word_average_cos_similarity(sent_1, sent_2, ft_model):
    sent_vec_1 = get_word_average_sentence_vector(sent_1, ft_model)
    sent_vec_2 = get_word_average_sentence_vector(sent_2, ft_model)
    return cosine_similarity(sent_vec_1.reshape(1, -1), sent_vec_2.reshape(1, -1))

def get_sentence_embedding_cos_similarity(sent_1, sent_2, ft_model):
    sent_vec_1 = ft_model.get_sentence_vector(sent_1)
    sent_vec_2 = ft_model.get_sentence_vector(sent_2)
    return cosine_similarity(sent_vec_1.reshape(1, -1), sent_vec_2.reshape(1, -1))