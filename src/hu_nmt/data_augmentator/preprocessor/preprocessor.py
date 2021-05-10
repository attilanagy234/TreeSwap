import pandas as pd
import os
from hu_nmt.data_augmentator.utils.data_helpers import get_config_from_yaml
from hu_nmt.data_augmentator.utils.logger import get_logger
import json

log = get_logger(__name__)


class Preprocessor:
    """
    Receives the entire dataset and returns a preprocessed subsample to augment
    based on predefined criteria
    """
    def __init__(self, eng_data_path, hun_data_path, config_path, output_path):
        eng_sents = []
        with open(eng_data_path) as file:
            for line in file:
                eng_sents.append(line.strip())
        hun_sents = []
        with open(hun_data_path) as file:
            for line in file:
                hun_sents.append(line.strip())
        data_dict = {
            'hun': hun_sents,
            'eng': eng_sents
        }
        self._df = pd.DataFrame(data_dict)
        self._config = get_config_from_yaml(config_path)
        self._output_path = output_path
        log.info('Starting preprocessing...')
        print(f'Preprocessor config: {self._config}')

    def preprocess(self):
        log.info(f'Length of dataframe before filtering: {len(self._df)}')
        log.info('Filtering based on length...')
        self.filter_by_length()
        log.info(f'Length of dataframe after filtering: {len(self._df)}')
        self.clean()
        self.dump_preprocessed_sents_to_files()

    def filter_by_length(self):
        self._df['hun_word_count'] = self._df['hun'].str.split().apply(len)
        self._df['eng_word_count'] = self._df['eng'].str.split().apply(len)
        return self._df[(self._df['hun_word_count'] > self._config.preprocessor.total_wordcount_min) &
                  (self._df['hun_word_count'] < self._config.preprocessor.total_wordcount_max) &
                  (self._df['eng_word_count'] > self._config.preprocessor.total_wordcount_min) &
                  (self._df['eng_word_count'] < self._config.preprocessor.total_wordcount_max) &
                  (
                          ((self._df['hun_word_count'] - self._df['eng_word_count']).abs() < self._config.preprocessor.wordcount_diff) |
                          (
                                  (self._df['hun_word_count'] / self._df['eng_word_count'] < self._config.preprocessor.wordcount_ratio_threshold) &
                                  (self._df['eng_word_count'] / self._df['hun_word_count'] < self._config.preprocessor.wordcount_ratio_threshold)
                          )
                  )
                  ]

    def clean(self):
        self._df['hun'] = self._df['hun'].apply(self.clean_sentence)
        self._df['eng'] = self._df['eng'].apply(self.clean_sentence)

    @staticmethod
    def clean_sentence(sentence):
        sentence = sentence.replace('\xad', '-')  # replace soft hyphens with normal hyphens
        if sentence.startswith('"') and sentence.endswith('"'):  # lots of sentences start and end with unnecessary double quotes
            sentence = sentence[1:-1]
        if sentence.startswith('-'):
            sentence = sentence[1:]
        return sentence

    def dump_preprocessed_sents_to_files(self):
        log.info('Dumping preprocessed sentences to files')
        eng_sents = list(self._df['eng'])
        hun_sents = list(self._df['hun'])
        with open(os.path.join(self._output_path, f'eng_sentences_{len(self._df)}'), 'w+') as file:
            for sent in eng_sents:
                file.write(sent)
                file.write('\n')

        with open(os.path.join(self._output_path, f'hun_sentences_{len(self._df)}'), 'w+') as file:
            for sent in hun_sents:
                file.write(sent)
                file.write('\n')
        log.info(f'Finished saving files at: {self._output_path}')







