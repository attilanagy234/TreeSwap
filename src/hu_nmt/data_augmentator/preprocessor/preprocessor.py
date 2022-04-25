from hu_nmt.data_augmentator.utils.data_helpers import get_config_from_yaml
from hu_nmt.data_augmentator.utils.logger import get_logger

log = get_logger(__name__)


class Preprocessor:
    """
    Receives the entire dataset and returns a preprocessed subsample to augment
    based on predefined criteria
    """
    def __init__(self, source_data_path: str, target_data_path: str, config_path: str, source_output_path: str, target_output_path: str):
        self._source_data_path = source_data_path
        self._target_data_path = target_data_path
        self._config = get_config_from_yaml(config_path)
        self._source_output_path = source_output_path
        self._target_output_path = target_output_path


    def preprocess(self):
        log.info('Starting preprocessing...')
        number_of_lines_saved_to_file = 0
        with open(self._source_data_path) as source_file, \
             open(self._target_data_path) as target_file, \
             open(self._source_output_path, 'w') as source_output_file, \
             open(self._target_output_path, 'w') as target_output_file:

            for i, (source_line, target_line) in enumerate(zip(source_file, target_file)):
                source_sentence, target_sentence = source_line.strip(), target_line.strip()
                source_sentence = self.clean_sentence(source_sentence)
                target_sentence = self.clean_sentence(target_sentence)
                if self.is_good_length(source_sentence, target_sentence):
                    source_output_file.write(source_sentence + '\n')
                    target_output_file.write(target_sentence + '\n')

                    number_of_lines_saved_to_file += 1

        log.info(f'Finished processing sentences. Number of sentences before and after: {i+1} -> {number_of_lines_saved_to_file}')

    def is_good_length(self, source_sentence: str, target_sentence: str) -> bool:
        source_len = len(source_sentence.split())
        target_len = len(target_sentence.split())

        return (source_len > self._config.preprocessor.total_wordcount_min) and \
            (source_len < self._config.preprocessor.total_wordcount_max) and \
            (target_len > self._config.preprocessor.total_wordcount_min) and \
            (target_len < self._config.preprocessor.total_wordcount_max) and \
            (
                    (abs(source_len - target_len) < self._config.preprocessor.wordcount_diff) or
                    (
                            (source_len / target_len < self._config.preprocessor.wordcount_ratio_threshold) and
                            (target_len / source_len < self._config.preprocessor.wordcount_ratio_threshold)
                    )
            )

    @staticmethod
    def clean_sentence(sentence):
        sentence = sentence.replace('\xad', '-')  # replace soft hyphens with normal hyphens
        copy = ""
        while copy != sentence:
            copy = sentence
            if sentence.startswith('"') and sentence.endswith('"'):  # lots of sentences start and end with unnecessary double quotes
                sentence = sentence[1:-1]
            if sentence.startswith("'") and sentence.endswith("'"):
                sentence = sentence[1:-1]
            if sentence.startswith("`") and sentence.endswith("`"):
                sentence = sentence[1:-1]

            if sentence.strip().count("'") == 1 and (sentence.strip().startswith("'") or sentence.strip().endswith("'")):
                sentence.replace("'", "")
            if sentence.count('"') == 1 and (sentence.strip().startswith("'") or sentence.strip().endswith("'")):
                sentence.replace('"', "")
            if sentence.count('`') == 1 and (sentence.strip().startswith("`") or sentence.strip().endswith("`")):
                sentence.replace('`', "")

            if sentence.startswith('-'):
                sentence = sentence[1:]
        return sentence

