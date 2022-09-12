import multiprocessing as mp
import os.path
from itertools import islice

from sacremoses import MosesPunctNormalizer
from tqdm import tqdm

from hu_nmt.data_augmentator.base.nlp_pipeline_base import NlpPipelineBase
from hu_nmt.data_augmentator.dependency_parsers.nlp_pipeline_factory import NlpPipelineFactory
from hu_nmt.data_augmentator.preprocessor.language_detector import LanguageDetector
from hu_nmt.data_augmentator.utils.data_helpers import get_config_from_yaml
from hu_nmt.data_augmentator.utils.logger import get_logger
from hu_nmt.data_augmentator.utils.preprocessing import create_mini_batches

log = get_logger(__name__)


class Preprocessor:
    """
    Receives the entire dataset and returns a preprocessed subsample to augment
    based on predefined criteria
    """

    def __init__(self, source_data_path: str, target_data_path: str, config_path: str, source_output_path: str,
                 target_output_path: str, dep_tree_output_path: str = None):
        self._source_data_path = source_data_path
        self._target_data_path = target_data_path
        self._config = get_config_from_yaml(config_path)
        self._source_output_path = source_output_path
        self._target_output_path = target_output_path
        self._dep_tree_output_path = dep_tree_output_path
        self.moses_punct_normalizer_src = MosesPunctNormalizer(lang=self._config.preprocessor.source_language)
        self.moses_punct_normalizer_tgt = MosesPunctNormalizer(lang=self._config.preprocessor.target_language)
        self.skip_batches = 0

    def preprocess(self):
        # init tokenizers and lang detect model
        self.source_tokenizer = NlpPipelineFactory.get_tokenizer(self._config.preprocessor.source_language)
        self.target_tokenizer = NlpPipelineFactory.get_tokenizer(self._config.preprocessor.target_language)
        self.langdetect = LanguageDetector(self._config.preprocessor.langdetect_model_path)

        log.info('Starting preprocessing...')
        number_of_lines_saved_to_file = 0
        with open(self._source_data_path) as source_file, \
                open(self._target_data_path) as target_file, \
                open(self._source_output_path, 'w') as source_output_file, \
                open(self._target_output_path, 'w') as target_output_file:

            for i, (source_line, target_line) in tqdm(enumerate(zip(source_file, target_file))):
                source_sentence, target_sentence = source_line.strip(), target_line.strip()
                source_sentence = self.clean_sentence(source_sentence)
                source_sentence = self.moses_punct_normalizer_src.normalize(source_sentence)

                target_sentence = self.clean_sentence(target_sentence)
                target_sentence = self.moses_punct_normalizer_src.normalize(target_sentence)

                source_doc = self.source_tokenizer.tokenize(source_sentence)
                target_doc = self.target_tokenizer.tokenize(target_sentence)

                src_word_count = self.source_tokenizer.count_tokens_from_doc(source_doc)
                tgt_word_count = self.target_tokenizer.count_tokens_from_doc(target_doc)

                if self.is_good_length(src_word_count, tgt_word_count) and \
                        self.is_correct_language(source_sentence, target_sentence) and \
                        self._contains_one_sentence(source_doc, target_doc):
                    source_output_file.write(source_sentence + '\n')
                    target_output_file.write(target_sentence + '\n')

                    number_of_lines_saved_to_file += 1

        log.info(
            f'Finished processing sentences. Number of sentences before and after: {i + 1} -> {number_of_lines_saved_to_file}')

    def preprocess_and_precompute(self):
        log.info('Starting preprocessing...')

        self.skip_batches = self._find_num_of_preprocessed_batches()
        if self.skip_batches > 0:
            log.info(f'Found previous run, skipping {self.skip_batches} batches and continuing preprocessing.')

        number_of_lines_saved_to_file = 0
        all_lines = 0
        file_idx = self.skip_batches + 1

        preprocess_output_path = os.path.dirname(self._source_output_path)
        if not os.path.exists(preprocess_output_path):
            os.makedirs(preprocess_output_path)

        line_batch_generator = self._get_file_line_batch_generator(self._source_data_path, self._target_data_path,
                                                                   self._config.preprocessor.batch_size)

        if self._config.preprocessor.use_multiprocessing:
            process_count = self._config.preprocessor.process_count
            for line_batch in line_batch_generator:
                mini_batches_of_sentences = create_mini_batches(process_count, line_batch)
                process_batches = [mini_batch for mini_batch in mini_batches_of_sentences]

                # map to processes
                log.info('Mapping sentences to processes')
                proc_pool = mp.get_context('spawn').Pool(process_count)
                list_of_results = proc_pool.map(self._init_models_and_filter_batch, process_batches)
                proc_pool.close()
                proc_pool.join()

                # unpack results
                log.info('Unpacking results')
                src_list_of_dep_rel_lists = []
                tgt_list_of_dep_rel_lists = []
                src_list_of_sentences = []
                tgt_list_of_sentences = []

                for sents, dep_rel_lists in list_of_results:
                    src_sents, tgt_sents = zip(*sents)
                    src_dep_rel_lists, tgt_dep_rel_lists = zip(*dep_rel_lists)
                    src_list_of_dep_rel_lists.extend(src_dep_rel_lists)
                    tgt_list_of_dep_rel_lists.extend(tgt_dep_rel_lists)
                    src_list_of_sentences.extend(src_sents)
                    tgt_list_of_sentences.extend(tgt_sents)

                self.write_preprocessed_sentences_to_files(src_list_of_sentences, tgt_list_of_sentences,
                                                           src_list_of_dep_rel_lists, tgt_list_of_dep_rel_lists,
                                                           file_idx)

                file_idx += 1
                number_of_lines_saved_to_file += len(src_list_of_sentences)
                all_lines += len(line_batch)

        else:
            self._init_models()
            for line_batch in line_batch_generator:
                sentences, dep_rel_lists = self._filter_batch(line_batch)
                src_sents, tgt_sents = zip(*sentences)
                src_dep_rel_lists, tgt_dep_rel_lists = zip(*dep_rel_lists)

                self.write_preprocessed_sentences_to_files(src_sents, tgt_sents,
                                                           src_dep_rel_lists, tgt_dep_rel_lists,
                                                           file_idx)
                file_idx += 1
                number_of_lines_saved_to_file += len(src_sents)
                all_lines += len(line_batch)

        log.info(
            f'Finished processing sentences. Number of sentences before and after: {all_lines} -> {number_of_lines_saved_to_file}')

    def _get_file_line_batch_generator(self, src_file, tgt_file, batch_size):
        with open(src_file, 'r') as src, open(tgt_file, 'r') as tgt:
            src_iter = iter(lambda: tuple(islice(src, batch_size)), ())
            tgt_iter = iter(lambda: tuple(islice(tgt, batch_size)), ())

            for i, (src_lines, tgt_lines) in enumerate(zip(src_iter, tgt_iter)):
                if i >= self.skip_batches:
                    yield list(zip(src_lines, tgt_lines))

    def _init_models(self):
        self.source_parser = NlpPipelineFactory.get_dependency_parser(self._config.preprocessor.source_language)
        self.target_parser = NlpPipelineFactory.get_dependency_parser(self._config.preprocessor.target_language)
        self.langdetect = LanguageDetector(self._config.preprocessor.langdetect_model_path)

    def _init_models_and_filter_batch(self, process_batch):
        # init models separately for processes
        self._init_models()
        return self._filter_batch(process_batch)

    def _filter_batch(self, process_batch):
        preprocessed_sentences = []
        list_of_dep_rel_list = []

        number_of_lines_saved_to_file = 0

        for i, (source_line, target_line) in tqdm(enumerate(process_batch)):
            source_sentence, target_sentence = source_line.strip(), target_line.strip()
            source_sentence = self.clean_sentence(source_sentence)
            source_sentence = self.moses_punct_normalizer_src.normalize(source_sentence)

            target_sentence = self.clean_sentence(target_sentence)
            target_sentence = self.moses_punct_normalizer_src.normalize(target_sentence)

            if self.is_correct_language(source_sentence, target_sentence) and source_sentence and target_sentence:
                source_dep_rel_list = self.source_parser.sentence_to_node_relationship_list(
                    self.source_parser.nlp_pipeline, source_sentence)
                target_dep_rel_list = self.target_parser.sentence_to_node_relationship_list(
                    self.target_parser.nlp_pipeline, target_sentence)

                source_dep_tree = self.source_parser.node_relationship_list_to_dep_parse_tree(source_dep_rel_list)
                target_dep_tree = self.target_parser.node_relationship_list_to_dep_parse_tree(target_dep_rel_list)

                source_word_count = self.source_parser.count_tokens_from_graph(source_dep_tree)
                target_word_count = self.target_parser.count_tokens_from_graph(target_dep_tree)

                if self.is_good_length(source_word_count, target_word_count):
                    preprocessed_sentences.append((source_sentence, target_sentence))
                    list_of_dep_rel_list.append((source_dep_rel_list, target_dep_rel_list))

                    number_of_lines_saved_to_file += 1
        return preprocessed_sentences, list_of_dep_rel_list

    def write_preprocessed_sentences_to_files(self, src_sents, tgt_sents, src_dep_rel_lists, tgt_dep_rel_lists,
                                              file_idx):
        src_dep_tree_output = os.path.join(self._dep_tree_output_path, self._config.preprocessor.source_language)
        tgt_dep_tree_output = os.path.join(self._dep_tree_output_path, self._config.preprocessor.target_language)

        with open(self._source_output_path, 'a+') as source_output_file:
            source_output_file.write('\n'.join(src_sents) + '\n')

        with open(self._target_output_path, 'a+') as target_output_file:
            target_output_file.write('\n'.join(tgt_sents) + '\n')

        NlpPipelineBase.write_dep_graphs_to_file(src_dep_tree_output, file_idx,
                                                 src_dep_rel_lists)
        NlpPipelineBase.write_dep_graphs_to_file(tgt_dep_tree_output, file_idx,
                                                 tgt_dep_rel_lists)

    def is_good_length(self, source_word_count, target_word_count) -> bool:
        return self._is_good_word_count(source_word_count) and self._is_good_word_count(target_word_count) and \
               self._is_good_ratio(source_word_count, target_word_count)

    def is_correct_language(self, source_sentence, target_sentence) -> bool:
        return (self.langdetect.predict(source_sentence) == self._config.preprocessor.source_language) and \
               (self.langdetect.predict(target_sentence) == self._config.preprocessor.target_language)

    def _is_good_word_count(self, length):
        return (length > self._config.preprocessor.total_wordcount_min) and \
               (length < self._config.preprocessor.total_wordcount_max)

    def _is_good_ratio(self, source_len, target_len):
        return (abs(source_len - target_len) < self._config.preprocessor.wordcount_diff) or \
               (
                       (source_len / target_len < self._config.preprocessor.wordcount_ratio_threshold) and
                       (target_len / source_len < self._config.preprocessor.wordcount_ratio_threshold)
               )

    def _contains_one_sentence(self, source_doc, target_doc) -> bool:
        source_sentence_count = self.source_tokenizer.count_sentences(source_doc)
        target_sentence_count = self.target_tokenizer.count_sentences(target_doc)
        return source_sentence_count == target_sentence_count == 1

    def clean_sentence(self, sentence):
        sentence = sentence.replace('\xad', '-')  # replace soft hyphens with normal hyphens
        copy = ""
        while copy != sentence:
            copy = sentence
            # lots of sentences start and end with unnecessary double quotes
            if sentence.startswith('"') and sentence.endswith('"'):
                sentence = sentence[1:-1]
            if sentence.startswith("'") and sentence.endswith("'"):
                sentence = sentence[1:-1]
            if sentence.startswith("`") and sentence.endswith("`"):
                sentence = sentence[1:-1]

            if sentence.count("'") == 1 and (sentence.strip().startswith("'") or sentence.strip().endswith("'")):
                sentence = sentence.replace("'", "")
            if sentence.count('"') == 1 and (sentence.strip().startswith('"') or sentence.strip().endswith('"')):
                sentence = sentence.replace('"', "")
            if sentence.count('`') == 1 and (sentence.strip().startswith("`") or sentence.strip().endswith("`")):
                sentence = sentence.replace('`', "")

            if sentence.startswith('-'):
                sentence = sentence[1:]
        return sentence

    def _find_num_of_preprocessed_batches(self) -> int:
        src_dep_tree_output = os.path.join(self._dep_tree_output_path, self._config.preprocessor.source_language)
        tsv_files = os.listdir(src_dep_tree_output)
        max_tsv_number = max(map(lambda f: int(f.split('.')[0]), tsv_files), default=0)
        return max_tsv_number
