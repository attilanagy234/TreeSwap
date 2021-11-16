import click
from tqdm import tqdm

from hu_nmt.data_augmentator.augmentators.subject_object_augmentator import \
    SubjectObjectAugmentator
from hu_nmt.data_augmentator.dependency_parsers.english_dependency_parser import \
    EnglishDependencyParser
from hu_nmt.data_augmentator.dependency_parsers.spacy_dependency_parser import \
    SpacyDependencyParser
from hu_nmt.data_augmentator.utils.logger import get_logger
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import \
    DependencyGraphWrapper

log = get_logger(__name__)


@click.command()
@click.argument('eng_data_folder')
@click.argument('hun_data_folder')
@click.argument('augmentation_output_path')
@click.argument('augmented_data_ratio')
@click.option('--output_format', default='tsv', help='Supported output formats: tsv (default), basic')
@click.option('--save_original/--dont_save_original', default=False)
def main(eng_data_folder, hun_data_folder, augmentation_output_path, augmented_data_ratio, output_format, save_original):
    eng_dep_parser = EnglishDependencyParser()
    eng_dep_tree_generator = eng_dep_parser.read_parsed_dep_trees_from_files(eng_data_folder, per_file=True)
    # log.info(f'Number of English sentences used for augmentation: {len(eng_wrappers)}')
    hun_dep_parser = SpacyDependencyParser(lang='hu')
    hun_dep_tree_generator = hun_dep_parser.read_parsed_dep_trees_from_files(hun_data_folder, per_file=True)
    # log.info(f'Number of Hungarian sentences used for augmentation: {len(eng_wrappers)}')

    augmentator = SubjectObjectAugmentator(None, None, augmented_data_ratio, random_seed=15, output_path=augmentation_output_path,
                                           output_format=output_format, save_original=save_original)

    log.info('Reading parsed dependency trees')
    graph_cnt = 0
    with tqdm() as pbar:
        for eng_dep_tree_batch, hun_dep_tree_batch in zip(eng_dep_tree_generator, hun_dep_tree_generator):
            eng_wrapper_batch = [DependencyGraphWrapper(tree) for tree in eng_dep_tree_batch]
            hun_wrapper_batch = [DependencyGraphWrapper(tree) for tree in hun_dep_tree_batch]

            graph_cnt += len(eng_wrapper_batch)

            augmentator.add_augmentable_candidates(hun_wrapper_batch, eng_wrapper_batch)

            pbar.update(len(eng_wrapper_batch))

    log.info(f'Have parsed {graph_cnt} sentence graphs')

    augmentator.augment()


if __name__ == '__main__':
    main()

    # During local testing
    # $1 /Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/eng/parsed_sentences
    # $2 /Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/hu/parsed_sentences
    # $3 /Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/augmented_data
    # $4 15

    # During hu-nmt use
    # /home1/hu-nmt/data/Hunglish2/augmented/dep_trees/en /home1/hu-nmt/data/Hunglish2/augmented/dep_trees/hu /home1/hu-nmt/data/Hunglish2/augmented/generated-0.25 0.25
