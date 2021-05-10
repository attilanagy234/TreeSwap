import click
from hu_nmt.data_augmentator.augmentators.subject_object_augmentator import SubjectObjectAugmentator
from hu_nmt.data_augmentator.dependency_parsers.english_dependency_parser import EnglishDependencyParser
from hu_nmt.data_augmentator.dependency_parsers.hungarian_dependency_parser import HungarianDependencyParser
from hu_nmt.data_augmentator.utils.logger import get_logger

log = get_logger(__name__)


@click.command()
@click.argument('eng_data_folder')
@click.argument('hun_data_folder')
@click.argument('augmentation_output_path')
def main(eng_data_folder, hun_data_folder, augmentation_output_path):
    # TODO: Could add a limit on the number of permutations --> should sample from it
    eng_dep_parser = EnglishDependencyParser()
    eng_wrappers = eng_dep_parser.get_graph_wrappers_from_files(eng_data_folder)
    log.info(f'number of English sentences used for augmentation: {len(eng_wrappers)}')
    hun_dep_parser = HungarianDependencyParser()
    hun_wrappers = hun_dep_parser.get_graph_wrappers_from_files(hun_data_folder)
    log.info(f'number of Hungarian sentences used for augmentation: {len(eng_wrappers)}')
    augmentator = SubjectObjectAugmentator(eng_wrappers, hun_wrappers)
    augmentator.augment(random_seed=15, output_path=augmentation_output_path)


if __name__ == '__main__':
    main()

    # During local testing
    # $1 /Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/eng/parsed_sentences
    # $2 /Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/hu/parsed_sentences
    # $3 /Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/augmented_data
    # $4 15
