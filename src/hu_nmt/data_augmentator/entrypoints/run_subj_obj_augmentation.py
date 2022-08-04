import click
from tqdm import tqdm

from hu_nmt.data_augmentator.augmentators.graph_based_augmentator import GraphBasedAugmentator
from hu_nmt.data_augmentator.augmentators.subject_object_augmentator import SubjectObjectAugmentator
from hu_nmt.data_augmentator.dependency_parsers.dependency_parser_factory import DependencyParserFactory
from hu_nmt.data_augmentator.dependency_parsers.stanza_dependency_parser import StanzaDependencyParser
from hu_nmt.data_augmentator.dependency_parsers.spacy_dependency_parser import SpacyDependencyParser
from hu_nmt.data_augmentator.utils.logger import get_logger
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper
from hu_nmt.data_augmentator.filters.bleu_filter import BleuFilter

log = get_logger(__name__)


@click.command()
@click.argument('src_language')
@click.argument('tgt_language')
@click.argument('src_data_folder')
@click.argument('tgt_data_folder')
@click.argument('augmentation_output_path')
@click.argument('augmented_data_ratio')
@click.option('--use_filters', is_flag=True, default=False, help='Use filters after the augmentation')
@click.option('--filter_quantile', default=0.0,
              help='Quantile to use when filtering with cosine similarity. (Throw away anything below this quantile.)')
@click.option('--src_model_path', default='', help='Path to model to translate the source sentences with')
@click.option('--tgt_model_path', default='', help='Path to model to translate the target sentences with')
@click.option('--sp_model_path', default='', help='Sentencepiece model path')
@click.option('--filter_batch_size', default=512, help='Batch size for the translations in the filter.')
@click.option('--output_format', default='basic', help='Supported output formats: basic (default), tsv')
@click.option('--save_original/--dont_save_original', default=False)
@click.option('--separate_augmentation', default=False)
@click.option('--filter_same_ancestor', default=True)
@click.option('--filter_same_pos_tag', default=True)
@click.option('--filter_for_noun_tags', default=False)
@click.option('--augmentation_type', default='base', type=click.Choice(['base', 'ged', 'edge_mapper']))
@click.option('--threshold', default=0.5)
def main(src_language, tgt_language, src_data_folder, tgt_data_folder, augmentation_output_path,
         augmented_data_ratio, use_filters, filter_quantile, src_model_path, tgt_model_path, sp_model_path,
         filter_batch_size, output_format, save_original, separate_augmentation, filter_same_ancestor,
         filter_same_pos_tag, filter_for_noun_tags, augmentation_type, threshold):

    src_parser = DependencyParserFactory.get_dependency_parser(src_language)
    tgt_parser = DependencyParserFactory.get_dependency_parser(tgt_language)

    src_dep_tree_generator = src_parser.read_parsed_dep_trees_from_files(src_data_folder, per_file=True)
    # log.info(f'Number of source sentences used for augmentation: {len(eng_wrappers)}')
    tgt_dep_tree_generator = tgt_parser.read_parsed_dep_trees_from_files(tgt_data_folder, per_file=True)
    # log.info(f'Number of target sentences used for augmentation: {len(eng_wrappers)}')

    filters = []
    if use_filters:
        filters.append(
            BleuFilter(filter_quantile, src_model_path, tgt_model_path, sp_model_path, tgt_language, filter_batch_size))
    augmentator = SubjectObjectAugmentator(None, None, augmented_data_ratio, random_seed=15, filters=filters,
                                           output_path=augmentation_output_path, output_format=output_format,
                                           save_original=save_original, separate_augmentation=separate_augmentation,
                                           filter_nsub_and_obj_have_same_ancestor=filter_same_ancestor,
                                           filter_same_pos_tag=filter_same_pos_tag,
                                           filter_for_noun_tags=filter_for_noun_tags)


    if augmentation_type == 'ged' or augmentation_type == 'edge_mapper':
        augmentator = GraphBasedAugmentator(src_language, tgt_language, threshold, None, None, augmented_data_ratio,
                                            random_seed=15, filters=filters, output_path=augmentation_output_path,
                                            output_format=output_format, save_original=save_original,
                                            separate_augmentation=separate_augmentation,
                                            similarity_type=augmentation_type)
    else:
        augmentator = SubjectObjectAugmentator(None, None, augmented_data_ratio, random_seed=15, filters=filters,
                                               output_path=augmentation_output_path, output_format=output_format,
                                               save_original=save_original, separate_augmentation=separate_augmentation)
    log.info('Reading parsed dependency trees')
    graph_cnt = 0
    with tqdm() as pbar:
        for src_dep_tree_batch, tgt_dep_tree_batch in zip(src_dep_tree_generator, tgt_dep_tree_generator):
            src_wrapper_batch = [DependencyGraphWrapper(tree) for tree in src_dep_tree_batch]
            tgt_wrapper_batch = [DependencyGraphWrapper(tree) for tree in tgt_dep_tree_batch]

            graph_cnt += len(src_wrapper_batch)

            augmentator.add_augmentable_candidates(tgt_wrapper_batch, src_wrapper_batch)

            pbar.update(len(src_wrapper_batch))

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
    #  /home1/hu-nmt/data/Hunglish2/augmented/dep_trees/en /home1/hu-nmt/data/Hunglish2/augmented/dep_trees/hu /home1/hu-nmt/data/Hunglish2/augmented/generated-0.25 0.25
