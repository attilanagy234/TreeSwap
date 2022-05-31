import os.path
from typing import Tuple, Set

import click

from tqdm import tqdm
from hu_nmt.data_augmentator.dependency_parsers.dependency_parser_factory import DependencyParserFactory
from networkx import graph_edit_distance
import numpy as np


def node_match(n1, n2):
    return n1['postag'] == n2['postag']


def edge_match(e1, e2):
    return e1['dep'].split(':')[0].lower() == e2['dep'].split(':')[0].lower()


def node_del_or_add(n):
    if n['postag'] == 'PUNCT':
        return 0
    else:
        return 1


def edge_del_or_add(n):
    if n['dep'] == 'punct':
        return 0
    else:
        return 1


def node_subst_cost(n1, n2):
    if node_match(n1, n2):
        return 0
    else:
        return 1


def edge_subs_cost(e1, e2):
    if edge_match(e1, e2):
        return 0
    else:
        return 1


@click.command()
@click.argument('src_lang_code')
@click.argument('tgt_lang_code')
@click.argument('src_file_name')
@click.argument('tgt_file_name')
@click.argument('count')
@click.argument('result_dir')
@click.option('--rnd', is_flag=True, default=False)
@click.option('--diff', is_flag=True, default=False)
def main(src_lang_code, tgt_lang_code, src_file_name, tgt_file_name, count, result_dir, rnd, diff):
    if diff:
        rnd = True

    src_dep_parser = DependencyParserFactory.get_dependency_parser(src_lang_code)
    tgt_dep_parser = DependencyParserFactory.get_dependency_parser(tgt_lang_code)

    with open(src_file_name, 'r') as src_file:
        src_sentences = src_file.readlines()
        src_sentences = [s.rstrip() for s in src_sentences]
    with open(tgt_file_name, 'r') as tgt_file:
        tgt_sentences = tgt_file.readlines()
        tgt_sentences = [s.rstrip() for s in tgt_sentences]

    index_pairs: Set[Tuple[int, int]] = set()
    if rnd:
        while len(index_pairs) < int(count):
            random_index_pair = np.random.choice(len(src_sentences), 2, replace=False)
            if random_index_pair[0] != random_index_pair[1]:
                index_pairs.add(tuple(random_index_pair))
    else:
        index_pairs = set((i, i) for i in range(len(src_sentences)))

    if not diff:
        results = ['src_idx,tgt_idx,src_size,tgt_size,max_dist,result,norm_dist']
        for i, j in tqdm(index_pairs):
            src_sent = src_sentences[i]
            tgt_sent = tgt_sentences[j]
            try:
                src_wrapper = src_dep_parser.sentence_to_graph_wrapper(src_sent)
                tgt_wrapper = tgt_dep_parser.sentence_to_graph_wrapper(tgt_sent)

                if len(src_wrapper.graph.nodes) == 0 or len(tgt_wrapper.graph.nodes) == 0:
                    continue

                result = graph_edit_distance(src_wrapper.graph, tgt_wrapper.graph, node_match, edge_match,
                                             node_subst_cost=node_subst_cost, node_del_cost=node_del_or_add,
                                             node_ins_cost=node_del_or_add, edge_subst_cost=edge_subs_cost,
                                             edge_del_cost=edge_del_or_add, edge_ins_cost=edge_del_or_add,
                                             roots=(src_wrapper.get_root(), tgt_wrapper.get_root()), upper_bound=None,
                                             timeout=10)

                max_dist = len(src_wrapper.graph.nodes) * 2 - 2 + 2 * len(tgt_wrapper.graph.nodes) - 2
                norm_dist = float(max_dist - result) / float(max_dist)
                results.append(f'{i},{j},{len(src_wrapper.graph.nodes)},{len(tgt_wrapper.graph.nodes)},{max_dist},'
                               f'{result},{norm_dist}')

            except Exception as e:
                print(e)
        with open(os.path.join(result_dir, f'{src_lang_code}-{tgt_lang_code}-rnd-{rnd}.csv'), 'w') as f:
            f.write('\n'.join(results))
    else:
        results = []
        for i, j in tqdm(index_pairs):
            src_sent1 = src_sentences[i]
            src_sent2 = src_sentences[j]
            tgt_sent1 = tgt_sentences[i]
            tgt_sent2 = tgt_sentences[j]
            try:
                src_wrapper1 = src_dep_parser.sentence_to_graph_wrapper(src_sent1)
                src_wrapper2 = src_dep_parser.sentence_to_graph_wrapper(src_sent2)
                tgt_wrapper1 = tgt_dep_parser.sentence_to_graph_wrapper(tgt_sent1)
                tgt_wrapper2 = tgt_dep_parser.sentence_to_graph_wrapper(tgt_sent2)

                if len(src_wrapper1.graph.nodes) == 0 or len(tgt_wrapper1.graph.nodes) == 0 or \
                        len(src_wrapper2.graph.nodes) == 0 or len(tgt_wrapper2.graph.nodes) == 0:
                    continue

                src_result = graph_edit_distance(src_wrapper1.graph, src_wrapper2.graph, node_match, edge_match,
                                                 node_subst_cost=node_subst_cost, node_del_cost=node_del_or_add,
                                                 node_ins_cost=node_del_or_add, edge_subst_cost=edge_subs_cost,
                                                 edge_del_cost=edge_del_or_add, edge_ins_cost=edge_del_or_add,
                                                 roots=(src_wrapper1.get_root(), src_wrapper2.get_root()),
                                                 upper_bound=None,
                                                 timeout=10)

                tgt_result = graph_edit_distance(tgt_wrapper1.graph, tgt_wrapper2.graph, node_match, edge_match,
                                                 node_subst_cost=node_subst_cost, node_del_cost=node_del_or_add,
                                                 node_ins_cost=node_del_or_add, edge_subst_cost=edge_subs_cost,
                                                 edge_del_cost=edge_del_or_add, edge_ins_cost=edge_del_or_add,
                                                 roots=(tgt_wrapper1.get_root(), tgt_wrapper2.get_root()),
                                                 upper_bound=None,
                                                 timeout=10)

                src_max_dist = len(src_wrapper1.graph.nodes) * 2 - 2 + 2 * len(src_wrapper2.graph.nodes) - 2
                src_norm_dist = float(src_max_dist - src_result) / float(src_max_dist)

                tgt_max_dist = len(tgt_wrapper1.graph.nodes) * 2 - 2 + 2 * len(tgt_wrapper2.graph.nodes) - 2
                tgt_norm_dist = float(tgt_max_dist - tgt_result) / float(tgt_max_dist)

                results.append(f'{i},{j},{src_result},{tgt_result},{src_norm_dist},{tgt_norm_dist},'
                               f'{abs(src_result - tgt_result)},{abs(src_norm_dist - tgt_norm_dist)}')

            except Exception as e:
                print(e)
        with open(os.path.join(result_dir, f'{src_lang_code}-{tgt_lang_code}-rnd-{rnd}-diff.csv'), 'w') as f:
            f.write('\n'.join(results))


if __name__ == '__main__':
    main()
