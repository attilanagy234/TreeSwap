from math import ceil
from typing import List
import multiprocessing as mp


def create_mini_batches(number_of_small_batches: int, batch: List) -> List[List]:
    batch_size = len(batch)
    items_per_small_batch = ceil(batch_size / number_of_small_batches)
    small_batch_list = []
    for i in range(0, batch_size, items_per_small_batch):
        small_batch_list.append(batch[i:i + items_per_small_batch])

    small_batch_list.extend([[] for _ in range(number_of_small_batches - len(small_batch_list))])

    return small_batch_list

