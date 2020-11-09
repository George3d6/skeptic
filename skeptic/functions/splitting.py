import numpy as np


def sample_fold_indexes(nr_folds, nr_datapoints):
    indexes = np.array([x for x in range(nr_datapoints)])
    np.random.shuffle(indexes)
    split_indexes = np.array_split(indexes, nr_folds)
    return split_indexes
