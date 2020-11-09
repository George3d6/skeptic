import numpy as np
from sklearn.metrics import balanced_accuracy_score


def ma_pct_acc(Y, Yh):
    '''
        Implementation of mean absloute percentage accuracy as the inverse of the error defined here: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    '''
    Y = np.array(Y)
    Yh = np.array(Yh)
    return 1 - np.mean(np.abs(Y-Yh)/Y)


def balanced_acc(Y, Yh):
    '''
        Implementation of accuracy as defined here: https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score
        @TODO: Custom impl so that sklearn isn't a dependency
    '''
    return balanced_accuracy_score(Y, Yh)
