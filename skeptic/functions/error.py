import numpy as np
from sklearn.metrics import balanced_accuracy_score


def ma_pct_acc(Y, Yh):
    Y = np.array(Y)
    Yh = np.array(Yh)
    return 1 - np.mean(np.abs(Y-Yh)/Y)

# @TODO: Custom impl so that sklearn isn't a dependency
def balanced_acc(Y, Yh):
    return balanced_accuracy_score(Y, Yh)
