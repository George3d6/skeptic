import pandas as pd
import numpy as np
from mindsdb_native import Predictor

from skeptic.functions.error import ma_pct_acc, balanced_acc
from skeptic.functions.splitting import sample_fold_indexes


def correlate(X, Y, nr_folds=9, accuracy_functions=None, Y_type_map=None):
    '''
        ## Summary:
        *Performs a k-fold cross validation in order to determine the predictive strength of a model `M` that tries to infer `Y` from `X`.
        *The model `M` is currently a simple Predictor from mindsdb_native (a tool chosen simply because I am very familiar with it). Ideally the model `M` would be a "good enough" predictive model where performance is a function of what is possible to achieve with SOTA ML and the amount of compute the researcher can allocate for the problem.

        ## Assumptions about the data:
        * If a component (columns) of `Y` is a categorical (kinda the same meaning as discrete with some ***) variable, we care about "balanced" predictive accuracy
        * If a component (columns) of `Y` is a numerical variable, we care about the % error in the form of `Sum(abs(yh-y)/y)`.
        * We use 9-fold cross validation if 9 > nr of datapoints, otherwise we use a `k` equal to the number of datapoints

        Note: Assumptions can be changed by modifying the optional argments


        ## Arguments:
        * X -> pandas.DataFrame of one or more columns
        * Y -> pandas.DataFrame of one or more columns
        * nr_folds -> nr of folds to use for the k-fold cross validation (i.e. the value of `k`)
        * accuracy_functions -> a list of the accuracy functions to use for each component of `Y`

        Return:
        * A number from 0 to 1 indicating an estimate of the predictive power `X` has for determining `Y`
    '''
    np.random.seed(len(X))

    if nr_folds > len(X):
        print('Warning, trying to use {} fold with only {} datapoints, will only be using {} folds instead'.format(nr_folds, len(X), len(X)))
        nr_folds = len(X)

    # Determine data types of the components of `Y`
    if Y_type_map is None:
        Y_type_map = {}
        for column in Y.columns:
            if sum(Y[column].astype(str).str.isnumeric()) == len(Y):
                Y_type_map[column] = 'numeric'
            else:
                Y_type_map[column] = 'categorical'

    if len(set(list(Y_type_map.values()))) > 1:
        print('Warning: multiple types for the components of Y, results might be hard to interpret due to potential accuracy function mixing!')


    if accuracy_functions is None:
        accuracy_functions = {}
        for column in Y.columns:
            if Y_type_map[column] == 'numeric':
                accuracy_functions[column] = ma_pct_acc
            else:
                accuracy_functions[column] = balanced_acc

    if len(set(list(accuracy_functions.values()))) > 1:
        print('Warning: multiple accuracy functions being used, results might be hard to interpret due to accuracy function mixing!')

    fold_indexes = sample_fold_indexes(nr_folds, len(X))

    folds = [(X.iloc[indexes],Y.iloc[indexes]) for indexes in fold_indexes]

    correlations = []
    for i in range(len(folds)):
        fit_df = None
        for ii in range(len(folds)):
            if i != ii:
                df = pd.concat([folds[ii][0],folds[ii][1]], axis=1)
                if fit_df is None:
                    fit_df = df
                else:
                    fit_df = pd.concat([fit_df, df])
            else:
                test_X = folds[ii][0]
                test_Y = folds[ii][1]

        df.reset_index(drop=True, inplace=True)
        test_X.reset_index(drop=True, inplace=True)
        test_Y.reset_index(drop=True, inplace=True)

        M = Predictor(str(np.random.randint(0,pow(2,15))))
        M.quick_learn(from_data=fit_df, to_predict=list(set(Y.columns) - set('Index')))
        predictions = M.quick_predict(when_data=test_X)
        for colum in Y_type_map:
            if Y_type_map[column] == 'numeric':
                predictions[column] = [float(x) for x in predictions[column]]
            else:
                predictions[column] = [str(x) for x in predictions[column]]

        accuracies = []
        for column in Y.columns:
            accuracy = accuracy_functions[column](list(test_Y[column]), predictions[column])
            accuracies.append(accuracy)
        correlations.append(np.mean(accuracies))

    return np.mean(correlations)
