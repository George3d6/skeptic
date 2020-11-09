import pandas as pd

def csv_to_X_Y(file, targets):
    df = pd.read_csv(file)
    X = df.loc[:, df.columns != targets]
    Y = df.loc[:, df.columns == targets]
    return X, Y
