from pathlib import PurePath
from skeptic.interface.correlation import correlate
from skeptic.interface.utilities import csv_to_X_Y

data_dir = PurePath(
    PurePath(__file__).parent,
    '..',
    'data')


def test_common():
    '''
        Using a csv file with various metrics about a country and it's "Human Development Index" (hdi), find the correlation between those metrics and the hdi.
    '''
    X, Y = csv_to_X_Y(PurePath(data_dir,'hdi.csv'), 'Development Index')
    correlation = correlate(X, Y)
    print(f'Got a correlation of: {correlation}')

test_common()
