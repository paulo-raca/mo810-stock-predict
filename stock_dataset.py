import os
import logging
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger('stock-dataset')

DATASET_DIR = "dataset/dataset-2017-10-11"
STOCK_DIR = f"{DATASET_DIR}/Stocks"
ETF_DIR = f"{DATASET_DIR}/ETFs"

SP500 = pd.read_csv('dataset/s&p500.tsv', sep='\t')['Ticker symbol']
FAVORITE_COMPANIES = ['GOOG', 'AMZN', 'NFLX', 'TSLA', 'FB', 'AAPL', 'INTC', 'QCOM', 'DIS', 'NVDA']
DEFAULT_SYMBOLS = FAVORITE_COMPANIES
DEFAULT_TEST_RATIO = 0.2
DEFAULT_MIN_DATE = '2013-01-01'
DEFAULT_TIMESERIES_LENGTH = 30

def read_datasets(symbols=DEFAULT_SYMBOLS, min_date=DEFAULT_MIN_DATE):
    symbols = set(symbols)
    failed_csvs = []
    datasets = {}
    timeseries_feature_data = {}
    for filename in os.listdir(STOCK_DIR):
        symbol = filename.split('.')[0].upper()
        if not symbol in symbols:
            continue

        raw = pd.read_csv(f"{STOCK_DIR}/{filename}")
        datasets[symbol] = raw[raw.Date >= min_date]

    if failed_csvs:
        logging.warning(f'Failed to read {len(failed_csvs)} CSV files: {", ".join(sorted(failed_csvs))}')

    missed_symbols = symbols - set(datasets.keys())
    if missed_symbols:
        logger.warning(f'Didn\'t find {len(missed_symbols)} symbols: {", ".join(sorted(missed_symbols))}')

    return datasets


def read_time_series(symbols=DEFAULT_SYMBOLS, min_date=DEFAULT_MIN_DATE, timeseries_length=DEFAULT_TIMESERIES_LENGTH):
    return {
        symbol: dataset_to_timeseries(dataset, timeseries_length)
        for (symbol, dataset) in read_datasets(symbols, min_date).items()
    }

def dataset_to_timeseries(dataset, timeseries_length):
    """
    Concatenates `days` sequential values to create a larger feature array.
    """
    parallel_series = [
        dataset[d : len(dataset) + d + 1 - timeseries_length]
        for d in range(timeseries_length)
    ]

    cols = ['Date']
    data = [parallel_series[-1].Date]
    for i in range(len(parallel_series)):
        s = parallel_series[i]
        for col in ['Low', 'High', 'Open', 'Close', 'Volume']:
            cols.append(f'{col}.{i}')
            data.append(s[col])

    return pd.DataFrame(list(zip(*data)), columns=cols)

def concat_data(datasets):
    """
    Concatenates a bunch of tables for different symbols,
    Adding and extra columns 'Symbol' which specifies which table it came from
    """
    df = pd.concat(
        df.assign(Symbol = symbol)
        for symbol, df in datasets.items()
    )
    return df.reindex(['Symbol'] + list(df.columns[:-1]), axis='columns')

def train_test_split_by_date(dataset, test_ratio=DEFAULT_TEST_RATIO):
    trading_dates = sorted(set(dataset.Date))
    train_dates, test_dates = train_test_split(trading_dates, test_size=test_ratio)
    train_dataset = dataset[dataset.Date.isin(set(train_dates))].sample(frac=1).reset_index(drop=True)
    test_dataset = dataset[dataset.Date.isin(set(test_dates))].sample(frac=1).reset_index(drop=True)
    return (train_dataset, test_dataset)

def minibatch_producer(symbols=DEFAULT_SYMBOLS, min_date=DEFAULT_MIN_DATE, timeseries_length=DEFAULT_TIMESERIES_LENGTH, test_ratio=DEFAULT_TEST_RATIO, minibatch_size=100, num_companies=10):
    all_timeseries = concat_data(read_time_series(symbols, min_date, timeseries_length))
    train_timeseries, test_timeseries = train_test_split_by_date(all_timeseries, test_ratio=test_ratio)

    def produce(set=None, minibatch_size=minibatch_size, num_companies=num_companies):
        if set == 'train':
            timeseries = train_timeseries
        elif set == 'test':
            timeseries = test_timeseries
        elif set is None:
            timeseries = all_timeseries
        else:
            raise Exception("Expected set do be 'train', 'test' or None")

        minibatch = []
        for i in range(minibatch_size):
            #print(len(timeseries.Date[10983]))
            date = random.choice(list(timeseries.Date))
            timeseries_at_date = timeseries[timeseries.Date == date]

            samples = timeseries_at_date.sample(n = num_companies, replace = True)
            samples = samples.as_matrix(columns=samples.columns[2:])
            samples = samples.reshape([num_companies, -1, 5])
            samples = samples.transpose((1,0,2))
            minibatch.append(samples)
        return np.stack(minibatch)
    return produce
