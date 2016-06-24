import os
import sys

import pandas as pd
from sklearn.cross_validation import train_test_split

if len(sys.argv) > 2:
    filename = sys.argv[1]
    pkl_train_file = os.path.splitext(filename)[0] + '_train.pkl'
    pkl_test_file = os.path.splitext(filename)[0] + '_test.pkl'
    in_test_size = float(sys.argv[2])

    df = pd.read_pickle(filename)
    train, test = train_test_split(df, test_size=in_test_size)
    train.to_pickle(pkl_train_file)
    test.to_pickle(pkl_test_file)

else:
    filename = 'path\to\pickle\file.pkl'
    pkl_train_file = os.path.splitext(filename)[0] + '_train.pkl'
    pkl_test_file = os.path.splitext(filename)[0] + '_test.pkl'
    in_test_size = 0.3

    df = pd.read_pickle(filename)
    train, test = train_test_split(df, test_size=in_test_size)
    train.to_pickle(pkl_train_file)
    test.to_pickle(pkl_test_file)
