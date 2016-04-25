# Script that splits data into folds. The last fold is testing data while the others are training data
#
# Usage:  pickle_train_test_data.py pickled_data number_of_folds
# Example pickle_...py "C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\restaurant_photos_with_labels.pkl" 6
import cPickle
import os
import sys

import pandas as pd
from sklearn.cross_validation import StratifiedKFold

if len(sys.argv) > 2:
    # Read system arguments
    filename = sys.argv[1]
    folds = int(sys.argv[2])

    # Output will be named as the input filename but appending _train.pkl or _test.pkl at the end.
    train_fn = os.path.splitext(filename)[0] + '_train.pkl'
    test_fn = os.path.splitext(filename)[0] + '_test.pkl'

    # Read the pandas dataframe with the JSON data
    df = pd.read_pickle(filename)
    # Change the label column to categorical type, and store the categories so that we know what they mean later
    df['label'] = df['label'].astype('category')
    with open('..\data\categories.pkl', 'wb') as f:
        cPickle.dump(df['label'].cat.categories, f, protocol=cPickle.HIGHEST_PROTOCOL)
    # Change from string to int
    df['label'] = df['label'].cat.codes.values
    # Create folds using a Stratified approach to keep proportions
    skf = StratifiedKFold(df['label'], n_folds=folds)

    # Use only the first configuration (train/test) and store as pickled data frames.
    train_ix, test_ix = next(iter(skf))
    df.loc[train_ix, :].to_pickle(train_fn)
    df.loc[test_ix, :].to_pickle(test_fn)
