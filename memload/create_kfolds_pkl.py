# This script creates several data batches based on the number of folds selected. The last batch is always named test
# Usage: python create_kfolds_pkl.py source folds
import pandas as pd
import os
import sys
import cPickle
from sklearn.cross_validation import StratifiedKFold

# Read python arguments
if len(sys.argv) > 2:
    filename = sys.argv[1]
    folds = int(sys.argv[2])
    pkl_train_file = []

    # Initialize the names for the pickles
    for i in range(1, folds):
        pkl_train_file.append(os.path.splitext(filename)[0] + '_train_batch' + str(i) + '.pkl')
    pkl_train_file.append(os.path.splitext(filename)[0] + '_test.pkl')

    # Read the labels as categories and based on these categories obtain k-folds in such way the proportions are kept.
    df = pd.read_pickle(filename)
    df['label'] = df['label'].astype('category')
    df['label'] = df['label'].cat.codes.values
    skf = StratifiedKFold(df['label'], n_folds=folds)

    # Pickle the training and testing data
    for i, (train, test) in enumerate(skf):
        df.loc[test, :].to_pickle(pkl_train_file[i])

    # For sklearn 0.18
    # from sklearn.model_selection import StratifiedKFold
    # df = pd.read_pickle(filename)
    # df['label'] = df['label'].astype('category')
    # f = open('./data/categories.pkl', 'wb')
    # cPickle.dump(df['label'].cat.categories, f, protocol=cPickle.HIGHEST_PROTOCOL)
    # f.close()
    # df['label'] = df['label'].cat.codes.values
    # skf = StratifiedKFold(n_folds=folds)
    #
    # for i, (train, test) in enumerate(skf.split(df['photo_id'], df['label'])):
    #     df.loc[test, :].to_pickle(pkl_train_file[i])

# Same thing only hard coded
else:
    filename = r'path\to\pickle\file.pkl'
    folds = 6
    pkl_train_file = []

    for i in range(1, folds):
        pkl_train_file.append(os.path.splitext(filename)[0] + '_train_batch' + str(i) + '.pkl')
    pkl_train_file.append(os.path.splitext(filename)[0] + '_test.pkl')

    df = pd.read_pickle(filename)
    skf = StratifiedKFold(df['label'], n_folds=folds)

    for i, (train, test) in enumerate(skf):
        df.loc[test, :].to_pickle(pkl_train_file[i])
