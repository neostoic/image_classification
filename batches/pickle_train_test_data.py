import os
import sys
import cPickle

import pandas as pd
from sklearn.cross_validation import StratifiedKFold

# Args:  "C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\restaurant_photos_with_labels.pkl" 6
if len(sys.argv) > 2:
    filename = sys.argv[1]
    folds = int(sys.argv[2])

    train_fn = os.path.splitext(filename)[0] + '_train.pkl'
    test_fn = os.path.splitext(filename)[0] + '_test.pkl'

    df = pd.read_pickle(filename)
    df['label'] = df['label'].astype('category')
    with open('..\data\categories.pkl', 'wb') as f:
        cPickle.dump(df['label'].cat.categories, f, protocol=cPickle.HIGHEST_PROTOCOL)
    df['label'] = df['label'].cat.codes.values
    skf = StratifiedKFold(df['label'], n_folds=folds)

    train_ix, test_ix = next(iter(skf))
    df.loc[train_ix, :].to_pickle(train_fn)
    df.loc[test_ix, :].to_pickle(test_fn)
