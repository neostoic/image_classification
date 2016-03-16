import pandas as pd
import os
import sys
import cPickle
from sklearn.model_selection import StratifiedKFold

# python create_kfolds_pkl.py "C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\restaurant_photos_with_labels.pkl" 6
if len(sys.argv) > 2:
    filename = sys.argv[1]
    folds = int(sys.argv[2])
    pkl_train_file = []

    for i in range(1, folds):
        pkl_train_file.append(os.path.splitext(filename)[0] + '_train_batch' + str(i) + '.pkl')
    pkl_train_file.append(os.path.splitext(filename)[0] + '_test.pkl')

    df = pd.read_pickle(filename)
    df['label'] = df['label'].astype('category')
    f = open('categories.pkl', 'wb')
    cPickle.dump(df['label'].cat.categories, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    df['label'] = df['label'].cat.codes.values
    skf = StratifiedKFold(n_folds=folds)

    for i, (train, test) in enumerate(skf.split(df['photo_id'], df['label'])):
        df.loc[test, :].to_pickle(pkl_train_file[i])

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
