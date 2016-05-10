import errno
import os
import shutil


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def copy_file(src, dest):
    try:
        shutil.copy(src, dest)
    # eg. src and dest are the same file
    except shutil.Error as e:
        print('Error: %s' % e)
    # eg. source or destination doesn't exist
    except IOError as e:
        print('Error: %s' % e.strerror)


photo_ids = [[1566, 5245, 5245, 8017, 14968, 16086, 24410, 24410, 27902, 29787, 30214, 39011, 40364, 41139, 44957, 47209,
            52064, 52138, 53841, 54683, 60205],
           [170, 170, 170, 3053, 3053, 9473, 9473, 10173, 10173, 10967, 10967, 11353, 11353, 11353, 12036, 13205,
            13232, 13232, 13624, 13624, 18008, 18749, 18749, 18749, 18804, 20692, 20692, 22401, 22401, 22401, 25902,
            25902, 28882, 28882, 28882, 30644, 32052, 32052, 32052, 32940, 48495, 48495, 53113, 55810, 55810],
           [14610, 18674, 18674, 32643, 39011, 53587, 60337],
           [5245, 5245, 12369, 12369, 14610, 14968, 14968, 14968, 14968, 14968, 14968, 14968, 14968, 16086,
            16086, 18674, 24410, 24410, 27902, 27902, 29787, 29787, 29787, 29787, 29787, 29787, 29787, 29787,
            30214, 30214, 30214, 30214, 30214, 30214, 30214, 30214, 32643, 39011, 39011, 39011, 39011, 39011, 39011,
            39011, 39011, 39011, 41139, 41139, 52064, 52064, 52064, 52064, 52064, 52064, 52064, 52064, 53587, 53841,
            53841, 54683, 54683, 54683, 54683, 54683, 54683, 54683, 54683, 60205, 60205, 60337]]

train_img_dir = r'D:\Yelp\caption_dataset\train\\'
test_img_dir = r'D:\Yelp\caption_dataset\test\\'
val_img_dir = r'D:\Yelp\caption_dataset\val\\'
target_dir = r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\gensim\CaptionsWithTopics\images\\'

for idx, review in enumerate(photo_ids):
    dst = mkdir_p(target_dir + 'review_{0}'.format(idx))
    for photo_id in review:
        file_name = str(photo_id).zfill(6) + '.jpg'
        src = train_img_dir + file_name
        dst = target_dir + 'review_{0}'.format(idx) + '\\' + file_name
        copy_file(src, dst)
        src = test_img_dir + file_name
        copy_file(src, dst)
        src = val_img_dir + file_name
        copy_file(src, dst)
