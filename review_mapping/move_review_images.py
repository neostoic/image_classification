import errno
import json
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


img_dir = r'D:\Yelp\restaurant_photos\\'
target_dir = r'.\results/'
reviews_suggestions = r"mon_ami_gabi_reviews_suggestions_2.json"

dataset = json.load(open(reviews_suggestions, mode='r'))

# {"review_id": "UpKdXO3jEElnqtsWjYxQ2w", "weight": 0.10000000000000002,
#  "text": "Excellent food, great atmosphere, a bit noisy.  $$", "business_id": "4bEjOyTaDG24SY5TxsaUNQ", "top_words": [],
#  "topic": 0, "suggested_images": []}

for idx in range(10):
    if dataset[idx]['suggested_images']:
        output_dir = target_dir + "{}/".format(dataset[idx]['review_id'])
        output_image_dir = target_dir + "{}/images/".format(dataset[idx]['review_id'])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)
        with open(output_dir + 'review.txt', mode='w') as fout:
            fout.write('Top words: ' + ', '.join(dataset[idx]['top_words']) + '\n')
            fout.write(dataset[idx]['text'])
        for image in dataset[idx]['suggested_images']:
            try:
                file_name = image + '.jpg'
                src = img_dir + file_name
                copy_file(src, output_image_dir)
            except:
                print "Error copying file"
