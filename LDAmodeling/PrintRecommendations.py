import json
import logging
import pickle
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


logging.info('Loading image_caption_dataset')
dataset_mon_ami_img = json.load(
   open("./data/mon_ami_recomended_image_dataset2.json"))


dataset=dataset_mon_ami_img['images']


for review in dataset:
    print("Review:")
    print(review['text'])
    print(review['review_topic_info'][1])
    print("Topic words:")
    print(review['review_topic_info'][2])
    print("Photos:")
    print(review['images_info'])
