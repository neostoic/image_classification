##Captioning scripts

This folder contains all of the scripts related to captioning, including some scripts that wre required to reformat the
data so that it was compatible with the model. Scripts that do not have a usage defined are mostly helper scripts
called from other scripts or have hard-coded parameters.

### Structure

In this section, we describe the functionality of each script:
* `__init__.py`: This file is empty and it is only included as a requirement for all packages.
* `add_business_id.py`: Helper script to add the business ID to the JSON files since this was not done initially.
* `add_predicted_labels.py`: This script uses a CNN model to obtain the predicted label for all of the images in the
dataset and creates a new JSON file that contains the predicted label.
* `bleu_score.py`: Computes the BLEU-n score for the captioning results.
* `captioning.py`: This script uses the CNN model and the LSTM model trained previously to obtain the predicted captions
for all of the images in the dataset and creates a new JSON file that contains the predicted captions.
* `jsonToSqlite.py`: This script takes as input the JSON file with the predicted labels/captions and stores the records
in a sqlite DB.
* `preprocess_db.py`: This script changes the JSON format so that it is compatible with the captioning algorithm, changing
it to a dictionary. The format is very similar to the one used by MS COCO dataset. It also tokenizes each caption and
splits the images based on which dataset they belong to: train/test/validation.
* `preprocess_images_caption.py`: This script uses a CNN model to obtain image features for all of the images in the
Yelp dataset which will be used by the captioning algorithm.
* `train_rnn.py`: This script creates and trains the LSTM model used for captioning. It takes as input the CNN features
and the captions.