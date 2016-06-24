#Enhancing Yelp Reviews through Data Mining

We present a data-mining-based framework that enhances restaurant Yelp reviews by 
suggesting images uploaded by other users which are relevant to a particular review. 
The framework developed consists of three main components: 
* a Convolutional Neural Network image classifier used to predict the label of each 
new image
* a Long Short-Term Memory neural network that generates a caption for an image in 
case a caption is not provided
* a Latent Dirichlet Analysis, where we identify the most probable topic per review 
and the top words that are present in the review to map them to captions that are in
the same topic and contain one or more of the top words.

There are several libraries required for the framework to work:
* Python 2.7
* pandas >= 0.18.0
* numpy >= 1.10.4
* theano >= 0.8.0
* lasagne >= 0.2.dev1
* nltk >= 3.2.1
* scikit-image >= 0.12.3
* scikit-learn >= 0.17.1
* django >= 1.9.5
* multiprocessing >= 2.6.2.1
* nolearn >= 0.6a0.dev0
* pillow >= 3.1.1
* textblob >= 0.11.1
* hickle >= 2.0.4

##Structure

Our project has a Python package structure but each folder in this structure has a particular
purpose:

0. **autoCap:** This folder contains all the scripts that create our website and help with its deployment. 
0. **batches:** This folder contains all the scripts required for image batch generation which was required to be able to handle images of larger size (224 x 224 pixels)
0. **captioning:** This folder contains all of the scripts related to captioning, including some scripts that wre required to reformat the data so that it was compatible with the model.
0. **convnets:** Includes some of the implementations of Convolutional Neural Network (CNN) models.
0. **data:** This folder includes temporary data as well as the table that maps categories to an integer.
0. **LDAmodeling:** This folder includes the implementation of the LDA model along with some results.
0. **memload:** Contains the scripts used for the images that can be loaded directly from memory (instead of batch processing), in particular images of 64 x 64 pixels.
0. **models:** This folder contains scripts that define and initialize the structure of each of the models that were tested, including VGG-16, VGG-19 and GoogleNet. We also include the script for Inception_v3 but this model did not work.
0. **other:** This folder contains scripts that were used for exploration of the different libraries available, but are not part of the framework. The code inside this folder is not documented.
0. **preprocessing:** Includes code to preprocess the images and the reviews.
0. **review_mapping:** Includes the code that maps a review to an image.
0. `__init__.py`: This file is empty and it is only included as a requirement for all packages.
0. `keras.sh` and `lasagne.sh`: Auxiliary scripts that are used for job scheduling in a SLURM based cluster.

##Documentation

Documentation is available online: https://auto-captioning.herokuapp.com/
