##Memory load scripts

Contains the scripts used for the images that can be loaded directly from memory (instead of batch processing), in
particular images of 64 x 64 pixels. Scripts that do not have a usage defined are mostly helper scripts called from other
scripts or have hard-coded parameters.

### Structure

In this section, we describe the functionality of each script:
* `__init__.py`: This file is empty and it is only included as a requirement for all packages.
* `create_kfolds_pkl.py`: This script creates several pickles based on the number of folds selected.
The last batch is always named test.
Usage: `python create_kfolds_pkl.py source folds`
* `load_yelp_data.py`: This script loads the data from the image pickles and preprocesses it so that the Keras models
can use it.
* `pickle_images.py`: This script pickles the images taking as input the size and number of batches that will be dataset
will be splitted into.
