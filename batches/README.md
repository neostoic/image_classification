##Batch Generator scripts

This folder contains all the scripts required for image batch generation which was required to be able to 
handle images of larger size (224 x 224 pixels). Scripts that do not have a usage defined are mostly helper scripts
called from other scripts or have hard-coded parameters.

### Structure

In this section, we describe the functionality of each script:
* `__init__.py`: This file is empty and it is only included as a requirement for all packages.
* `batchgen.py`: Context manager to generate batches in the background via a process pool
* `json_to_pickle.py`: Script to pickle json data automatically
Usage: `python path\to\json_to_pickle.py path\to\json_file.json`
* `load_batch.py`: Helper script to load batches of data for image classification using the BatchGen class.
* `pickle_train_test_data.py`: Script that splits data into folds. The last fold is testing data while the others are training data 
Usage:  `python pickle_train_test_data.py pickled_data number_of_folds` 
* `produce.py`: Script that creates small batches of numpy compressed arrays by calling the batch function from `load_batch.py`. 
