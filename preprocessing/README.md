##Preprocessing scripts

Includes code to preprocess the images and the reviews. Scripts that do not have a usage defined are mostly helper scripts
called from other scripts or have hard-coded parameters.

### Structure

In this section, we describe the functionality of each script:
* `__init__.py`: This file is empty and it is only included as a requirement for all packages.
* `loadWikidictionary.py`: Script that creates a corpus from the Wikipedia data.
* `image_preprocessing.py`: Preprocess the images depending on the model selected.