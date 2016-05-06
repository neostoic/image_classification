##Batch Generator scripts

This folder contains all the scripts required for image batch generation which was required to be able to
handle images of larger size (224 x 224 pixels). Scripts that do not have a usage defined are mostly helper scripts
called from other scripts or have hard-coded parameters.

### Structure

In this section, we describe the functionality of each script:
* `__init__.py`: This file is empty and it is only included as a requirement for all packages.
* `convnet_keras.py`: Train two simple deep CNN on the Yelp small images (64 x 64 pixels) dataset in ASUS laptop.
GPU Usage: `THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python convnet_keras.py`
* `convnet_keras_griffin.py`: Train two simple deep CNN on the Yelp small images (64 x 64 pixels) dataset in Griffin cluster.
GPU Usage: `THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python convnet_keras_griffin.py`
* `convnet_pretrained.py`: Train two simple deep CNN on the Yelp using batch processing in Griffin cluster.
# GPU Usage: `THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python convnet_pretrained.py`