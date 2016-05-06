##Model scripts

This folder contains scripts that define and initialize the structure of each of the models that were tested, including
VGG-16, VGG-19 and GoogleNet. We also include the script for Inception_v3 but this model did not work. Scripts that do
not have a usage defined are mostly helper scripts called from other scripts or have hard-coded parameters.

### Structure

In this section, we describe the functionality of each script:
* `__init__.py`: This file is empty and it is only included as a requirement for all packages.
* `googlenet.py`: BLVC Googlenet, model from the paper:
["Going Deeper with Convolutions"]{http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7298594}
* `inception_v3.py`: Inception-v3, model from the paper:
["Rethinking the Inception Architecture for Computer Vision"]{http://arxiv.org/abs/1512.00567}
* `vgg16.py`: VGG-16, 16-layer model from the paper:
["Very Deep Convolutional Networks for Large-Scale Image Recognition"]{http://arxiv.org/abs/1409.1556}
* `vgg19.py`: VGG-19, 19-layer model from the paper:
["Very Deep Convolutional Networks for Large-Scale Image Recognition"]{http://arxiv.org/abs/1409.1556}