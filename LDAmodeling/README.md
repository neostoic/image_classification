# LDAModeling

* `LDAUtilities.py`
Contains all the definitions needed to apply an LDA model to a document

* `ldaModel.py`
Creates an LDA model out of data from a pkl file

* `ldaTesting.py`
Contains code to make experiments for LSA and LDA

* `Opentags.py`
Auxiliary functions to open some files from the data

* `PrintRecommendations.py`
Prints recommendations given in a json file

* `MonAmiGabiCapsTrainingLDAtopicModel.mm`
* `MonAmiGabiCapsTrainingLDAtopicModel.mm.state`
Contains the lda model created with the Captions only method

* `MonAmiGabiTrainingLDAtopicModel.mm`
* `MonAmiGabiTrainingLDAtopicModel.mm.state`
Contains the lda model created with removing stopwords method

* data folder: 
Contains all the data generated and files used to create and use the LDA model

  * `LDA_results_20.err`
  Contains the logs for the last version of our All-in model
  * `reviewsTraining.dict`
  Dictionary of terms for this version of the model
  * `reviewsTraining.mm`
  Corpus for the All-in model
  * `reviewsTraining.mm.index`
  Contains the corresponding indexes
  * `reviewsTrainingLDAtopicModel.mm`
  Contains the last version of our model

  *`MonAmiGabiCapsTraining.pkl`
  Captions from MonAmi Gabi designated for training(60%) testing(40%)

  * `MonAmiGabiCapsTesting.pkl`
  Captions from MonAmi Gabi designated for training(60%) testing(40%)

  * `MonAmiGabiTraining.mm`
  Corpus using MonAmiGabi

