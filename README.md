# Machine Learning Projects

**1. [House occupancy](./proj1_house_occupancy_gluon.ipynb)**  - 

*Problem* - This is a binary classification problem where observation of environmental factors such as temperature and humidity is used to classify whether a room is occupied or not.

*Dataset* - https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+

*Result* - Recurrent neural network (LSTM) model to predict occupancy of a room, 98% accuracy on the test set compared to a baseline accuracy of 94% for a softmax classifier

**2. [Seedling with CNN](./proj2_cnn_plant_seedling.ipynb)**- 

*Problem* - This is a multiclass classification problem where an image of a plant seedling has to be classified as one of the 12 species.

*Dataset* - https://www.kaggle.com/c/plant-seedlings-classification/data

*Result* - A simple CNN with accuracy around 70% 

**3. [Seedling Classification with model tuning on Amazon SageMaker](../../sagemaker-seedling)** -

*Problem* - This is a multiclass classification problem where an image of a plant seedling has to be classified as one of the 12 species.

*Dataset* - https://www.kaggle.com/c/plant-seedlings-classification/data

*Result* - Performed transfer learning (VGG19 model) and hyperparameter optimization using AWS Sagemaker (build your own container for Keras) and got an accuracy of 83.76%

**4. [Music Mood Classification using Fast.ai and Pytorch](../../proj4_MusicMoodClassification)** [Work in Progress] -

Building a music mood classifier that classifies a music clip to one of 6 moods (happy, sad, angry, scary, funny, tender) by converting audio to image (Spectrogram, Melspectrogram, Tempogram) and using fastai library, transfer learning, pytorch to train the mode

*Problem* - This is a multiclass classification problem where an audio is classified based on its mood, as one of 6 classes (happy, sad, angry, scary, funny, tender).

*Dataset* - https://research.google.com/audioset/dataset/index.html

*Result* - Performed transfer learning (Resnet50) on Melspectrogram using fastai library and got an accuracy of 66.84%.
Removed the ambiguous mood/class - 'Tender' and got an accuracy of 79%

