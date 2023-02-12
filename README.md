# Build-CNN-Model-for-EEG-Classification
The target of this project is to build a model that can recognize seizure from signals.
the process will be described below:

### 1. Loading Dataset
In this project, first, we load a dataset from the three first folders of this source: https://physionet.org/content/chbmit/1.0.0/ 
In the following step, we should consider just two channels for each data point and after extracting data points from each channel, we split the vector into smaller ones with batch_size = 5.   
Since our frequency is 256, to extract 5 seconds we should consider 5 * 256 for each channel and then combine them to have a vector like this : (1,1280,2) which means that we have 1280 numbers for just one vector with 2 channels. It is
worth noting that we use 7 files that involve seizures from folder 1 and two files from folder 2 and 1 file from folder 3 to extract the seizure since the amount of information was low, we decided to use data augmentation methods. 
Data augmentation is a technique used to artificially increase the size of a dataset by generating new data samples from the existing ones.
Data augmentation can also be applied to vector data, such as text or numerical data. Here we used 4 different methods for this purpose including
* Addition of random noise: Adding small random values to the data is a common method to create more diverse and robust training data. This can help the model better handle small variations in the data, improving its ability to generalize to unseen examples.
* Scaling:  Scaling is a method to change the scale of the data. For example, you could multiply each element in the vector by a scalar value, or divide the entire vector by a scalar value. Scaling can help the model be more robust to different scales of input data.  
* Time Shifting: Time shifting is a method to shift the time axis of the data. For example, you could shift the entire vector by a fixed number of time steps, either forward or backward. This can help the model be more robust to temporal variations in the data.  
* Amplitude scaling: Amplitude scaling is a method to change the amplitude of the data. For example, you could multiply each element in the vector by a different scalar value, creating different amplitudes for different parts of the vector. This can help the model be more robust to changes in the amplitude of the data.  
All of these techniques aim to increase the diversity of the training data and reduce overfitting by introducing small variations to the input data.  
  
the way of choosing data:
* from category 1:extracting seizures from 7 files that only 4 of them is accompanying by extracting normal data.
  extracting normal data from 5 other files without seizures.
* from category 2:extracting seizures from 2 with normal data.
  extracting normal data from 2 other files without seizures.
  
* from category 3:extracting seizures from 1 without normal data.
### 2. Building Model
Here, we want to build a CNN model that can process our data.  
Before building the model, we should extract important features of our data by specific methods, which is called feature extraction and for each (1280,2) we select 15 features by considering each channel separately. The feature that we have already extracted:  
FEATURES = ['MIN', 'MAX', 'MEAN', 'RMS', 'VAR', 'STD', 'POWER', 'PEAK', 'P2P', 'CREST FACTOR',
            'MAX_f', 'SUM_f', 'MEAN_f', 'VAR_f', 'PEAK_f']  
Now we should build the model with 3 Conv1D layers and 2 Dense layers and also check alternatives.   
In the following step, we should concatenate our features (300,15,2) with the feature that our model have extracted. After that we should pass unique vectors to the classification layer in order to predict our labels.
 
### 3. Results for 2240 vectors
in each iteration data are shuffled.
* after 20 epochs (3 Conv1D and 2 Dense layer and kernel = 5):
  * Accuracy : 87%
  * Recall: 0.94
  * Precision: 0.45
    
* after 20 epochs (2 Conv1D and 2 Dense layer and kernel = 3):
  * Accuracy : 80%
  * Recall: 0.70
  * Precision: 27%
  
* after 20 epochs (3 Conv1D and 1 Dense layer and kernel = 5):
  * Accuracy : 75%
  * Recall: 0.58
  * Precision: 0.78