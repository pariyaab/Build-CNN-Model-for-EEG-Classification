# Build-CNN-Model-for-EEG-Classification
The target of this project is to build a model that can recognize seizure from signals.
the process will be described bellow:

### 1. Loading Dataset
In this project, first we load dataset from three first folder of this dataset : https://physionet.org/content/chbmit/1.0.0/  
In the following step, we should consider just two channels for each data point and after extracting data points from each channel, we split the vector into smaller ones with batch_size = 5.  
Since our frequency is 256, to extracting 5 second we should consider 5 * 256 for each channel and then combine them to have a vector like this : (1,1280,2) which means that we have 1280 numbers for just one vector with 2 channels. It is 
worth noting that we use 7 files that involves seizures from folder1 and two file from folder 2 and 1 file from folder 3 to extracting the seizure and since
the amount of information was low, we decided to use data augmentation methods.  
Data augmentation is a technique used to artificially increase the size of a dataset by generating new data samples from the existing ones.
Data augmentation can also be applied to vector data, such as text or numerical data. Here we used 4 different methods for this purpose including 
* Rotation : Rotate the vector by a random angle within the range of -30 to 30 degrees, with a step of 5 degrees.
* Scaling: Increasing or decreasing the size of the data.
* Flipping: Reversing the data along an axis (e.g., left-right flipping for image data).
* Noise injection: Adding random noise to the data to increase the variability of the training data.  
  
All of these techniques aim to increase the diversity of the training data and reduce overfitting by introducing small variations to the input data.  
### 2. Building Model
Here, we want to build a CNN model that can process our data.  
Before building the model, we should extract important feature of our data by specific methods, which is called feature extraction anf for ech (1280,2) we select 15 features by considering each channel separately. The feature that we have already extracted:  
FEATURES = ['MIN', 'MAX', 'MEAN', 'RMS', 'VAR', 'STD', 'POWER', 'PEAK', 'P2P', 'CREST FACTOR',
            'MAX_f', 'SUM_f', 'MEAN_f', 'VAR_f', 'PEAK_f']  
Now we should build the model with 3 Conv1D layers and 2 Dense layers.  
In the following step, we should concatenate our features (300,15,2) with the feature that our model have extracted. After that we should pass unique vectors to the classification layer in order to predict our labels.

### 3. Results
* after 20 epochs (3 Conv1D and 2 Dense layer and kernel = 5):
  * Accuracy : 74%
  * Recall: 0.357
  * Precision: 0.588
    
* after 20 epochs (2 Conv1D and 2 Dense layer and kernel = 3):
  * Accuracy : 64%
  * Recall: 0.257
  * Precision: 0.478