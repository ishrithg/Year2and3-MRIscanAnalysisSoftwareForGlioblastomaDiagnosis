# The Development of a Software Apparatus and Mechanism for the Analysis of MRI Scans of Glioblastoma Brain Tumors for Feature Extraction, Image Modification, and Diagnosis
**This project builds a convolutional neural network model to classify MRI brain scans as normal or containing a glioblastoma tumor.**

**Data**
- The data consists of MRI scans in DICOM format separated into training and test folders
- Each scan is labeled as normal or glioblastoma tumor based on pathology
- Scans are loaded using the MasterImage class to process images
  
**Preprocessing**
- Scans are converted to grayscale and resized to a standard pixel dimension
- Pixel values are normalized to the 0-1 range
- Image augmentation (rotations, zooms etc) is used to increase diversity
- Data is split into training, validation, and test sets
  
**Model Architecture**
- A convolutional neural network is built in Keras
- Layers include Conv2D, MaxPool2D, Flatten, Dense
- Categorical cross-entropy loss and accuracy metrics
- Model is trained for 50 epochs with early stopping
  
**Training**
- Model is fit on the training and validation sets
- Learning curves are plotted to monitor loss and accuracy
- High accuracy is achieved in classification
  
**Applications**
- The model outputs a tumor likelihood score for a scan
- This could aid in diagnosis and treatment decisions
- With more data, model performance can be improved further
- The pipeline demonstrates how deep learning can be applied to complex medical imaging tasks. MRI scan classification has important clinical relevance for brain tumor patients.

**Usage**

**To apply the model to new data:**
- Prepare MRI scans in same format as training data
- Preprocess scans with same parameters
- Pass scans through trained model to predict tumor likelihood
- Use model outputs to assist diagnosis
