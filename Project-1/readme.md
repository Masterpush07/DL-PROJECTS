Face Mask Detection Project
This project focuses on detecting whether a person is wearing a face mask using a Convolutional Neural Network (CNN). It utilizes an image dataset to train a machine learning model for classification purposes.

Table of Contents
Project Overview
Dataset Description
Model Details
How to Use
Technologies
License
Results
Contributors
Project Overview
The objective of this project is to create a system that can classify images into two categories:

With Mask
Without Mask
The solution involves:

Preprocessing an image dataset.
Building and training a CNN-based model.
Evaluating the model's performance.
Using the model for predictions on new images.
Dataset Description
The dataset used is the Face Mask Dataset, available on Kaggle. It consists of labeled images in two categories:

With Mask: Images of individuals wearing face masks.
Without Mask: Images of individuals not wearing face masks.
Data Summary
Number of Images:
With Mask: 3,725
Without Mask: 3,828
Image Dimensions: Resized to 128x128 pixels for uniformity.
Labels
1: With Mask
0: Without Mask
Model Details
The predictive model is a Convolutional Neural Network (CNN), which is highly effective for image classification tasks.

Steps:
Data Preprocessing:

Resizing images to 128x128.
Converting images to RGB format.
Normalizing pixel values (scaling between 0 and 1).
Model Architecture:

Convolutional Layers with ReLU activation.
MaxPooling Layers for down-sampling.
Fully Connected Dense Layers.
Dropout for regularization.
Output Layer with Sigmoid activation for binary classification.
Compilation:

Loss Function: Sparse Categorical Crossentropy.
Optimizer: Adam.
Evaluation Metric: Accuracy.
Training:

80-20 train-test split.
Validation split of 10% from the training data.
Model trained for 5 epochs.
How to Use
Google Colab Users:
Upload the Jupyter Notebook (face_mask_detection.ipynb) and the kaggle.json API key file.

Run the following to set up the environment:

bash
Copy code
!pip install kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
Download and extract the dataset:

bash
Copy code
!kaggle datasets download -d omkargurav/face-mask-dataset
from zipfile import ZipFile
with ZipFile('/content/face-mask-dataset.zip', 'r') as zip:
    zip.extractall()
Follow the steps in the notebook to train and test the model.

Other Environments:
Install the required libraries:
bash
Copy code
pip install numpy pandas matplotlib tensorflow pillow scikit-learn
Run the Jupyter Notebook (face_mask_detection.ipynb) locally.
Technologies
Python
NumPy, Pandas
Matplotlib, OpenCV
TensorFlow, Keras
Google Colab
License
This project is licensed under the MIT License - see the LICENSE file for details.

Results
The CNN model achieved high accuracy during evaluation. Here’s a summary of results:

Test Accuracy: 95% (approx.)
Visualization:
Loss and Accuracy curves demonstrate the model's performance over training epochs.
Contributors
Pushpanathan: Development and implementation.
This project showcases a practical application of computer vision in healthcare and safety measures.
