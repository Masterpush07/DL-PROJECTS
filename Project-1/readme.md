# Face Mask Detection Project

This project focuses on detecting whether a person is wearing a face mask using a **Convolutional Neural Network (CNN)**. It uses an image dataset to train a machine learning model for classification purposes.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Model Details](#model-details)
4. [How to Use](#how-to-use)
5. [Technologies](#technologies)
6. [License](#license)
7. [Results](#results)
8. [Contributors](#contributors)

---

## Project Overview
The objective of this project is to create a system that can classify images into two categories:
- **With Mask**
- **Without Mask**

The solution involves:
1. Preprocessing an image dataset.
2. Building and training a CNN-based model.
3. Evaluating the model's performance.
4. Using the model for predictions on new images.

---

## Dataset Description
The dataset used is the **Face Mask Dataset**, available on Kaggle. It consists of labeled images in two categories:
- **With Mask**: Images of individuals wearing face masks.
- **Without Mask**: Images of individuals not wearing face masks.

### Data Summary:
- **Number of Images:**
  - With Mask: 3,725
  - Without Mask: 3,828
- **Image Dimensions**: Resized to `128x128` pixels for uniformity.

### Labels:
- `1`: With Mask
- `0`: Without Mask

---

## Model Details
The predictive model is a **Convolutional Neural Network (CNN)**, which is highly effective for image classification tasks.

### Steps:
1. **Data Preprocessing**:
   - Resizing images to `128x128`.
   - Converting images to RGB format.
   - Normalizing pixel values (scaling between 0 and 1).

2. **Model Architecture**:
   - **Convolutional Layers** with ReLU activation.
   - **MaxPooling Layers** for down-sampling.
   - **Fully Connected Dense Layers**.
   - **Dropout** for regularization.
   - **Output Layer** with Sigmoid activation for binary classification.

3. **Compilation**:
   - **Loss Function**: Sparse Categorical Crossentropy.
   - **Optimizer**: Adam.
   - **Evaluation Metric**: Accuracy.

4. **Training**:
   - **Train-Test Split**: 80-20.
   - **Validation Split**: 10% from the training data.
   - **Epochs**: 5.

---

## How to Use
### Google Colab Users:
1. Upload the Jupyter Notebook (`face_mask_detection.ipynb`) and the `kaggle.json` API key file.
2. Set up the environment:
    ```bash
    !pip install kaggle
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    ```
3. Download and extract the dataset:
    ```python
    !kaggle datasets download -d omkargurav/face-mask-dataset
    from zipfile import ZipFile
    with ZipFile('/content/face-mask-dataset.zip', 'r') as zip:
        zip.extractall()
    ```
4. Follow the steps in the notebook to train and test the model.

### Other Environments:
1. Install the required libraries:
    ```bash
    pip install numpy pandas matplotlib tensorflow pillow scikit-learn
    ```
2. Run the Jupyter Notebook (`face_mask_detection.ipynb`) locally.

---

## Technologies
- **Python**
- **NumPy**, **Pandas**
- **Matplotlib**, **OpenCV**
- **TensorFlow**, **Keras**
- **Google Colab**

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Results
The CNN model achieved high accuracy during evaluation.  
Here’s a summary of the results:
- **Test Accuracy**: Approximately **95%**

### Visualization:
- **Loss and Accuracy Curves**: Demonstrate the model's performance over training epochs.

---

## Contributors
- **Pushpanathan**: Development and implementation.

This project showcases a practical application of computer vision in healthcare and safety measures.
