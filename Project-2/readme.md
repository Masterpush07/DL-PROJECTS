# IMDB Movie Reviews Sentiment Analysis using LSTM

This project focuses on analyzing the sentiment of movie reviews from the IMDB dataset using a Long Short-Term Memory (LSTM) neural network. The goal is to classify reviews as either **positive** or **negative** based on their content.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset Description](#dataset-description)  
3. [Model Details](#model-details)  
4. [How to Use](#how-to-use)  
5. [Technologies Used](#technologies-used)  
6. [Results](#results)  
7. [Contributors](#contributors)  
8. [License](#license)  

## Project Overview
The objective of this project is to build a sentiment analysis model capable of predicting whether a movie review conveys a positive or negative sentiment. The key steps in this project include:  
- Preprocessing the text data.  
- Training an LSTM-based neural network.  
- Evaluating model performance.  
- Deploying the model for predictions on new reviews.  

## Dataset Description
The dataset used for this project is the **IMDB Movie Reviews Dataset**, which contains 50,000 movie reviews labeled as **positive** or **negative**.  

### Data Details:
- **Training Set**: 25,000 labeled reviews.  
- **Testing Set**: 25,000 labeled reviews.  
- Reviews are equally distributed between positive and negative labels.  

### Preprocessing Steps:
- Tokenizing the reviews.  
- Padding/truncating sequences to a uniform length.  
- Converting text to numerical format using word embeddings.  

## Model Details
The sentiment analysis model is built using a Long Short-Term Memory (LSTM) network, which is effective for sequential data like text.  

### Model Architecture:
- **Embedding Layer**: Converts input words into dense vectors.  
- **LSTM Layer**: Captures sequential dependencies in the data.  
- **Dense Layer**: Fully connected layer with ReLU activation.  
- **Output Layer**: Single neuron with sigmoid activation for binary classification.  

### Compilation Details:
- **Optimizer**: Adam  
- **Loss Function**: Binary Crossentropy  
- **Metrics**: Accuracy  

### Training Configuration:
- **Batch Size**: 32  
- **Number of Epochs**: 5  
- **Validation Split**: 20%  

## How to Use
### Google Colab Users:
1. Upload the provided Jupyter Notebook (`IMDB_Reviews_Sentiment_Analysis_LSTM.ipynb`) to Google Colab.  
2. Install the required dependencies:  
   ```bash
   !pip install tensorflow numpy pandas matplotlib
   # IMDB Movie Reviews Sentiment Analysis using LSTM

This project focuses on analyzing the sentiment of movie reviews from the IMDB dataset using a Long Short-Term Memory (LSTM) neural network. The goal is to classify reviews as either **positive** or **negative** based on their content.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset Description](#dataset-description)  
3. [Model Details](#model-details)  
4. [How to Use](#how-to-use)  
5. [Technologies Used](#technologies-used)  
6. [Results](#results)  
7. [Contributors](#contributors)  
8. [License](#license)  

## Project Overview
The objective of this project is to build a sentiment analysis model capable of predicting whether a movie review conveys a positive or negative sentiment. The key steps in this project include:  
- Preprocessing the text data.  
- Training an LSTM-based neural network.  
- Evaluating model performance.  
- Deploying the model for predictions on new reviews.  

## Dataset Description
The dataset used for this project is the **IMDB Movie Reviews Dataset**, which contains 50,000 movie reviews labeled as **positive** or **negative**.  

### Data Details:
- **Training Set**: 25,000 labeled reviews.  
- **Testing Set**: 25,000 labeled reviews.  
- Reviews are equally distributed between positive and negative labels.  

### Preprocessing Steps:
- Tokenizing the reviews.  
- Padding/truncating sequences to a uniform length.  
- Converting text to numerical format using word embeddings.  

## Model Details
The sentiment analysis model is built using a Long Short-Term Memory (LSTM) network, which is effective for sequential data like text.  

### Model Architecture:
- **Embedding Layer**: Converts input words into dense vectors.  
- **LSTM Layer**: Captures sequential dependencies in the data.  
- **Dense Layer**: Fully connected layer with ReLU activation.  
- **Output Layer**: Single neuron with sigmoid activation for binary classification.  

### Compilation Details:
- **Optimizer**: Adam  
- **Loss Function**: Binary Crossentropy  
- **Metrics**: Accuracy  

### Training Configuration:
- **Batch Size**: 32  
- **Number of Epochs**: 5  
- **Validation Split**: 20%  

## How to Use
### Google Colab Users:
1. Upload the provided Jupyter Notebook (`IMDB_Reviews_Sentiment_Analysis_LSTM.ipynb`) to Google Colab.  
2. Install the required dependencies:  
   ```bash
   !pip install tensorflow numpy pandas matplotlib
   # IMDB Movie Reviews Sentiment Analysis using LSTM

This project focuses on analyzing the sentiment of movie reviews from the IMDB dataset using a Long Short-Term Memory (LSTM) neural network. The goal is to classify reviews as either **positive** or **negative** based on their content.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset Description](#dataset-description)  
3. [Model Details](#model-details)  
4. [How to Use](#how-to-use)  
5. [Technologies Used](#technologies-used)  
6. [Results](#results)  
7. [Contributors](#contributors)  
8. [License](#license)  

## Project Overview
The objective of this project is to build a sentiment analysis model capable of predicting whether a movie review conveys a positive or negative sentiment. The key steps in this project include:  
- Preprocessing the text data.  
- Training an LSTM-based neural network.  
- Evaluating model performance.  
- Deploying the model for predictions on new reviews.  

## Dataset Description
The dataset used for this project is the **IMDB Movie Reviews Dataset**, which contains 50,000 movie reviews labeled as **positive** or **negative**.  

### Data Details:
- **Training Set**: 25,000 labeled reviews.  
- **Testing Set**: 25,000 labeled reviews.  
- Reviews are equally distributed between positive and negative labels.  

### Preprocessing Steps:
- Tokenizing the reviews.  
- Padding/truncating sequences to a uniform length.  
- Converting text to numerical format using word embeddings.  

## Model Details
The sentiment analysis model is built using a Long Short-Term Memory (LSTM) network, which is effective for sequential data like text.  

### Model Architecture:
- **Embedding Layer**: Converts input words into dense vectors.  
- **LSTM Layer**: Captures sequential dependencies in the data.  
- **Dense Layer**: Fully connected layer with ReLU activation.  
- **Output Layer**: Single neuron with sigmoid activation for binary classification.  

### Compilation Details:
- **Optimizer**: Adam  
- **Loss Function**: Binary Crossentropy  
- **Metrics**: Accuracy  

### Training Configuration:
- **Batch Size**: 32  
- **Number of Epochs**: 5  
- **Validation Split**: 20%  

## How to Use
### Google Colab Users:
1. Upload the provided Jupyter Notebook (`IMDB_Reviews_Sentiment_Analysis_LSTM.ipynb`) to Google Colab.  
2. Install the required dependencies:  
   ```bash
   !pip install tensorflow numpy pandas matplotlib








