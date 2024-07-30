# Share Price Forecasting

This repository is dedicated to forecasting share prices using Recurrent Neural Networks (RNNs). By leveraging historical stock data, the project aims to build an accurate predictive model for stock prices, providing valuable insights for investors and financial analysts.

## Introduction

Stock price forecasting is a critical task in finance, aiming to predict future stock prices based on historical data. This project employs Recurrent Neural Networks (RNNs) to capture temporal dependencies in stock price movements, making it possible to predict future prices more accurately.

## Features

- **Data Preprocessing**: Clean and prepare historical stock data for training the model.
- **Model Architecture**: Implement a Recurrent Neural Network (RNN) using TensorFlow/Keras.
- **Training and Validation**: Train the RNN model on historical stock data and validate its performance.
- **Prediction**: Forecast future stock prices based on the trained model.
- **Evaluation Metrics**: Use metrics like Mean Squared Error (MSE) to evaluate model performance.

### Prerequisites

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib

### Evaluation

1. After training, the model will evaluate its performance on the test dataset and output relevant metrics.

## Project Structure

- `recurrent_neural_network.py`: The main script for building, training, and evaluating the RNN model.
- `stock_dataset.numbers`: Historical stock prices dataset used for training.
- `stock_dataset_test.numbers`: Historical stock prices dataset used for testing.
- `requirements.txt`: List of required Python libraries.

## Data Preparation

1. **Load Data**: Import the historical stock data from the provided dataset files.
2. **Clean Data**: Handle missing values and normalize the data for better model performance.
3. **Split Data**: Divide the dataset into training and testing sets to evaluate the model's generalization ability.

## Model Training

1. **Define Model**: Create an RNN architecture using TensorFlow/Keras.
2. **Compile Model**: Configure the model with appropriate loss functions and optimizers.
3. **Train Model**: Train the model on the training dataset and validate it on the testing dataset.
