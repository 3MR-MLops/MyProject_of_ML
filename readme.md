House Price Prediction API
# 🏠 House Price Prediction API (End-to-End MLOps Project)

This project is a complete Machine Learning pipeline that predicts housing prices using a Neural Network model and serves it through a high-performance FastAPI.

## 🚀 Project Overview
I developed a regression model to estimate house prices based on 4 key features. The goal was to demonstrate a full MLOps workflow, from model training to local deployment.

### Key Features:
* **Area**: Total square footage of the property.
* **Rooms**: Total number of bedrooms.
* **Age**: How many years since the house was built.
* **Has Pool**: A binary feature (0 or 1) indicating if there's a swimming pool.

## 🧠 Model Details
* **Framework**: Built using **TensorFlow/Keras**.
* **Architecture**: A Sequential Neural Network with multiple Dense layers using **ReLU** activation.
* **Optimization**: Trained using the **Adam** optimizer and **MSE** loss function.
* **Performance**: Achieved a stable Loss of **27,410**, providing highly realistic price predictions.



## 🛠️ Tech Stack
* **Python**: Core programming language.
* **FastAPI**: To create the web API for real-time predictions.
* **Uvicorn**: ASGI server for running the FastAPI application.
* **NumPy**: For data preprocessing and array manipulation.
* **Git**: For version control and deployment to GitHub.

## 💻 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/3MR-MLops/MyProject_of_ML.git](https://github.com/3MR-MLops/MyProject_of_ML.git)
