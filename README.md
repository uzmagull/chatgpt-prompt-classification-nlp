# ChatGPT Prompt Classification using NLP, ML & Deep Learning
## Project Overview
This project implements an end-to-end Natural Language Processing (NLP) pipeline to classify ChatGPT user prompts into multiple categories using Machine Learning and Deep Learning techniques.
The system takes a text prompt as input and predicts its category such as Coding, Academic, Creative, etc.
It demonstrates the complete workflow from data preprocessing to model evaluation.

## Problem Statement
Given a user-written prompt, automatically determine its category.
This is a multi-class text classification problem.
### Target Categories
Academic
Business
Coding
Creative
Personal
Summarization
Translation

## Dataset
### Dataset Name:
chatgpt_prompts_dataset
### Size:
1000 samples
### Format:
CSV
The dataset is clean, balanced, and suitable for NLP classification tasks.

## Technologies Used
Python
Google Colab
Pandas, NumPy
NLTK
Scikit-learn
TensorFlow / Keras

## NLP Pipeline
1. Text Cleaning (lowercasing, regex, stopword removal)
2. Tokenization
3. TF-IDF Vectorization
4. Label Encoding
5. Train-Test Split (80% train, 20% test with stratification)

## Models Implemented
### Machine Learning Models
1. Logistic Regression
2. Naive Bayes
3. Support Vector Machine (SVM)
4. Random Forest

### Deep Learning Model
1. Deep Neural Network (DNN)
2. Fully connected (Dense) layers
3. ReLU activation
4. Dropout regularization
5. Softmax output layer

## Model Evaluation
### Models were evaluated using:
Accuracy, 
Precision, 
Recall, 
F1-score,
Confusion Matrix

Evaluation was performed on unseen test data to ensure fair comparison.

## Project Structure
📦 chatgpt-prompt-classification-nlp
 ┣ 📓 ChatGPT_Prompts_Classification.ipynb
 ┣ 📜 chatgpt_prompts_dataset_1000.csv
 ┣ 📄 README.md

## How to Run the Project
1. Clone the repository: git clone https://github.com/your-username/chatgpt-prompt-classification-nlp.git
2. Open the notebook in Google Colab
3. Upload the dataset when prompted
4. Run the notebook cells step by step

## Key Learning Outcomes
Understanding of NLP preprocessing techniques

Practical use of TF-IDF for text vectorization

Comparison between ML and Deep Learning models

Model evaluation using multiple performance metrics

## Future Improvements
Increase dataset size

Use word embeddings (Word2Vec, GloVe)

Implement LSTM / GRU models

Deploy as a web application

## Author
Uzma Gul

NLP & Machine Learning Project

## Version Control

This project was developed using Google Colab and version-controlled on GitHub.
