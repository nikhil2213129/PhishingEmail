# Phishing Email Classification

This project implements a machine learning model to classify emails as phishing or not phishing using various algorithms. The dataset used contains labeled email texts and their corresponding types.

## Table of Contents

- [Overview](#overview)
- [Technologies](#technologies)
- [Dataset](#dataset)
- [Setup](#setup)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [License](#license)

## Overview

The main goal of this project is to build a model that can accurately classify emails as phishing or legitimate. The project utilizes text preprocessing, feature extraction with TF-IDF, and classification using Random Forest, Multinomial Naive Bayes, and Bernoulli Naive Bayes algorithms.

## Technologies

- Python 3.x
- Pandas
- NumPy
- scikit-learn
- Regex
- Pickle

## Dataset

The dataset used in this project is `Phishing_Email.csv`, which contains two columns:
- **Email Text**: The content of the email.
- **Email Type**: The label indicating whether the email is phishing (1) or legitimate (0).

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/nikhil2213129/PhishingEmail.git
   cd phishing-email-classification

Install the required packages:

bash
Copy code
pip install pandas numpy scikit-learn
Place the Phishing_Email.csv file in the project directory.

Usage
Run the main script to preprocess the data, train the models, and save the trained models and the TF-IDF vectorizer:

bash
Copy code
python main.py
This will generate the following files in the project directory:

random_forest_model.pkl
multinomial_nb_model.pkl
bernoulli_nb_model.pkl
tfidf_vectorizer.pkl
Example of Loading the Models
You can load the trained models for inference using the following code:

python
Copy code
import pickle

# Load models
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Example inference
email_text = "Your account has been compromised. Click here to secure it."
processed_text = preprocess_text(email_text)  # Ensure to apply the same preprocessing
features = tfidf_vectorizer.transform([processed_text])
prediction = rf_model.predict(features)

print("Predicted Email Type:", prediction)
Models
Three different models are trained in this project:

Random Forest Classifier: An ensemble learning method that constructs multiple decision trees and outputs the mode of their predictions.
Multinomial Naive Bayes: A probabilistic classifier suitable for discrete data, commonly used for text classification.
Bernoulli Naive Bayes: Similar to Multinomial Naive Bayes, but suitable for binary/boolean features.
Evaluation
To evaluate the performance of the trained models, consider the following metrics:

Accuracy: The proportion of correct predictions.
Precision: The ratio of true positives to the sum of true and false positives.
Recall: The ratio of true positives to the sum of true positives and false negatives.
F1 Score: The harmonic mean of precision and recall.
You can generate a classification report using scikit-learn’s classification_report function.

Results
After training, the models are saved for later use. During testing, you can expect to see varying performance metrics based on the model used. Typically, Random Forest tends to perform better in text classification tasks due to its ensemble nature.

Future Work
Explore additional text preprocessing techniques (e.g., stemming, lemmatization).
Experiment with more advanced models such as Support Vector Machines or Deep Learning methods.
Implement a web interface for real-time email classification.
Gather and use a larger dataset for improved model accuracy.


Feel free to customize any sections to better fit your project specifics!
