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
   git clone https://github.com/yourusername/phishing-email-classification.git
   cd phishing-email-classification
