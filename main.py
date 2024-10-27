# main.py

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

# Load Data
df = pd.read_csv(r"Phishing_Email.csv")

# Clean Data
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Encode Labels
lbl = LabelEncoder()
df['Email Type'] = lbl.fit_transform(df['Email Type'])

# Text Preprocessing Function
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['Email Text'] = df['Email Text'].apply(preprocess_text)

# Vectorization
tf = TfidfVectorizer(stop_words='english', max_features=10000)
feature_x = tf.fit_transform(df['Email Text']).toarray()
y_tf = np.array(df['Email Type'])

# Split Data
x_train, x_test, y_train, y_test = train_test_split(feature_x, y_tf, test_size=0.2, random_state=0)

# Train Models
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)

mnb = MultinomialNB()
mnb.fit(x_train, y_train)

bnb = BernoulliNB()
bnb.fit(x_train, y_train)

# Save Models and Vectorizer
pickle.dump(rf, open("random_forest_model.pkl", "wb"))
pickle.dump(mnb, open("multinomial_nb_model.pkl", "wb"))
pickle.dump(bnb, open("bernoulli_nb_model.pkl", "wb"))
pickle.dump(tf, open("tfidf_vectorizer.pkl", "wb"))
