# app.py

import pickle
import logging
import re
from flask import Flask, request, render_template

# Load Models and Vectorizer
rf = pickle.load(open("random_forest_model.pkl", "rb"))
mnb = pickle.load(open("multinomial_nb_model.pkl", "rb"))
bnb = pickle.load(open("bernoulli_nb_model.pkl", "rb"))
tf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Configure Logging
logging.basicConfig(filename='email_detection.log', level=logging.INFO)

app = Flask(__name__)

# Text Preprocessing
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_headers(headers):
    suspicious_keywords = ['urgent', 'action required', 'verify', 'confirm', 'account', 'suspicious']
    for keyword in suspicious_keywords:
        if keyword in headers.lower():
            return "Suspicious header detected."
    return "Headers appear normal."

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'txt', 'csv', 'log'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_attachment(file):
    if file and allowed_file(file.filename):
        file_content = file.read().decode('utf-8', errors='ignore')
        if "suspicious" in file_content.lower():
            return "Suspicious content detected in attachment."
        return file_content
    return ""

# Log predictions
def log_prediction(email_text, predictions):
    logging.info(f'Email: {email_text}, Predictions: {predictions}')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_headers = request.form['email_headers']
    email_text = request.form['email_text']
    
    # Process Headers and Body
    email_headers_processed = analyze_headers(email_headers)
    email_text_processed = preprocess_text(email_text)
    combined_content = email_headers_processed + " " + email_text_processed

    # Handle File Upload
    if 'email_attachment' in request.files:
        file = request.files['email_attachment']
        attachment_content = analyze_attachment(file)
        combined_content += " " + preprocess_text(attachment_content)

    # Transform Content
    features = tf.transform([combined_content])

    # Make Predictions
    predictions = {
        'Random Forest': rf.predict(features)[0],
        'Multinomial Naive Bayes': mnb.predict(features)[0],
        'Bernoulli Naive Bayes': bnb.predict(features)[0],
    }

    # Convert Prediction Labels
    for model in predictions:
        predictions[model] = "Phishing" if predictions[model] == 0 else "Safe"

    log_prediction(combined_content, predictions)
    
    return render_template('result.html', predictions=predictions, email_text=combined_content)

@app.route('/feedback', methods=['POST'])
def feedback():
    email_text = request.form['email_text']
    feedback = request.form['feedback']
    logging.info(f'Feedback for Email: {email_text}, Feedback: {feedback}')
    return render_template('thank_you.html')

if __name__ == '__main__':
    app.run(debug=True)
