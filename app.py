import os
import pickle
from flask import Flask, render_template, redirect, url_for, request, session, jsonify
from functools import wraps
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import google.oauth2.credentials
import base64
import tensorflow as tf
from tensorflow import keras
import numpy as np
from bs4 import BeautifulSoup
import re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import requests
from google.oauth2.credentials import Credentials

# Set environment variable to allow OAuth2 to work over HTTP (for development only)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'credentials' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def credentials_to_dict(credentials):
    return {'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes}

def get_gmail_service():
    if 'credentials' not in session:
        return redirect(url_for('login'))
    
    credentials = google.oauth2.credentials.Credentials(**session['credentials'])
    
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            return redirect(url_for('login'))
    
    session['credentials'] = credentials_to_dict(credentials)
    return build('gmail', 'v1', credentials=credentials)

def get_emails(service, num_emails=100):
    results = service.users().messages().list(userId='me', maxResults=num_emails).execute()
    messages = results.get('messages', [])
    emails = []
    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
        payload = msg['payload']
        headers = payload['headers']
        subject = next((header['value'] for header in headers if header['name'].lower() == 'subject'), 'No Subject')
        body = ''
        if 'parts' in payload:
            parts = payload['parts']
            for part in parts:
                if part['mimeType'] == 'text/plain':
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
        else:
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        emails.append({"subject": subject, "body": body})
    return emails

def preprocess_email(email_text):
    # Remove HTML tags
    soup = BeautifulSoup(email_text, "html.parser")
    text = soup.get_text()
    
    # Remove numbers and punctuation
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words])
    
    # Stem words
    stemmer = SnowballStemmer("english")
    words = text.split()
    text = ' '.join([stemmer.stem(word) for word in words])
    
    return text

def predict_spam_for_emails(emails, model, tokenizer, max_sequence_len):
    results = []
    for email in emails:
        subject = email['subject']
        body = email['body']
        full_text = f"Subject: {subject}\n\n{body}"
        
        processed_email = preprocess_email(full_text)
        if not processed_email:
            results.append({"subject": subject, "prediction": "error", "confidence": 0.0})
            continue
        
        sequence = tokenizer.texts_to_sequences([processed_email])
        padded = keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_sequence_len, padding='post')
        prediction = model.predict(padded)[0][0]
        
        result = 'spam' if prediction > 0.5 else 'ham'
        results.append({"subject": subject, "prediction": result, "confidence": float(prediction)})
    
    return results

@app.route('/')
@app.route('/login')
def login():
    flow = Flow.from_client_secrets_file(
        'credentials.json', SCOPES)
    flow.redirect_uri = url_for('oauth2callback', _external=True)
    
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true')
    
    session['state'] = state
    return redirect(authorization_url)

@app.route('/oauth2callback')
def oauth2callback():
    state = session['state']
    flow = Flow.from_client_secrets_file(
        'credentials.json', SCOPES, state=state)
    flow.redirect_uri = url_for('oauth2callback', _external=True)
    
    # Use the authorization response from the callback
    authorization_response = request.url
    flow.fetch_token(authorization_response=authorization_response)

    credentials = flow.credentials
    session['credentials'] = credentials_to_dict(credentials)
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/fetch-emails')
@login_required
def fetch_emails():
    if 'credentials' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    credentials = Credentials(**session['credentials'])
    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
            session['credentials'] = credentials_to_dict(credentials)
        else:
            return jsonify({'error': 'Invalid credentials'}), 401

    # Load the model and necessary components
    model = keras.models.load_model('spam_detection.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('max_sequence_len.txt', 'r') as f:
        max_sequence_len = int(f.read())

    service = build('gmail', 'v1', credentials=credentials)
    emails = get_emails(service)
    predictions = predict_spam_for_emails(emails, model, tokenizer, max_sequence_len)
    return jsonify(predictions)

@app.route('/logout')
def logout():
    if 'credentials' in session:
        credentials = Credentials(**session['credentials'])
        if credentials and credentials.valid:
            try:
                # Revoke the token
                requests.post('https://oauth2.googleapis.com/revoke',
                    params={'token': credentials.token},
                    headers = {'content-type': 'application/x-www-form-urlencoded'})
            except Exception as e:
                print(f"An error occurred while revoking the token: {e}")

        # Clear the user's session
        session.clear()
        
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)