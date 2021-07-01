from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pickle
import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer




app = Flask(__name__)

# Simple route for receiving data
@app.route('/english', methods=['POST'])
def eng():
    data = request.form['msg']
    # remove punctuations
    message = [char for char in data if char not in string.punctuation]
# join the message after removing   
    message = ''.join(message)
# remove stop words
    message = [word for word in message.split() if word.lower() not in stopwords.words('english')]

    vectorizer = pickle.load(open('vectorizer', 'rb'))
    message = vectorizer.transform([data])

    transformer = pickle.load(open('transformer', 'rb'))
    message = transformer.transform(message)
    print(message)

    with open('naive_bayes','rb') as mod:
        model = pickle.load(mod)
    prediction = model.predict(message)
    pred = ""
    if prediction=='ham':
        pred = "Not Spam"
    else: 
        pred = "Spam"

    return pred


@app.route('/urdu', methods=['POST'])
def urdu():
    data = request.form['msg']
    # remove punctuations
    message = [char for char in data if char not in string.punctuation]
    # join the message after removing   
    message = ''.join(message)
    # remove stop words
    message = [word for word in message.split() if word.lower() not in stopwords.words('english')]

    # make vectors
    vectorizer = pickle.load(open('vectorizer_urdu', 'rb'))
    message = vectorizer.transform([data])

    # make frequency vectors 
    transformer = pickle.load(open('transformer_urdu', 'rb'))
    message = transformer.transform(message)

    # Predict the outcome
    with open('naive_bayes_urdu','rb') as mod:
        model = pickle.load(mod)
    prediction = model.predict(message)
    pred = ""
    if prediction=='ham':
        pred = "Not Spam"
    else: 
        pred = "Spam"

    # Return Prediction
    return pred    
    


if __name__ == '__main__':
  app.run(host='127.0.0.1', port=5000, debug=True)
 