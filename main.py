from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Downloading NLTK libraries
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the TF-IDF vectorizer and the model
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_sms = request.form['sms']

        # Preprocess the input
        transform_sms = transform_text(input_sms)

        # Vectorize the input
        vector_input = tfidf.transform([transform_sms])

        # Predict
        result = model.predict(vector_input)[0]
        # Convert result to string
        if result == 1:
            result_text = "SPAM"
        else:
            result_text = "NOT SPAM"

        # Return prediction result
        return render_template('result.html', result=result_text)

if __name__ == '__main__':
    app.run(debug=True)
