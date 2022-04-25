import pandas as pd
import flask
from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

app = Flask(__name__)
app.add_url_rule('/photos/<path:filename>', endpoint='photos',
                 view_func=app.send_static_file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv(
        '/home/sc/Documents/nlp/spooky-author-identification/train.csv')
    import string
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    lemmatiser = WordNetLemmatizer()


    def text_process(tex):
        
        nopunct = [char for char in tex if char not in string.punctuation]
        nopunct = ''.join(nopunct)

        a = ''
        i = 0
        for i in range(len(nopunct.split())):
            b = lemmatiser.lemmatize(nopunct.split()[i], pos="v")
            a = a+b+' '

        return [word for word in a.split() if word.lower() not
                in stopwords.words('english')]
    from sklearn.preprocessing import LabelEncoder
    y = df['author']
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)
    X = df['text']
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    bow_transformer = CountVectorizer(analyzer=text_process).fit(X_train)
    text_bow_train = bow_transformer.transform(X_train)  
    text_bow_test = bow_transformer.transform(X_test) 

    from sklearn.naive_bayes import MultinomialNB

    model = MultinomialNB()

    model = model.fit(text_bow_train, y_train)
    if request.method == 'POST':
        author = request.form['author']
        msg = bow_transformer.transform([author])
        prediction = model.predict(msg)
        prediction = labelencoder.inverse_transform(prediction)
        output = prediction[0]

    return render_template('index.html', prediction_text='The author is  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
