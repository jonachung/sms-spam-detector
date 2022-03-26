import os
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import *
from sklearn.svm import *
import pandas as pd

app = Flask(__name__)
global Classifier
global Vectorizer

# load data
data = pd.read_csv('spam.csv', encoding='latin-1')
train_data = data[:4400] # 4400 items
test_data = data[4400:] # 1172 items

# train model
Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(train_data.v2)
Classifier.fit(vectorize_text, train_data.v1)


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        

        message = request.form.get("message")
        error = ''
        predict = ''

        global Classifier
        global Vectorizer
        try:
            if len(message) > 0:
                vectorize_message = Vectorizer.transform([message])
                predict = Classifier.predict(vectorize_message)[0]
                if predict == "ham":
                    predict = "not spam"
        except BaseException as inst:
            error = str(type(inst).__name__) + ' ' + str(inst)
        return jsonify(
                message=message,
                predict=predict, error=error)

    return render_template("form.html")

if __name__ == '__main__':
    app.run()
