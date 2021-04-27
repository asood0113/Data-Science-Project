from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def index():
    sepal_length = request.form['sepal_length']
    sepal_width= request.form['sepal_width']
    petal_length = request.form['petal_length']
    petal_width = request.form['petal_width']
    arr = np.array([[sepal_length,sepal_width, petal_length, petal_width]])
    pred = model.predict(arr)
    return render_template('result.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True,host='localhost',port=3000)
