from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('wqu.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def index():
    a=request.form['a']
    b=request.form['b']
    c=request.form['c']
    d=request.form['d']
    e=request.form['e']
    f=request.form['f']
    g=request.form['g']
    h=request.form['h']
    i=request.form['i']
    j=request.form['j']
    k=request.form['k']
    arr = np.array([[a,b,c,d,e,f,g,h,i,j,k]])
    pred = model.predict(arr)
    return render_template('result.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)
