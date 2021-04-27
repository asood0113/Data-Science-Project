from flask import Flask, render_template, request
import pickle
import numpy as np
import h5py
#from keras.models import load_model

#model = load_model('RFC.pkl')
model = pickle.load(open('salest1.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('indexofsal.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('indexofsal.html', prediction_text='Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True,host='localhost',port=8000)