from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('housepp.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('hpp.html')


@app.route('/predict', methods=['POST'])
def index():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('hpp.html', prediction_text='House price should be $ {}'.format(output))


if __name__ == "__main__":
    app.run()