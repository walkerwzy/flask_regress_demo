from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import flash
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('house.html')

# Avg. Area Income / Avg. Area House Age / 
# Avg. Area Number of Rooms / Avg. Area Number of Bedrooms /
# Area Population
@app.route('/predict/<float:income>/<float:age>/<float:rooms>/<float:bedrooms>/<float:population>')
def predict(income, age, rooms, bedrooms, population):
    model = joblib.load('best.mod')
    data  = np.array([income, age, rooms, bedrooms, population]).reshape(-1, 5)
    res   = model.predict(data)
    res   = {"predict price:": res.tolist()[0]}
#     flash(res)
    return jsonify(res)


app.secret_key = 'super secret key'

if __name__ == '__main__':
	app.run()