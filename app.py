from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime

final_pipeline = joblib.load('model/final_model.pkl')
brand_encoder = joblib.load('model/brand_encoder.pkl')

app = Flask(__name__)

def preprocess_input(data):
    """Preprocess user input into the format expected by the model."""
    data['horsepower'] = data['engine'].apply(lambda x: float(re.search(r"(\d+\.\d+)HP", x).group(1)) if re.search(r"(\d+\.\d+)HP", x) else np.nan)
    data['engine_type'] = data['engine'].str.extract(r"(\d+\.\d+L \d+ Cylinder)")[0]

    data['accident'] = data['accident'].astype(int)
    data['clean_title'] = data['clean_title'].astype(int)

    current_year = datetime.now().year
    data['car_age'] = current_year - data['model_year']
    data['mileage_per_year'] = data['milage'] / data['car_age']
    data['mileage_per_year'] = data['mileage_per_year'].replace([np.inf, -np.inf], np.nan)
    data['milage_car_age_interaction'] = data['milage'] * data['car_age']
    data['log_milage'] = np.log1p(data['milage'])
    data['log_mileage_per_year'] = np.log1p(data['mileage_per_year'])
    data['cylinders'] = data['engine_type'].str.extract(r'(\d+) Cylinder').astype(float)

    luxury_brands = ['Porsche', 'Lamborghini', 'Bentley']
    data['brand_luxury'] = data['brand'].apply(lambda x: 1 if x in luxury_brands else 0)

    data['brand_encoded'] = brand_encoder.transform(data['brand'])

    data = data.drop(columns=['engine', 'brand'])

    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = {
            'brand': request.form['brand'],
            'model': request.form['model'],
            'model_year': int(request.form['model_year']),
            'milage': float(request.form['milage']),
            'fuel_type': request.form['fuel_type'],
            'transmission': request.form['transmission'],
            'ext_col': request.form['ext_col'],
            'int_col': request.form['int_col'],
            'accident': int(request.form['accident']),
            'clean_title': int(request.form['clean_title']),
            'engine': request.form['engine']
        }

        input_data = pd.DataFrame([form_data])

        preprocessed_data = preprocess_input(input_data)

        predicted_price = final_pipeline.predict(preprocessed_data)

        return render_template(
            'index.html',
            prediction=f"Estimated Price: ${predicted_price[0]*10:,.2f}",
            form_data=form_data
        )
    except Exception as e:
        return render_template(
            'index.html',
            prediction=f"Error: {str(e)}",
            form_data=request.form
        )

if __name__ == '__main__':
    app.run(debug=True)
