import pandas as pd
import pickle
import os
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

model_path = os.getenv("MODEL_PATH", "models/model_xgboost.bin")
dv_path = os.getenv("DV_PATH", "models/dv.bin")

def load_model_and_dv(model_path, dv_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(dv_path, 'rb') as f:
        dv = pickle.load(f)
    return model, dv

def preprocess_input_data(input_data, dv):
    # Preprocess input data similarly to training data
    input_data = input_data.dropna()
    
    input_data['year'] = pd.DatetimeIndex(input_data['date']).year
    input_data['month'] = pd.DatetimeIndex(input_data['date']).month
    input_data['day'] = pd.DatetimeIndex(input_data['date']).day
    
    numerical_column_list = ['PRECTOT','PS','QV2M','T2M','T2MDEW','T2M_MAX','T2M_MIN','T2M_RANGE','WS10M','WS10M_MAX','WS10M_MIN','WS10M_RANGE','WS50M','WS50M_MAX','WS50M_MIN','WS50M_RANGE']
    categorical_column_list = ['year','month','day']
    
    input_data = input_data.drop("fips", axis=1)
    input_data = input_data.drop("date", axis=1)
    input_data = input_data.drop("TS",axis=1)
    input_data = input_data.drop("T2MWET",axis=1)
    
    input_dicts = input_data[categorical_column_list + numerical_column_list].to_dict(orient='records')
    X_input = dv.transform(input_dicts)
    
    return X_input

@app.route('/predict', methods=['POST'])
#def predict(input_data, model_path=model_path, dv_path=dv_path):
def predict():
    input_data = request.json
    input_data = pd.DataFrame(input_data)
    # Load the trained model and DictVectorizer
    model, dv = load_model_and_dv(model_path, dv_path)
    
    # Load and preprocess input data
    X_input = preprocess_input_data(input_data, dv)
    
    # Make predictions
    predictions = model.predict(X_input)
    
    #return predictions
    return jsonify({'predictions': predictions.tolist()})

#def create_random_example(num_rows=1):
#    # Define the column lists
#    #numerical_column_list = [
#    #    'PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2M_MAX', 'T2M_MIN', 
#    #    'T2M_RANGE', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 
#    #    'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE', 'fips', 'TS', 'T2MWET'
#    #]    
#
#    # Generate random data for numerical columns
#    #numerical_data = {col: np.random.rand(num_rows) for col in numerical_column_list}
#
#    prectot = np.random.uniform(1, 40, size=num_rows)  
#    ps = np.random.uniform(66, 105, size=num_rows)  
#    qv2m = np.random.uniform(0.1, 22.5, size=num_rows)  
#    t2m = np.random.uniform(-38.6, 40.3, size=num_rows)  
#    t2mdew = np.random.uniform(-41, 27, size=num_rows)  
#    t2m_max = np.random.uniform(-30, 50, size=num_rows)  
#    t2m_min = np.random.uniform(-45, 32, size=num_rows)  
#    t2m_range = np.random.uniform(0, 30, size=num_rows)  
#    t2m_wet = np.random.uniform(-38, 27, size=num_rows)  
#    ts = np.random.uniform(-41, 43, size=num_rows)  
#    ws10m = np.random.uniform(0, 17, size=num_rows)  
#    ws10m_max = np.random.uniform(0, 25, size=num_rows)  
#    ws10m_min = np.random.uniform(0, 15, size=num_rows)  
#    ws10m_range = np.random.uniform(0, 22, size=num_rows)  
#    ws50m = np.random.uniform(0, 17, size=num_rows)  
#    ws50m_max = np.random.uniform(0, 25, size=num_rows)  
#    ws50m_min = np.random.uniform(0, 15, size=num_rows)  
#    ws50m_range = np.random.uniform(0, 22, size=num_rows)  
#    fips = np.random.uniform(1001, 56000, size=num_rows)  
#
#    numerical_data = {
#        'PRECTOT': prectot,
#        'PS': ps, 
#        'QV2M':qv2m, 
#        'T2M':t2m, 'T2MDEW':t2mdew, 
#        'T2M_MAX':t2m_max, 'T2M_MIN':t2m_min, 
#        'T2M_RANGE':t2m_range, 
#        'WS10M':ws10m, 
#        'WS10M_MAX':ws10m_max, 
#        'WS10M_MIN':ws10m_min, 
#        'WS10M_RANGE':ws10m_range, 
#        'WS50M':ws50m, 
#        'WS50M_MAX':ws50m_max, 
#        'WS50M_MIN':ws50m_min, 
#        'WS50M_RANGE':ws50m_range, 
#        'fips':fips, 
#        'TS':ts,
#        'T2MWET':t2m_wet
#    }
#
#    # Generate random data for categorical columns
#    years = np.random.randint(2024, 2030, size=num_rows)  # Random years between 2000 and 2022
#    months = np.random.randint(1, 13, size=num_rows)  # Random months between 1 and 12
#    days = np.random.randint(1, 29, size=num_rows)  # Random days between 1 and 28 (to avoid issues with February)
#    
#    categorical_data = {
#        'year': years,
#        'month': months,
#        'day': days
#    }
#    
#    # Combine the numerical and categorical data into a single DataFrame
#    data = {**numerical_data, **categorical_data}
#    df = pd.DataFrame(data)
#    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
#    df = df.drop("year",axis=1)
#    df = df.drop("month",axis=1)
#    df = df.drop("day",axis=1)
#
#    
#    return df

if __name__ == "__main__":
    #input_data_path = 'path_to_input_data.csv'
    #input_data = create_random_example()
    #predictions = predict(input_data)
    #print(f"The Drought score is: {predictions[0]}")
    app.run(host='0.0.0.0', port=5000)