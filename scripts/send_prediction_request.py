import requests
import json
import numpy as np

def convert_ndarray_to_list(data):
    """
    Convert ndarray in a dictionary to lists.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            elif isinstance(value, dict):
                convert_ndarray_to_list(value)
    elif isinstance(data, list):
        for i in range(len(data)):
            if isinstance(data[i], np.ndarray):
                data[i] = data[i].tolist()
            elif isinstance(data[i], dict):
                convert_ndarray_to_list(data[i])
    return data

def generate_data():
    ## Generate sample data
    #data = [
    #    {
    #        "date": "2021-08-01",
    #        "PRECTOT": 0.1,
    #        "PS": 1013,
    #        "QV2M": 0.02,
    #        "T2M": 295,
    #        "T2MDEW": 290,
    #        "T2M_MAX": 298,
    #        "T2M_MIN": 290,
    #        "T2M_RANGE": 8,
    #        "WS10M": 3,
    #        "WS10M_MAX": 5,
    #        "WS10M_MIN": 2,
    #        "WS10M_RANGE": 3,
    #        "WS50M": 4,
    #        "WS50M_MAX": 6,
    #        "WS50M_MIN": 3,
    #        "WS50M_RANGE": 3
    #    }
    #]
    #return data

    num_rows = 1

    prectot = np.random.uniform(1, 40, size=num_rows)  
    ps = np.random.uniform(66, 105, size=num_rows)  
    qv2m = np.random.uniform(0.1, 22.5, size=num_rows)  
    t2m = np.random.uniform(-38.6, 40.3, size=num_rows)  
    t2mdew = np.random.uniform(-41, 27, size=num_rows)  
    t2m_max = np.random.uniform(-30, 50, size=num_rows)  
    t2m_min = np.random.uniform(-45, 32, size=num_rows)  
    t2m_range = np.random.uniform(0, 30, size=num_rows)  
    t2m_wet = np.random.uniform(-38, 27, size=num_rows)  
    ts = np.random.uniform(-41, 43, size=num_rows)  
    ws10m = np.random.uniform(0, 17, size=num_rows)  
    ws10m_max = np.random.uniform(0, 25, size=num_rows)  
    ws10m_min = np.random.uniform(0, 15, size=num_rows)  
    ws10m_range = np.random.uniform(0, 22, size=num_rows)  
    ws50m = np.random.uniform(0, 17, size=num_rows)  
    ws50m_max = np.random.uniform(0, 25, size=num_rows)  
    ws50m_min = np.random.uniform(0, 15, size=num_rows)  
    ws50m_range = np.random.uniform(0, 22, size=num_rows)  
    fips = np.random.uniform(1001, 56000, size=num_rows)  

    # Generate random data for categorical columns
    years = np.random.randint(2024, 2030, size=num_rows)  # Random years between 2000 and 2022
    months = np.random.randint(1, 13, size=num_rows)  # Random months between 1 and 12
    days = np.random.randint(1, 29, size=num_rows)  # Random days between 1 and 28 (to avoid issues with February)

    date = []
    for i, year in enumerate(years):
        date.append("{}-{}-{}".format(year, months[i], days[i]))

    data = {
        'PRECTOT': prectot,
        'PS': ps, 
        'QV2M':qv2m, 
        'T2M':t2m, 'T2MDEW':t2mdew, 
        'T2M_MAX':t2m_max, 'T2M_MIN':t2m_min, 
        'T2M_RANGE':t2m_range, 
        'WS10M':ws10m, 
        'WS10M_MAX':ws10m_max, 
        'WS10M_MIN':ws10m_min, 
        'WS10M_RANGE':ws10m_range, 
        'WS50M':ws50m, 
        'WS50M_MAX':ws50m_max, 
        'WS50M_MIN':ws50m_min, 
        'WS50M_RANGE':ws50m_range, 
        'fips':fips, 
        'TS':ts,
        'T2MWET':t2m_wet,
        'date':date
    }


    return data

def send_request(data):
    url = "http://localhost:5000/predict"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        print("Response received:")
        print(response.json())
    else:
        print(f"Failed to get response: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    data = generate_data()
    data = convert_ndarray_to_list(data)
    send_request(data)