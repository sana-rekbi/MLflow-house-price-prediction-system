import json

import requests

# URL of the MLflow prediction server
url = "http://127.0.0.1:8000/invocations"

# Sample input data for prediction
# Replace the values with the actual features your model expects
input_data = {
    "dataframe_records": [
        {
            "Order": 1,
            "PID": 5286,
            "MS SubClass": 20,
            "Lot Frontage": 80.0,
            "Lot Area": 9600,
            "Overall Qual": 5,
            "Overall Cond": 7,
            "Year Built": 1961,
            "Year Remod/Add": 1961,
            "Mas Vnr Area": 0.0,
            "BsmtFin SF 1": 700.0,
            "BsmtFin SF 2": 0.0,
            "Bsmt Unf SF": 150.0,
            "Total Bsmt SF": 850.0,
            "1st Flr SF": 856,
            "2nd Flr SF": 854,
            "Low Qual Fin SF": 0,
            "Gr Liv Area": 1710.0,
            "Bsmt Full Bath": 1,
            "Bsmt Half Bath": 0,
            "Full Bath": 1,
            "Half Bath": 0,
            "Bedroom AbvGr": 3,
            "Kitchen AbvGr": 1,
            "TotRms AbvGrd": 7,
            "Fireplaces": 2,
            "Garage Yr Blt": 1961,
            "Garage Cars": 2,
            "Garage Area": 500.0,
            "Wood Deck SF": 210.0,
            "Open Porch SF": 0,
            "Enclosed Porch": 0,
            "3Ssn Porch": 0,
            "Screen Porch": 0,
            "Pool Area": 0,
            "Misc Val": 0,
            "Mo Sold": 5,
            "Yr Sold": 2010,
        }
    ]
}

# Convert the input data to JSON format
json_data = json.dumps(input_data)

# Set the headers for the request
headers = {"Content-Type": "application/json"}

# Send the POST request to the server
response = requests.post(url, headers=headers, data=json_data)

# Check the response status code
if response.status_code == 200:
    # If successful, print the prediction result
    prediction = response.json()
    print("Prediction:", prediction)
else:
    # If there was an error, print the status code and the response
    print(f"Error: {response.status_code}")
    print(response.text)
