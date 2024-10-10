from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from concurrent.futures import ThreadPoolExecutor
import threading
import requests
import time
from xapp_curl import NexranClient
import numpy as np
import joblib
import pandas as pd  # For data manipulation
from datetime import datetime
from util import *
import torch
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from torch.utils.data import Dataset, DataLoader
import numpy as np
from lstm_predictor import LSTMModel
app = Flask(__name__)
api = Api(app)

# Thread pool to handle concurrent requests
executor = ThreadPoolExecutor(max_workers=5)
data_lock = threading.Lock()

# Data storage
iot_sensor_data = []  # List to store IoT sensor data as dictionaries
end_user_data = {}    # Dictionary to store end-user data


# Load LSTM model and preprocessor once at the start
lstm_model = LSTMModel(input_size=20, hidden_size=50, output_size=1)
lstm_model.load_state_dict(torch.load('lstm_model_checkpoint.pth'))
lstm_model.eval()

# Create the preprocessor
categorical_features = ['route_id', 'day', 'time_of_day']
numerical_features = ['longitude', 'latitude']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

# Initialize NexranClient
nexran_client = None

from flask import request
from flask_restful import Resource
from datetime import datetime
import threading

# Assuming iot_sensor_data and data_lock are defined elsewhere in your code
iot_sensor_data = []
data_lock = threading.Lock()

class IoTSensorService(Resource):
    def post(self):
        """
        Handles POST requests from IoT Sensor UEs to submit transport data.
        """
        data = request.get_json()
        required_fields = ['Date', 'Time', 'Mode', 'Code', 'ESIM_ID', 'Route_ID', 'Source', 'Destination',
                           'Intermediate_Stop', 'Location', 'Capacity', 'Filled_Seats', 'day', 'time_of_day']

        # Check if all required fields are in the data
        if not all(field in data for field in required_fields):
            missing_fields = [field for field in required_fields if field not in data]
            return {'message': f'Missing fields in IoT Sensor data: {", ".join(missing_fields)}'}, 400

        try:
            # Parse the Location into longitude and latitude
            location_str = data.get('Location', "0,0")  # Default to "0,0" if Location is missing
            longitude, latitude = map(float, location_str.strip("()").split(','))

            # Add parsed longitude and latitude to the data
            data['longitude'] = longitude
            data['latitude'] = latitude
        except (ValueError, AttributeError):
            return {'message': 'Invalid Location format. Expected format: "(longitude, latitude)"'}, 400

        with data_lock:
            # Add a timestamp to the data
            data['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Convert Route_ID to the format expected by iot_df
            data['route_id'] = data.pop('Route_ID', 'Unknown Route')

            # Append the processed data to the global iot_sensor_data list
            iot_sensor_data.append(data)

            # Optionally, save the data to a CSV file or database for persistence
            # pd.DataFrame(iot_sensor_data).to_csv('iot_sensor_data.csv', index=False)

            return {'message': 'IoT Sensor data received successfully'}, 201


class EndUserService(Resource):
    def post(self):
        """
        Handles POST requests from End-User UEs to request transport predictions.
        """
        data = request.get_json()
        required_fields = ['ue_id', 'bus_query_time', 'timestamp', 'location', 'user_query_time', 'src', 'dst']

        if not all(field in data for field in required_fields):
            return {'message': 'Missing fields in End-User data'}, 400

        ue_id = data['ue_id']

        with data_lock:
            # Store the end-user data
            end_user_data[ue_id] = data

        # Perform prediction by passing lstm_model and preprocessor to perform_prediction
        predictions = perform_prediction(data, lstm_model, preprocessor)


        # Allocate UE to the fast slice
        with data_lock:
            allocate_ue_slice(ue_id)

        # Send predictions back to the user
        result = {
            'ue_id': ue_id,
            'predictions': predictions,
            'message': 'Transport options predicted and UE allocated to normal slice.'
        }

        return result, 201

class IoTSensorData(Resource):
    def get(self):
        """
        Handles GET requests to retrieve all IoT Sensor data (for debugging purposes).
        """
        with data_lock:
            return jsonify(iot_sensor_data)

class EndUserData(Resource):
    def get(self):
        """
        Handles GET requests to retrieve all End-User data (for debugging purposes).
        """
        with data_lock:
            return jsonify(end_user_data)

# Slice Switch API Resource
class SwitchSlice(Resource):
    def post(self):
        # Extract data from the incoming request
        data = request.get_json()
        ue_id = data.get('ue_id')
        slice_name = data.get('slice_name')

        # Validate the input data
        if not ue_id or not slice_name:
            return {"error": "ue_id and slice_name are required."}, 400
        
        # Perform the slice switch, which also handles UE deletion from other slices
        slice_switch(ue_id, slice_name)
        
        return {"message": "UE slice binding successfully switched."}, 200

# Adding resources to the API
api.add_resource(IoTSensorService, '/iot_sensor')
api.add_resource(EndUserService, '/end_user')
api.add_resource(IoTSensorData, '/iot_sensor_data')
api.add_resource(EndUserData, '/end_user_data')
api.add_resource(SwitchSlice, '/slice_swtich')

def run_app():
    app.run(host='0.0.0.0', port=2345, debug=True, use_reloader=False)

def perform_prediction(end_user_request, lstm_model, preprocessor):
    """
    Use the IoT sensor data and LSTM model to predict available transport modes and capacities near the user.
    """
    with data_lock:
        if not iot_sensor_data:
            return {'message': 'No IoT sensor data available for prediction.'}

        # Convert IoT sensor data to DataFrame
        iot_df = pd.DataFrame(iot_sensor_data)

    # Parse user's location and query time
    user_location = end_user_request['location']  # Should be a list [longitude, latitude]
    user_query_time = datetime.strptime(end_user_request['user_query_time'], '%Y-%m-%d %H:%M:%S')

    # Filter IoT data based on proximity and time
    proximity_threshold = 0.09  # Approximate degrees (~1km)

    def is_within_proximity(iot_loc, user_loc):
        lat_diff = abs(iot_loc[1] - user_loc[1])
        long_diff = abs(iot_loc[0] - user_loc[0])
        return lat_diff <= proximity_threshold and long_diff <= proximity_threshold

    available_options = []
    for idx, row in iot_df.iterrows():
        # Handle Location field
        try:
            iot_loc = tuple(map(float, row['Location'].strip("()").split(',')))
        except (ValueError, AttributeError, KeyError):
            continue  # Skip this row if Location is invalid or missing

        # Check proximity
        if is_within_proximity(iot_loc, user_location):
            # Prepare the features
            features = {
                'route_id': row.get('route_id', 1),
                'longitude': iot_loc[0],
                'latitude': iot_loc[1],
                'day': row.get('day', 'Day 1'),
                'time_of_day': row.get('time_of_day', 'Afternoon')
            }

            # Transform the features using the fitted preprocessor
            X_processed = preprocessor.transform(pd.DataFrame([features]))

            if hasattr(X_processed, "toarray"):
                X_processed = X_processed.toarray()

            # Reshape for LSTM input
            X_lstm = X_processed.reshape((X_processed.shape[0], 1, X_processed.shape[1]))
            X_tensor = torch.tensor(X_lstm, dtype=torch.float32)

            # Perform prediction
            with torch.no_grad():
                prediction = lstm_model(X_tensor).numpy().flatten()[0]

            # Calculate available seats based on predicted capacity
            option = {
                'Mode': row.get('Mode', 'Bus'),
                'Capacity': int(prediction),
                'Filled_Seats': int(row.get('Filled_Seats', 0)),
                'Available_Seats': int(prediction) - int(row.get('Filled_Seats', 0)),
                'Location': row.get('Location', '0,0')
            }
            available_options.append(option)

    if not available_options:
        return {'message': 'No transport options available near you at this time.'}

    return available_options




def allocate_ue_slice(ue_id, slice='normal'):
    """
    Use NexranClient to allocate the UE to the fast slice.
    """
    # First, ensure the UE exists in xApp
    time.sleep(2)
    response = nexran_client.post_ue({'imsi': ue_id})
    if response.status_code  in [200, 201]:
        print(f'Successfully added UE {ue_id} to xApp')
    elif response.status_code == 403 and 'already exists' in response.text:
        print(f'UE {ue_id} already exists in xApp')
    else:
        print(f'Failed to add UE {ue_id} to xApp: {response.text}')
        return

    # Ensure the fast slice exists
    slice_name = slice
    slice_response = nexran_client.get_slice(slice_name)
    if slice_response.status_code not in [200, 201]:
        # Create the fast slice if it doesn't exist
        slice_data = {
            'name': slice_name,
            'allocation_policy': {
                'type': 'proportional',
                'share': 1024 if slice_name == 'fast' else 256
            }
        }
        create_slice_resp = nexran_client.post_slice(slice_data)
        if create_slice_resp.status_code  in [200, 201]:
            print(f'Slice {slice_name} created.')
        else:
            print(f'Failed to create slice {slice_name}: {create_slice_resp.text}')
            return

    # Bind the UE to the fast slice
    bind_response = nexran_client.post_slice_ue_binding(slice_name, ue_id)
    if bind_response.status_code  in [200, 201]:
        print(f'UE {ue_id} bound to slice {slice_name}.')
    elif bind_response.status_code == 403 and 'already bound' in bind_response.text:
        print(f'UE {ue_id} is already bound to slice {slice_name}.')
    else:
        print(f'Failed to bind UE {ue_id} to slice {slice_name}: {bind_response.text}')

def slice_switch(ue_id, slice_name):
    
    slices_response = nexran_client.get_slices()
   
    # Parse the response JSON content into a dictionary
    slices = slices_response.json()
    
    slice_names = [slice_data['name'] for slice_data in slices['slices']]
    print(slice_names)
    for slice in slice_names:
        if slice_name != slice:
            print(delete_ue_from_slice(ue_id, slice))
            print(slice)
    

    allocate_ue_slice(ue_id, slice_name)


def delete_ue_from_slice(ue_id, slice_name):
    # Log that the deletion process is starting
    print(f"Starting to delete UE {ue_id} from slice {slice_name}")

    # Delete UE slice binding
    response = nexran_client.delete_slice_ue_binding(slice_name, ue_id)
    print(f"DELETE request to slice {slice_name} for UE {ue_id} returned status {response.status_code}")
    time.sleep(1)
    # Check if the response from the DELETE request was successful

    print(f"UE {ue_id} successfully unbound from slice {slice_name}")
    
    # Attempt to delete the UE from the system
    delete_ue_response = nexran_client.delete_ue(ue_id)
    print(f"Deleted UE {ue_id} from the system: {delete_ue_response}")
    time.sleep(1)
    
    # Attempt to delete the slice from the system
    nexran_client.delete_slice(slice_name)
    print(f"Deleted slice {slice_name} from the system.")
    time.sleep(1)
    
    return  204







if __name__ == '__main__':
    nexran_ip = get_nexran_ip()
    base_url='http://localhost:8000/v1'
    
    if nexran_ip:
        # Now you can use this IP for further processes or REST API calls
        # For example, you can initialize NexranClient with this IP
        base_url = f"http://{nexran_ip}:8000/v1"
        print(f"Base URL for NEXRAN xApp: {base_url}")

    nexran_client= NexranClient(base_url)  # Update with the correct xApp URL
    # Start the Flask server
    run_app()
