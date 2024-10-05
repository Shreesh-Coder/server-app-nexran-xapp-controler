import requests
import json
from datetime import datetime

def send_end_user_data(base_url, data):
    url = f'{base_url}/end_user'
    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    if response.status_code == 201:
        print(f"Success: {response.json()}")
    else:
        print(f"Error: {response.status_code}, {response.json()}")

if __name__ == '__main__':
    base_url = 'http://172.16.0.1:5000'  # Flask server URL

    # Example End-User Data
    end_user_data = {
        "ue_id": "001010123456789",
        "bus_query_time": "2023-10-05 14:45:00",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "location": [37.7750, -122.4183],  # User's location (latitude, longitude)
        "user_query_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "src": "Station A",
        "dst": "Station B"
    }

    # Send the end-user data
    send_end_user_data(base_url, end_user_data)
