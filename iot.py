import requests
import json

def send_iot_sensor_data(base_url, data):
    url = f'{base_url}/iot_sensor'
    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    if response.status_code == 201:
        print(f"Success: {response.json()}")
    else:
        print(f"Error: {response.status_code}, {response.json()}")

if __name__ == '__main__':
    base_url = 'http://172.16.0.1:5000'  # Flask server URL

    iot_sensor_data = {
        "Date": "2023-10-05",
        "Time": "14:30:00",
        "Mode": "Bus",
        "Code": "B123",
        "ESIM_ID": "ESIM456",
        "Source": "Station A",
        "Destination": "Station B",
        "Intermediate_Stop": "Station C",
        "Location": "37.7749,-122.4194",  # Latitude and Longitude
        "Capacity": "50",
        "Filled_Seats": "30"
    }

    # Send the IoT sensor data
    send_iot_sensor_data(base_url, iot_sensor_data)
