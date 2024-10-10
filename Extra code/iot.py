import requests
import json
import pandas as pd
import time

def send_iot_sensor_data(base_url, data):
    url = f'{base_url}/iot_sensor'
    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    if response.status_code == 201:
        print(f"Success: {response.json()}")
    else:
        print(f"Error: {response.status_code}, {response.json()}")

def read_and_send_data_from_csv(base_url, csv_file):
    # Load CSV file
    csv_data = pd.read_csv(csv_file)
    
    # Loop through each row and send the data
    for index, row in csv_data.iterrows():
        # Prepare the IoT sensor data, filling missing fields with default values
        iot_sensor_data = {
            "Date": "2023-10-05",  # Assuming current date for all rows
            "Time": "14:30:00",  # Default time, can be modified if available in dataset
            "Mode": "Bus",  # Assuming all entries are for Bus
            "Code": "B123",  # Placeholder code
            "ESIM_ID": "ESIM456",  # Placeholder ESIM ID
            "Route_ID": str(row.get("route_id", "Unknown Route")),  # Route ID from CSV
            "Source": row.get("source", "Unknown Source"),  # Source from CSV
            "Destination": row.get("destination", "Unknown Destination"),  # Destination from CSV
            "Intermediate_Stop": row.get("next_stop", "Unknown Stop"),  # Next stop from CSV
            "Location": row.get("vehicle_position", "0,0"),  # Vehicle position from CSV
            "Capacity": "50",  # Assuming capacity is 50
            "Filled_Seats": str(row.get("available_seats", 0)),  # Available seats from CSV
            "day": row.get("day", "Unknown Day"),  # Day from CSV
            "time_of_day": row.get("time_of_day", "Unknown Time of Day")  # Time of day from CSV
        }

        time.sleep(2)
        # Send the IoT sensor data
        send_iot_sensor_data(base_url, iot_sensor_data)

if __name__ == '__main__':
    base_url = 'http://localhost:2345'  # Flask server URL
    csv_file = '/home/oran11/server-app-nexran-xapp-controler/extended_path.csv'  # Replace with the path to your CSV

    # Read data from CSV and send to server
    read_and_send_data_from_csv(base_url, csv_file)