

# NexRAN XApp Controller

This repository contains the NexRAN XApp Controller application that communicates with an open-source XApp (by OAIC) to control the Radio Access Network (RAN) and predict bus capacity based on data received from IoT devices and end-users. 

## Overview

The NexRAN XApp Controller is designed to interact with an OAIC-supported XApp for managing the RAN. The core functionality of this project is to predict the capacity in buses using data provided by IoT devices. This repository contains the main XApp server along with other supporting modules.

### Key Components

1. **`app.py`**: The central XApp server responsible for handling requests and communicating with the OAIC XApp. It gathers information from IoT devices and end-user requests to predict bus capacity and sends those predictions back through the XApp.
   
2. **`iot.py`**: This module gathers and processes data sent from IoT devices (such as sensors deployed in buses). The data is crucial for predicting the current capacity.
   
3. **`enduser.py`**: Acts as the interface between end-users and the XApp. It sends capacity prediction requests to the XApp and receives the results for further action.

4. **`xapp_curl.py`**: Provides utility functions for making CURL requests to the XApp API.

5. **`util.py`**: Contains helper functions used across the application.

### Data Flow
- The **IoT devices** continuously send data to the `iot.py` script.
- **End-users** query the XApp server through `enduser.py`, asking for bus capacity predictions.
- The **XApp server** (`app.py`) processes the data and communicates with the OAIC XApp to fetch the predicted capacity.

## Installation

To clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your-username/nexran-xapp-controller.git
cd nexran-xapp-controller
pip install -r requirements.txt
```

Make sure to have the OAIC XApp repository set up and running. You can find it here:
- OAIC NexRAN XApp GitHub: [https://github.com/openaicellular/nexran](https://github.com/openaicellular/nexran)

## Model Training

The prediction model used for bus capacity is trained in a separate repository. For details on model training, refer to:

- Model Training GitHub: [https://github.com/AnishPawar/ITU-WTSA-24-ML](https://github.com/AnishPawar/ITU-WTSA-24-ML)

Ensure that the trained models from the model training repository are available and linked to the XApp server for correct predictions.

## Usage

1. Run the OAIC XApp from the [NexRAN GitHub repository](https://github.com/openaicellular/nexran).
2. Launch the XApp Controller:
   ```bash
   python app.py
   ```
3. The XApp server will now accept requests from `enduser.py` and process the incoming data from `iot.py` for bus capacity prediction.

## Contributing

Feel free to fork this repository and contribute to its development by creating pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

