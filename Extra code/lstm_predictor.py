
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from torch.utils.data import Dataset, DataLoader

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

class VehicleDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

def predict(ckpt = 'lstm_model_checkpoint.pth', data = 'extended_path.csv'):
    input_size = 20
    hidden_size = 50
    output_size = 1

    loaded_model = LSTMModel(input_size, hidden_size, output_size)
    loaded_model.load_state_dict(torch.load(ckpt))
    loaded_model.eval()

    df_new = pd.read_csv(data)
    df_new[['longitude', 'latitude']] = df_new['vehicle_position'].str.extract(r'\((.*), (.*)\)').astype(float)

    features = ['route_id', 'longitude', 'latitude', 'day', 'time_of_day']
    target = 'available_seats'

    categorical_features = ['route_id', 'day', 'time_of_day']
    numerical_features = ['longitude', 'latitude']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ]
    )

    X_processed = preprocessor.fit_transform(df_new[features])

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    X_lstm = X_processed.reshape((X_processed.shape[0], 1, X_processed.shape[1]))
    X_tensor = torch.tensor(X_lstm, dtype=torch.float32)

    inference_dataset = VehicleDataset(X_tensor)
    inference_loader = DataLoader(inference_dataset, batch_size=32, shuffle=False)

    loaded_model.eval()
    predictions = []

    with torch.no_grad():
        for X_batch in inference_loader:
            outputs = loaded_model(X_batch)
            predictions.append(outputs.detach().numpy())

    predictions = np.concatenate(predictions).flatten()

    return predictions.astype(np.uint8)
