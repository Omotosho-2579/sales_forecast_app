import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

class SalesPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=0.2
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.output_layer(last_step)

def prepare_data(df, window_size=30):
    """Prepare data for LSTM training"""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(df[['sales']])
    
    sequences, targets = [], []
    for i in range(len(scaled) - window_size):
        sequences.append(scaled[i:i+window_size])
        targets.append(scaled[i+window_size])
    
    return np.array(sequences), np.array(targets), scaler

def create_sample_data(num_days=500):
    """Generate synthetic sales data"""
    dates = pd.date_range(start='2020-01-01', periods=num_days)
    base = np.linspace(50, 150, num_days)
    weekly_pattern = 25 * np.sin(2 * np.pi * np.arange(num_days) / 7)
    monthly_pattern = 15 * np.sin(2 * np.pi * np.arange(num_days) / 30)
    event_spikes = np.zeros(num_days)
    for day in [30, 90, 150, 210, 300]:
        if day < num_days:
            event_spikes[day] = 40
            if day+1 < num_days: 
                event_spikes[day+1] = 20
    random_noise = np.random.normal(0, 8, num_days)
    sales = base + weekly_pattern + monthly_pattern + event_spikes + random_noise
    sales = np.abs(sales)
    return pd.DataFrame({'date': dates, 'sales': sales})

def train_model(X, y, window_size=30, epochs=100, device='cpu'):
    """Train the LSTM model"""
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32), 
        torch.tensor(y_test, dtype=torch.float32)
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = SalesPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()
    
    return model, X_test, y_test

def make_forecast(model, scaler, data, window_size=30, forecast_days=60):
    """Generate future sales forecast"""
    model.eval()
    scaled_values = scaler.transform(data[['sales']])
    current_window = scaled_values[-window_size:]
    forecast = []
    
    with torch.no_grad():
        for _ in range(forecast_days):
            inputs = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0)
            pred = model(inputs)
            forecast.append(pred.numpy()[0][0])
            current_window = np.vstack((current_window[1:], pred.numpy()))
    
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast.flatten()

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model performance"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(X_test)):
            inputs = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0)
            pred = model(inputs)
            predictions.append(pred.numpy()[0][0])
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = scaler.inverse_transform(y_test.reshape(-1, 1))
    mae = mean_absolute_error(actuals, predictions)
    
    return predictions.flatten(), actuals.flatten(), mae