import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from lstm import LstmModel


def forecast():
    data = pd.read_csv("combined.csv")
    
    data = data.fillna(0)
    
    data["date"] = pd.to_datetime(data["date"])
    
    X = np.array(data["date"])
    
    solar_radiation = data["solar_radiation"]
    features = ["temperature", "rain", "humidity", "pressure", "visibility", "wind_speed", "wind_direction", "total_cloud"]
    
    input_dim = 8
    hidden_dim = 6
    num_layer = 48
    
    trainX, trainy, testX, testy = train_test_split(X, np.array(solar_radiation), train_size=0.7, random_state=42)
    
    X_train = torch.tensor(trainX, dtype=torch.float32)
    y_train = torch.tensor(trainy, dtype=torch.float32)
    X_test = torch.tensor(testX, dtype=torch.float32)
    y_test = torch.tensor(testy, dtype=torch.float32)

    scaler = MinMaxScaler()

    model = LstmModel(input_dim, hidden_dim, num_layer)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 100
    
    for epoch in range(num_epochs):
        outputs = model(X_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
    model.eval()
    
    with torch.no_grad():
        test_outputs = model(X_test)
        
        mse = mean_squared_error(y_test, test_outputs)
        mae = mean_absolute_error(y_test, test_outputs)
        mape = mean_absolute_percentage_error(y_test, test_outputs)
        rmse = np.sqrt(mse)
        
        print("MSE: {}, MAE: {}, RMSE: {}, MAPE: {}".format(round(mse, 3),  round(mae, 3), round(rmse, 3), round(mape, 3)))
    