import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from visualize import visualize_stock
from sklearn.model_selection import train_test_split
from lstm import StockPredictor
import time

def stock():
    start = time.time()
    data = pd.read_csv("stock.csv")
    data

    price = data["Close"].values.astype(float)

    # Scaling - 최대 / 최소 설정
    scaler_x = MinMaxScaler()
    price = scaler_x.fit_transform(price.reshape(-1, 1))

    input_dim = 1
    hidden_dim = 64
    num_layers = 2

    X, y = [], []

    seq_length = 10

    for i in range(len(price) - seq_length):
        X.append(price[i:i+seq_length])
        y.append(price[i+seq_length])

    X = np.array(X)
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = StockPredictor(input_dim, hidden_dim, num_layers)
    
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

    # 테스트
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        mse = mean_squared_error(y_test, test_outputs)
        print(f'Mean Squared Error on Test Data: {mse}')

    predicted_prices = scaler_x.inverse_transform(test_outputs.numpy())
    actual_prices = scaler_x.inverse_transform(y_test.numpy())
    
    data = {
        "Actual": [round(val[0], 3) for val in actual_prices],
        "Predicted": [round(val[0], 3) for val in predicted_prices]
    }

    result_frame = pd.DataFrame(data)
    
    result_frame["Error"] = result_frame["Actual"] - result_frame["Predicted"]
    
    print(result_frame)
    
    end = time.time()
    
    print("Elapsed Time: {}".format(end - start))
    
    visualize_stock(result_frame)
    
    return result_frame
