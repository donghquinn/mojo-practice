import matplotlib.pyplot as plt
import pandas as pd

def visualize_stock(data: pd.DataFrame, fileName: str):
    plt.title("Stock Prediction")
    
    plt.plot(data["Actual"], label="Raw Data", c="g")
    plt.plot(data["Predicted"], label="Predicted", c="orange")
    
    plt.legend(loc="best")
    plt.savefig("graph/{}.png".format(fileName))