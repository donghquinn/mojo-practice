import matplotlib.pyplot as plt
import pandas as pd

def visualize_stock(data: pd.DataFrame):
    plt.plot(data["Actual"], label="Raw Data")
    plt.plot(data["Predicted"], label="Predicted")
    
    plt.legend(loc="best")
    plt.show()