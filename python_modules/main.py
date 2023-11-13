from multiRegression import multiRegression
import lstm.lstm_stock as ls
import rnn.rnn_stock as rn


def main():
    multiRegression("combined.csv")


def lstm_stock_predict():
    ls.stock()


def rnn_stock_predict():
    rn.stock()


lstm_stock_predict()
