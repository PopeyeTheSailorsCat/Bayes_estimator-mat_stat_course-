import pandas as pd


def get_data():
    white_wine_data = pd.read_csv('data/winequality-white.csv', sep=';')
    red_wine_data = pd.read_csv('data/winequality-red.csv', sep=';')
    # print(white_wine_data.head())
    return white_wine_data, red_wine_data
