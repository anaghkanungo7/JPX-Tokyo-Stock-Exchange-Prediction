import numpy as np
import pandas as pd
import mplfinance

print("Hello World!")

path = "dataset/train_files/stock_prices.csv"


df = pd.read_csv(path)
# print(df.head)
print(df.columns)
print(df["Target"])


mplfinance.plot(data=df)
