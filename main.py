# Loading the Dataset
import pandas as pd
import re

data = "HistoricalData_1689225905434.csv"
df = pd.read_csv(data)

# Cleaning the Dataset
df.fillna(0)
df = df.astype(str)
df["Close/Last"] = df["Close/Last"].str.replace("$","")
df["Volume"] = df["Volume"].str.replace(",","")
df["Open"] = df["Open"].str.replace("$","")
df["High"] = df["High"].str.replace("$","")
df["Low"] = df["Low"].str.replace("$","")

df["Close/Last"].astype(float)
df["Volume"].astype(float)
df["Open"].astype(float)
df["High"].astype(float)
df["Low"].astype(float)

df['Date'] = df['Date'].astype('datetime64[ns]')
df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')

# Downloading the New Dataset
df.to_csv("fluor_data.csv", sep=',', index=False, encoding='utf-8')