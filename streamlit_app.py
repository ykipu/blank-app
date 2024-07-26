import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Stock Price Predicitior App")

st.text_input("Enter the stock ID", 'NVDA')

from datetime import datetime
end= datetime.now()
start= datetime(end.year-20,end.monyh,end.day)

nvidia_data = yf.download(stock, start, end)

model = looad_model("Latest_stock_price_model.keras")
st.subheader("stock Data")
st.write(nvidia_data)

splittinf_len = int(len(nvidia_data)*0.7)
x_test = pd.DataFrame(nvidia_dat.clode[splittin_len:])

def plot_graph(figsize, values, full_data):
    fig= plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.close, 'b')
    return fig
