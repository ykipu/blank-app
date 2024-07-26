import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import date
import numpy as np

# Set up the Streamlit app
st.title('Stock Price Prediction App')

# Sidebar inputs
st.sidebar.header('User Input Parameters')

def user_input_features():
    ticker = st.sidebar.text_input("Ticker", 'AAPL')
    start_date = st.sidebar.date_input("Start Date", date(2015, 1, 1))
    end_date = st.sidebar.date_input("End Date", date.today())
    return ticker, start_date, end_date

ticker, start_date, end_date = user_input_features()

# Fetch stock data
st.subheader('Stock Data')
data = yf.download(ticker, start=start_date, end=end_date)
st.write(data.tail())

# Preprocess data
data['Date'] = data.index
data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].map(pd.Timestamp.toordinal)

# Split data into training and testing sets
X = np.array(data['Date']).reshape(-1, 1)
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Display predictions
st.subheader('Model Predictions vs Actual')
results = pd.DataFrame({'Date': X_test.flatten(), 'Actual': y_test, 'Predicted': predictions})
results['Date'] = pd.to_datetime(results['Date'], origin='1970-01-01')
results = results.sort_values('Date')

st.write(results)

# Plot the results
st.subheader('Price Prediction Plot')
st.line_chart(results.set_index('Date'))

# Evaluate the model
st.subheader('Model Evaluation')
mse = mean_squared_error(y_test, predictions)
st.write(f'Mean Squared Error: {mse:.2f}')
