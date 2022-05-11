
import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Asset forecast model by Synkarnia')

stocks = ('MSFT','GOOG','AAPL','INFY','BNB-USD',
	 'BTC-USD','ETH-USD','DOGE-USD','SOL-USD')
selected_stock = st.selectbox('Select an asset for prediction (The price is in USD)', stocks)

n_years = st.slider('Years of prediction:', 1, 2)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Fetching data..')

st.subheader('Historical Data')
st.write(data.tail())

# raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} year(s)')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast attributes")
fig2 = m.plot_components(forecast)
st.write(fig2)
