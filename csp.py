import datetime
import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
import streamlit as st
import time
from ta import add_all_ta_features
from ta.utils import dropna

# Start timer
start_time = time.time()

st.title("Stock Price Prediction App")

# Update start_date to 2015
start_date = datetime.date(2015, 1, 1)
end_date = datetime.date.today()

stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL): ")

# Update days_to_predict to 5 years (365 * 5 = 1825 days)
days_to_predict = st.number_input("Days to Predict into the Future:", min_value=1, max_value=1825, value=1825)

if stock_symbol:
    if start_date > end_date:
        st.error("End Date cannot be earlier than Start Date!")
    else:
        try:
            data = yf.download(stock_symbol, start=start_date, end=end_date)

            if data.empty or len(data.dropna()) < 2:
                st.error("Not enough data available. Please adjust the date range or check the stock symbol.")
                st.stop()

        except (ConnectionError, ValueError) as e:
            st.error(f"Error fetching data: {e}")
            st.stop()
        else:
            st.success("Data Fetched Successfully!")

            data = dropna(data)
            data.reset_index(inplace=True)
            data = add_all_ta_features(
                data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )

            data_prophet = data[['Date', 'Adj Close']].rename(columns={'Date': 'ds', 'Adj Close': 'y'})

            model = Prophet()
            model.fit(data_prophet)

            future_dates = model.make_future_dataframe(periods=days_to_predict)
            forecast = model.predict(future_dates)

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.plot(data_prophet['ds'], data_prophet['y'], label='Actual Data', color='black')
            ax.plot(forecast['ds'], forecast['yhat'], label='Predicted Data', color='blue')

            ax.set_title(f"Stock Price Prediction for {stock_symbol.upper()}", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Stock Price ", fontsize=12)
            ax.legend(loc='upper left')

            plt.xticks(rotation=45)

            st.pyplot(fig)
            st.write("Historical Data:")
            st.dataframe(data)

            forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast_table.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
            forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m-%d')
            st.write("Future Forecast:")
            st.dataframe(forecast_table)

            # Fetch financial information
            stock_info = yf.Ticker(stock_symbol).info
            st.write("Stock Information:")
            
            # Handle potential missing data gracefully
            market_cap = stock_info.get('marketCap', 'N/A')
            dividend_yield = stock_info.get('dividendYield', 'N/A')
            total_revenue = stock_info.get('totalRevenue', 'N/A')
            earnings_growth = stock_info.get('earningsGrowth', 'N/A')

            st.write(f"**Market Cap:** {market_cap}")
            st.write(f"**Dividend Yield:** {dividend_yield}")
            st.write(f"**Total Revenue:** {total_revenue}")
            st.write(f"**Earnings Growth:** {earnings_growth}")

            # End timer and display time taken
            end_time = time.time()
            time_taken = end_time - start_time
            st.write(f"Time taken to load the site and perform predictions: {time_taken:.2f} seconds")
