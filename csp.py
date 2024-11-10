import datetime
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import streamlit as st

st.title("Stock Price Prediction App")

start_date = datetime.date(2019, 10, 8)
end_date = datetime.date.today()

stock_symbol = st.text_input("Enter Stock Symbol (e.g., STOCKNAME.NS): ")

days_to_predict = st.number_input("Days to Predict into the Future:", min_value=1, max_value=365, value=30)

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

            data = data.dropna()

            data.reset_index(inplace=True)
            data_lstm = data[['Date', 'Adj Close']]

            # Preprocessing data
            data_lstm['Adj Close'] = data_lstm['Adj Close'].astype(float)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data_lstm['Adj Close'].values.reshape(-1, 1))

            # Create training and testing datasets
            train_data = scaled_data[:int(0.8 * len(data_lstm))]
            test_data = scaled_data[int(0.8 * len(data_lstm)):]

            def create_dataset(dataset, time_step=1):
                X, Y = [], []
                for i in range(len(dataset) - time_step - 1):
                    a = dataset[i:(i + time_step), 0]
                    X.append(a)
                    Y.append(dataset[i + time_step, 0])
                return np.array(X), np.array(Y)

            time_step = 60
            X_train, y_train = create_dataset(train_data, time_step)
            X_test, y_test = create_dataset(test_data, time_step)

            # Reshape input to be [samples, time steps, features] which is required for LSTM
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))

            # Compile and train the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=1, epochs=1)

            # Prediction
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)

            # Transform back to original scale
            train_predict = scaler.inverse_transform(train_predict)
            test_predict = scaler.inverse_transform(test_predict)

            # Prepare the data for plotting
            train_predict_plot = np.empty_like(scaled_data)
            train_predict_plot[:, :] = np.nan
            train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

            test_predict_plot = np.empty_like(scaled_data)
            test_predict_plot[:, :] = np.nan
            test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1, :] = test_predict

            # Plot the results
            plt.figure(figsize=(10, 5))
            plt.plot(data_lstm['Date'], data_lstm['Adj Close'], label='Actual Data', color='black')

            # Ensure x and y dimensions match before plotting
            valid_train_dates = data_lstm['Date'][time_step:len(train_predict) + time_step]
            valid_test_dates = data_lstm['Date'][len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1]

            if len(valid_train_dates) == len(train_predict):
                plt.plot(valid_train_dates, train_predict, label='Train Prediction', color='blue')
            if len(valid_test_dates) == len(test_predict):
                plt.plot(valid_test_dates, test_predict, label='Test Prediction', color='red')

            plt.title(f"Stock Price Prediction for {stock_symbol.upper()}", fontsize=14)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Stock Price", fontsize=12)
            plt.legend(loc='upper left')
            plt.xticks(rotation=45)

            st.pyplot(plt)

            st.write("Historical Data:")
            st.dataframe(data)

            # Prepare future prediction data
            last_60_days = scaled_data[-60:]
            future_input = last_60_days.reshape(1, -1)
            future_input = future_input.reshape((1, time_step, 1))

            future_predictions = []
            for _ in range(days_to_predict):
                next_pred = model.predict(future_input)
                future_predictions.append(next_pred[0][0])
                future_input = np.append(future_input[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            future_dates = pd.date_range(start=end_date, periods=days_to_predict + 1).tolist()

            forecast_table = pd.DataFrame(future_predictions, columns=['Predicted Price'])
            forecast_table['Date'] = future_dates[1:]  # exclude the first
