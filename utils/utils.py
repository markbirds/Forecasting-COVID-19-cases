from tensorflow.keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import json
import pickle

# the api gives updated total confirmed cases in each day
url = 'https://covid-api.mmediagroup.fr/v1/history?country=Philippines&status=confirmed'
x = requests.get(url)
loaded_ph_data = json.loads(x.text)['All']['dates']
dates = list(loaded_ph_data.keys())
cases = list(loaded_ph_data.values())
# latest date in the covid data
latest_date = datetime.strptime(dates[0], '%Y-%m-%d').date()

# loading saved models
loaded_model = load_model('output/forecasting_covid_19_ph_model.h5')
loaded_scaler = pickle.load(open('output/scaler.pickle','rb'))

# returns dataframe with historical records (new cases everyday)
def new_cases_everyday():
  # get the number of new cases everyday
  today = np.array(cases[:-1])
  yesterday = np.array(cases[1:])
  # reversing the numpy array
  new_cases = np.flip(today - yesterday)
  cases_df = pd.DataFrame({'Date':dates[::-1][1:],'New cases':new_cases})
  cases_df['Date'] = pd.to_datetime(cases_df['Date'])
  # retrieve and scale down the new cases in the last 60 days(timesteps required for prediction)
  return cases_df


cases_df = new_cases_everyday()
# new cases in the last 60 days
new_cases_60_days = loaded_scaler.transform(np.array(cases_df.tail(60)['New cases']).reshape(-1, 1))

def forecast(days):
  future_predictions = []
  # reshaping into shape [batch, timesteps, feature]
  reshaped_new_cases_60_days = new_cases_60_days.reshape(1,60,1)
  for i in range(days):
    # predicts and adding prediction to list
    prediction = loaded_model.predict(reshaped_new_cases_60_days)
    future_predictions.append(max(0,prediction[0][0])) 
     # appending prediction to new cases and updating new cases in the last 60 days
    reshaped_new_cases_60_days = np.append(reshaped_new_cases_60_days,prediction.reshape(1,1,1),axis=1)
    reshaped_new_cases_60_days = reshaped_new_cases_60_days[:,1:]
  return np.array(future_predictions)


def display_prediction(days_to_forecast):
  # we include the latest date new cases in order to connect predicted new cases to current cases
  prediction_dates = pd.date_range(latest_date,periods=days_to_forecast+1)
  predicted_new_cases = loaded_scaler.inverse_transform(forecast(days_to_forecast).reshape(-1,1)).astype(int).reshape(-1)
  # inserting new cases of the latest date
  predicted_new_cases = np.insert(predicted_new_cases,0,np.array(cases_df.tail(1)['New cases']),axis=0)
  prediction_df = pd.DataFrame({'Date':prediction_dates,'Predicted new cases':predicted_new_cases})

  last_row = prediction_df.tail(1)
  last_date_prediction = last_row['Date'].dt.strftime('%B %d, %Y').values[0]
  last_date_prediction_new_cases = last_row['Predicted new cases'].values[0]
  
  # display plot
  plt.figure()
  plt.title(f'{days_to_forecast} day forecast')
  plt.plot_date(cases_df['Date'],cases_df['New cases'],fmt='-',lw=2)
  plt.plot_date(prediction_df['Date'],prediction_df['Predicted new cases'],fmt='-',lw=2)
  plt.legend(['True new cases', 'Predicted new cases'])
  plt.xlabel(f'{last_date_prediction}: {last_date_prediction_new_cases:,} new cases')
  plt.ylabel('New cases')
  plt.grid()
  plt.show()



