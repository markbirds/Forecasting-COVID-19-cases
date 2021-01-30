from utils.utils import display_prediction

def app():
  days_to_forecast = 365*5
  display_prediction(days_to_forecast)

if __name__ == "__main__":
  app()