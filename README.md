# Rain Predictor

A machine learning project that predicts the chance of rain in the next 3 hours at any location, using pressure, humidity and temperature data.

**Live app:** https://rain-predictor-dnsigznead3xugv7ktp8bx.streamlit.app/#rain-predictor

## What it does
- Takes a latitude and longitude.
- Downloads the latest hourly weather from Open-Meteo.
- Builds features and gives the probability of rain in the next 3 hours.

## Data
- Hourly data from the Open-Meteo Historical API (2018 to 2026): temperature, humidity, dew point, surface pressure, rain.
- The location is set with `LATITUDE` and `LONGITUDE` at the top of the notebook, so it works for any place.

## Method
1. Target: 1 if any of the next 3 hours has 0.1 mm or more rain, else 0.
2. Features: current pressure, humidity, temperature, dew point gap, changes over 1 to 12 hours, recent rain, and time of day and month.
3. Split by time: train 2018-2023, validation 2024, test 2025 onwards. No random split, so there is no leakage from the future.
4. Models: persistence rule, logistic regression, LightGBM.
5. Metrics: precision, recall, F1 and ROC-AUC. Accuracy is not used because only about 15% of hours have rain.

## Results (test set)
| Model | F1 | ROC-AUC |
|---|---|---|
| Persistence rule (rain recently = rain soon) | 0.630 | 0.783 |
| Logistic regression | 0.619 | 0.898 |
| LightGBM | 0.634 | 0.907 |
| LightGBM without rain history | 0.561 | 0.882 |

## What I learned
- Recent rain is the strongest clue. It gives about 57% of the model's feature importance.
- Without any rain history, the model still reaches ROC-AUC 0.882 using humidity, pressure and dew point. So the atmosphere alone has real forecasting skill.
- Pressure level mattered more than short-term pressure change in my data.
- F1 at threshold 0.5 was only about equal to persistence. The model's main gain is in ranking hours by rain risk (ROC-AUC).

## Limits
- Trained on one location. Results at other climates are not tested yet.
- Training used historical reanalysis data, and the live app uses forecast-system data. They are close but not identical.
- This is a learning project, not an official forecast.

## Next steps
- Test the model on other climates and report how well it transfers.
- Add a real sensor node (ESP32 + BME280) and collect my own data.

## Files
- `app.py`: Streamlit app
- `rain_model.txt`: trained LightGBM model
- `requirements.txt`: libraries
- `rain_predictor.ipynb`: training notebook (add it, see below)
