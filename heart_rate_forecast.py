import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error

FORECAST_HORIZON = 14  
USE_SYNTHETIC = True   

if USE_SYNTHETIC:
    np.random.seed(42)
    end_date = pd.to_datetime("2025-10-04")
    n_train = 60
    dates_train = pd.date_range(end=end_date, periods=n_train, freq="D")

    t = np.arange(n_train)
    baseline = 72.0
    trend = 0.02 * t
    weekly_amp = 3.0
    weekly = weekly_amp * np.sin(2 * np.pi * t / 7.0)
    noise = np.random.normal(scale=1.2, size=n_train)
    y_train = baseline + trend + weekly + noise

    df_train = pd.DataFrame({"ds": dates_train, "y": y_train})

    t_future = np.arange(n_train, n_train + FORECAST_HORIZON)
    trend_future = 0.02 * t_future
    weekly_future = weekly_amp * np.sin(2 * np.pi * t_future / 7.0)
    noise_future = np.random.normal(scale=1.2, size=FORECAST_HORIZON)
    y_actual_future = baseline + trend_future + weekly_future + noise_future
    df_actual_future = pd.DataFrame({"ds": pd.date_range(dates_train[-1] + pd.Timedelta(days=1),
                                                        periods=FORECAST_HORIZON),
                                     "y": y_actual_future})
else:
    df = pd.read_csv("my_hr.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("D").mean().interpolate() 
    df_train = df.rename(columns={"heart_rate": "y"}).reset_index().rename(columns={"date": "ds"})
    df_actual_future = None 


forecast_df = None
model_used = None

try:
    from prophet import Prophet
    model_used = "Prophet"
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False,
                interval_width=0.95)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=FORECAST_HORIZON)
    forecast = m.predict(future)
    forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(FORECAST_HORIZON)
    forecast_df = forecast_df.rename(columns={"yhat": "yhat_forecast"})
except Exception:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model_used = "Holt-Winters"
    hw = ExponentialSmoothing(df_train["y"], trend="add", seasonal="add", seasonal_periods=7)
    hw_fit = hw.fit(optimized=True)
    yhat = hw_fit.forecast(FORECAST_HORIZON)
    resid_std = np.std(hw_fit.resid)
    lower, upper = yhat - 1.96*resid_std, yhat + 1.96*resid_std
    forecast_df = pd.DataFrame({"ds": pd.date_range(df_train["ds"].iloc[-1] + pd.Timedelta(days=1),
                                                    periods=FORECAST_HORIZON),
                                "yhat_forecast": yhat,
                                "yhat_lower": lower,
                                "yhat_upper": upper})


if df_actual_future is not None:
    forecast_df["y_actual"] = df_actual_future["y"].values

print("Model used:", model_used)


if df_actual_future is not None:
    mae = mean_absolute_error(df_actual_future["y"], forecast_df["yhat_forecast"])
    rmse = np.sqrt(mean_squared_error(df_actual_future["y"], forecast_df["yhat_forecast"]))
    mape = np.mean(np.abs((df_actual_future["y"] - forecast_df["yhat_forecast"])
                          / df_actual_future["y"])) * 100
    print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%")

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

x_train = mdates.date2num(df_train["ds"].dt.to_pydatetime())
x_fore = mdates.date2num(forecast_df["ds"].dt.to_pydatetime())

axes[0].plot_date(x_train, df_train["y"], fmt="-", label="Train")
axes[0].plot_date(x_fore, forecast_df["yhat_forecast"], fmt="--", label="Forecast")
axes[0].fill_between(x_fore, forecast_df["yhat_lower"], forecast_df["yhat_upper"], alpha=0.2)
if "y_actual" in forecast_df:
    axes[0].scatter(x_fore, forecast_df["y_actual"], c="red", label="Actual")
axes[0].legend(); axes[0].set_title("Full Forecast")

axes[1].plot_date(x_fore, forecast_df["yhat_forecast"], fmt="--", label="Forecast")
axes[1].fill_between(x_fore, forecast_df["yhat_lower"], forecast_df["yhat_upper"], alpha=0.2)
if "y_actual" in forecast_df:
    axes[1].scatter(x_fore, forecast_df["y_actual"], c="red", label="Actual")
axes[1].legend(); axes[1].set_title("Zoom: Forecast Horizon")

for ax in axes:
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

plt.tight_layout()
plt.show()

day_index = 6  
day67_row = forecast_df.iloc[day_index]
print(f"\nDay 67 forecast ({day67_row['ds'].date()}): {day67_row['yhat_forecast']:.2f} bpm")
print(f"95% CI ≈ [{day67_row['yhat_lower']:.2f}, {day67_row['yhat_upper']:.2f}] bpm")

if model_used == "Prophet":
    m.plot_components(forecast)
    plt.show()
