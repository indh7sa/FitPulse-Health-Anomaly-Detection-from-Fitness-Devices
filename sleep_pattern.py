import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

FORECAST_HORIZON = 7
USE_SYNTHETIC = True  

if USE_SYNTHETIC:
    np.random.seed(42)
    n_days = 90
    end_date = pd.to_datetime("2025-10-04")
    dates = pd.date_range(end=end_date, periods=n_days, freq="D")

    t = np.arange(n_days)
    baseline = 7.0
    weekly_amp = 0.8
    weekly_pattern = weekly_amp * np.sin(2 * np.pi * t / 7.0)
    noise = np.random.normal(scale=0.3, size=n_days)
    sleep_hours = baseline + weekly_pattern + noise

    df_sleep = pd.DataFrame({"ds": dates, "y": sleep_hours})

    t_future = np.arange(n_days, n_days + FORECAST_HORIZON)
    weekly_future = weekly_amp * np.sin(2 * np.pi * t_future / 7.0)
    noise_future = np.random.normal(scale=0.3, size=FORECAST_HORIZON)
    y_actual_future = baseline + weekly_future + noise_future
    df_actual_future = pd.DataFrame({"ds": pd.date_range(dates[-1]+pd.Timedelta(days=1),
                                                        periods=FORECAST_HORIZON),
                                     "y": y_actual_future})
else:
    df_sleep = pd.read_csv("sleep_data.csv")
    df_sleep["ds"] = pd.to_datetime(df_sleep["date"])
    df_sleep = df_sleep.set_index("ds").resample("D").mean().interpolate().reset_index()
    df_sleep = df_sleep.rename(columns={"sleep_hours": "y"})
    df_actual_future = None

m = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False,
            interval_width=0.95)
m.fit(df_sleep)
future = m.make_future_dataframe(periods=FORECAST_HORIZON)
forecast = m.predict(future)

if df_actual_future is not None:
    forecast["y_actual"] = pd.concat([df_sleep["y"].iloc[-0:], df_actual_future["y"]], ignore_index=True)[-FORECAST_HORIZON:]

if df_actual_future is not None:
    mae = mean_absolute_error(df_actual_future["y"], forecast["yhat"].tail(FORECAST_HORIZON))
    rmse = np.sqrt(mean_squared_error(df_actual_future["y"], forecast["yhat"].tail(FORECAST_HORIZON)))
    mape = np.mean(np.abs((df_actual_future["y"] - forecast["yhat"].tail(FORECAST_HORIZON)) / df_actual_future["y"])) * 100
    print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%")

fig, axes = plt.subplots(2,1, figsize=(12,8), sharex=True)
x_train = mdates.date2num(df_sleep["ds"].dt.to_pydatetime())
x_fore = mdates.date2num(forecast["ds"].dt.to_pydatetime())

axes[0].plot_date(x_train, df_sleep["y"], fmt="-", label="Observed")
axes[0].plot_date(x_fore, forecast["yhat"], fmt="--", label="Forecast")
axes[0].fill_between(x_fore, forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.2)
if "y_actual" in forecast:
    axes[0].scatter(x_fore[-FORECAST_HORIZON:], forecast["y_actual"].tail(FORECAST_HORIZON), color="red", label="Actual")
axes[0].legend(); axes[0].set_title("Full Sleep Forecast")

axes[1].plot_date(x_fore[-FORECAST_HORIZON:], forecast["yhat"].tail(FORECAST_HORIZON), fmt="--", label="Forecast")
axes[1].fill_between(x_fore[-FORECAST_HORIZON:], forecast["yhat_lower"].tail(FORECAST_HORIZON),
                     forecast["yhat_upper"].tail(FORECAST_HORIZON), alpha=0.2)
if "y_actual" in forecast:
    axes[1].scatter(x_fore[-FORECAST_HORIZON:], forecast["y_actual"].tail(FORECAST_HORIZON), color="red", label="Actual")
axes[1].legend(); axes[1].set_title("Zoomed: Forecast Horizon")

for ax in axes:
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
plt.tight_layout()
plt.show()

df_sleep["weekday"] = df_sleep["ds"].dt.day_name()
avg_sleep_by_day = df_sleep.groupby("weekday")["y"].mean().sort_values(ascending=False)
print("Average sleep by weekday:\n", avg_sleep_by_day)

trend_slope = (forecast["yhat"].iloc[-1] - forecast["yhat"].iloc[0]) / len(forecast)
print(f"Sleep trend over time: {'increasing' if trend_slope>0 else 'decreasing'}")

m.plot_components(forecast)
plt.show()
