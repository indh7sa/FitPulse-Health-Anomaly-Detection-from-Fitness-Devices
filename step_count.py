import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

np.random.seed(42)
days = 120
date_rng = pd.date_range(start="2025-01-01", periods=days, freq="D")
steps = 8000 + np.random.normal(0, 500, days) 

steps[29:37] -= 1500    
steps[59:62] -= 3000   
steps[89] += 4000       

df = pd.DataFrame({"ds": date_rng, "y": steps})

holidays = pd.DataFrame({
    "holiday": ["vacation"] * 8 + ["sick"] * 3 + ["marathon"] * 1,
    "ds": pd.to_datetime(
        ["2025-01-30","2025-01-31","2025-02-01","2025-02-02",
         "2025-02-03","2025-02-04","2025-02-05","2025-02-06",
         "2025-03-01","2025-03-02","2025-03-03",
         "2025-03-31"]
    ),
    "lower_window": 0,
    "upper_window": 0
})
print("Holidays DataFrame:")
print(holidays.head())


model_no_holiday = Prophet()
model_no_holiday.fit(df)
future_no_holiday = model_no_holiday.make_future_dataframe(periods=30)
forecast_no_holiday = model_no_holiday.predict(future_no_holiday)

model_with_holiday = Prophet(holidays=holidays)
model_with_holiday.fit(df)
future_with_holiday = model_with_holiday.make_future_dataframe(periods=30)
forecast_with_holiday = model_with_holiday.predict(future_with_holiday)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
model_no_holiday.plot(forecast_no_holiday, ax=plt.gca())
plt.title("Forecast WITHOUT Holidays")

plt.subplot(2, 1, 2)
model_with_holiday.plot(forecast_with_holiday, ax=plt.gca())
plt.title("Forecast WITH Holidays")

plt.tight_layout()
plt.show()

holiday_effects = forecast_with_holiday[["ds", "holidays", "yhat"]].tail(40)
print("\nRecent forecast (last 40 days):")
print(holiday_effects.head())

print("\n=== Interpretation ===")
print("Holidays clearly modify the predicted step count around event days.")
print("- Vacation and Sick days cause noticeable dips.")
print("- Marathon day shows a significant spike.")
print("Among them, the 'Sick' event had the largest negative effect on step count.")
