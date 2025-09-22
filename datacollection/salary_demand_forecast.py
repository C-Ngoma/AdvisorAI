import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 1. Load your dataset
df = pd.read_csv('datacollection/ai_job_dataset_merged.csv')
df['posting_date'] = pd.to_datetime(df['posting_date'])

# 2. Aggregate data: average salary and job demand per day
daily = df.groupby('posting_date').agg({
    'salary_usd': 'mean',
    'job_id': 'count'
}).rename(columns={'job_id': 'demand'})

# 3. Visualize the trends
plt.figure(figsize=(12, 5))
plt.plot(daily.index, daily['salary_usd'], label='Average Salary (USD)')
plt.plot(daily.index, daily['demand'], label='Job Demand (Postings)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.title(' Job Salary & Demand Over Time')
plt.tight_layout()
plt.show()

# 4. ARIMA Forecast: Salary
# ARIMA requires stationary data; we'll use simple (1,1,1) for demonstration
salary_model = ARIMA(daily['salary_usd'], order=(1,1,1))
salary_fit = salary_model.fit()
salary_forecast = salary_fit.forecast(steps=12)  # Predict next 12 periods (days)
print("\nSalary Forecast for Next 12 Days:")
print(salary_forecast)

# 5. ARIMA Forecast: Demand
demand_model = ARIMA(daily['demand'], order=(1,1,1))
demand_fit = demand_model.fit()
demand_forecast = demand_fit.forecast(steps=12)
print("\nJob Demand Forecast for Next 12 Days:")
print(demand_forecast)

# 6. Visualize Forecasts (appending predictions to the original series)
future_dates = pd.date_range(start=daily.index[-1], periods=13, freq='D')[1:]  # Next 12 dates

plt.figure(figsize=(12, 5))
plt.plot(daily.index, daily['salary_usd'], label='Historical Salary')
plt.plot(future_dates, salary_forecast, label='Forecasted Salary', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Salary (USD)')
plt.title('Salary Forecast (ARIMA)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(daily.index, daily['demand'], label='Historical Demand')
plt.plot(future_dates, demand_forecast, label='Forecasted Demand', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Job Postings')
plt.title('Job Demand Forecast (ARIMA)')
plt.legend()
plt.tight_layout()
plt.show()


