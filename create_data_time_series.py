import pandas as pd
import numpy as np

# Generate sample time series supplier performance data
np.random.seed(42)
date_range = pd.date_range(start='2020-01-01', periods=1000, freq='D')
supplier_ids = np.random.randint(1, 10, 1000)
time_series_data = pd.DataFrame({
    'Date': date_range,
    'Supplier_ID': supplier_ids,
    'Delivery_Time': np.random.randint(1, 30, 1000),
    'Quality_Rating': np.random.randint(1, 11, 1000),
    'Quantity_Delivered': np.random.randint(1, 1000, 1000)
})

time_series_data.to_csv('time_series_data.csv', index=False)
print("Sample time series data generated and saved as time_series_data.csv")
