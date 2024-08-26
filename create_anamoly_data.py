import pandas as pd
import numpy as np

# Generate sample anomaly detection data
np.random.seed(42)
date_range = pd.date_range(start='2020-01-01', periods=1000, freq='D')
supplier_ids = np.random.randint(1, 10, 1000)
anomaly_data = pd.DataFrame({
    'Date': date_range,
    'Supplier_ID': supplier_ids,
    'Delivery_Time': np.random.randint(1, 30, 1000),
    'Quality_Rating': np.random.randint(1, 11, 1000),
    'Quantity_Delivered': np.random.randint(1, 1000, 1000)
})

anomaly_data.to_csv('anomaly_data.csv', index=False)
print("Sample anomaly detection data generated and saved as anomaly_data.csv")
