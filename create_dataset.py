import pandas as pd
import numpy as np

# Generate sample supplier performance data
np.random.seed(42)
supplier_data = pd.DataFrame({
    'Supplier_ID': np.random.randint(1, 100, 1000),
    'Part_ID': np.random.randint(1, 50, 1000),
    'Delivery_Time': np.random.randint(1, 30, 1000),
    'Quality_Rating': np.random.randint(1, 11, 1000),
    'Quantity_Delivered': np.random.randint(1, 1000, 1000),
    'Historical_Performance_Score': np.random.rand(1000) * 100,
    'Approval_Status': np.random.choice(['Approved', 'Rejected'], 1000)
})

# Generate sample anomaly detection data
anomaly_data = pd.DataFrame({
    'Supplier_ID': np.random.randint(1, 100, 500),
    'Part_ID': np.random.randint(1, 50, 500),
    'Delivery_Date': pd.date_range(start='1/1/2023', periods=500, freq='D'),
    'Delivery_Time': np.random.randint(1, 30, 500),
    'Quality_Rating': np.random.randint(1, 11, 500),
    'Quantity_Delivered': np.random.randint(1, 1000, 500)
})

supplier_data.to_csv('supplier_data.csv', index=False)
anomaly_data.to_csv('anomaly_data.csv', index=False)
