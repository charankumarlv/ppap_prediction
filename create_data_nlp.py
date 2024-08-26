import pandas as pd
import numpy as np

# Generate sample supplier feedback and audit report data
np.random.seed(42)
supplier_ids = np.random.randint(1, 100, 500)
feedback_data = pd.DataFrame({
    'Supplier_ID': supplier_ids,
    'Feedback': np.random.choice([
        'The supplier delivered on time and the quality was excellent.',
        'Delivery was delayed and the quality of parts was below expectations.',
        'Supplier met all requirements but communication could be improved.',
        'Parts were delivered ahead of schedule but some items were damaged.',
        'Outstanding performance in all aspects. Highly recommended.',
        'Quality of parts was satisfactory but delivery was inconsistent.',
        'Supplier failed to meet the quality standards and delivery was late.'
    ], 500),
    'Audit_Report': np.random.choice([
        'Audit revealed several non-conformances in the quality management system.',
        'All processes are compliant with industry standards. No major issues found.',
        'Minor issues were noted in the documentation process but overall compliance is good.',
        'Significant improvements needed in the supply chain management process.',
        'Supplier has a robust quality control system in place. No issues found.',
        'Several areas of non-compliance were identified, requiring immediate action.'
    ], 500)
})

feedback_data.to_csv('feedback_data.csv', index=False)
print("Sample feedback data generated and saved as feedback_data.csv")
