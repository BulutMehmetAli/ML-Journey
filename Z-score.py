import pandas as pd
from sklearn.preprocessing import StandardScaler

# Sample Data
data = {'Age': [25, 30, 35, 40, 45, 50, 55, 60],
        'Income': [50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000]}
df = pd.DataFrame(data)

print("Original Data:")
print(df)

# 1. Initialize the Scaler
scaler = StandardScaler()

# 2. Fit the scaler on the data (calculates mean and std dev)
#    In a real scenario, fit ONLY on training data
scaler.fit(df)

# 3. Transform the data (applies the scaling)
scaled_data = scaler.transform(df)

# Convert back to DataFrame for better readability
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

print("\nScaled Data (Standardization):")
print(scaled_df)

# You can inspect the learned parameters
print(f"\nLearned Mean: {scaler.mean_}")
print(f"Learned Scale (Std Dev): {scaler.scale_}") # scale_ is the standard deviation

# Notice how the scaled features are now centered around zero. 
# The exact values reflect their original position relative to the mean, measured in standard deviations.