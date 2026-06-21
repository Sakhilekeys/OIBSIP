import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset safely
try:
    df = pd.read_csv('Task2_Unemployment_Analysis/Unemployment_in_India.csv')
    print("Dataset successfully loaded!")
except FileNotFoundError:
    try:
        df = pd.read_csv('Unemployment_in_India.csv')
        print("Dataset successfully loaded from root folder!")
    except FileNotFoundError:
        print("Error: Could not find the CSV file.")
        print("Please make sure your data file is named 'Unemployment_in_India.csv' and placed inside your Task2 folder.")
        exit()

# Clean up column spaces (Oasis datasets often have extra spaces in headers)
df.columns = df.columns.str.strip()

# 2. Basic Inspection & Data Cleansing
print("\n--- Dataset Overview ---")
print(df.head())
print("\n--- Missing Values Matrix ---")
print(df.isnull().sum())

# Drop rows that are completely empty
df.dropna(inplace=True)

# Parse Date values safely
df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%d-%m-%Y')

# 3. Exploratory Visualizations
plt.style.use('seaborn-v0_8-whitegrid')

# Chart 1: Timeline Trend Analysis
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Date', y='Estimated Unemployment Rate (%)', hue='Area', errorbar=None)
plt.title('Unemployment Trends Over Time (Urban vs Rural Areas)')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.xlabel('Timeline')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Chart 2: Regional Distribution Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Estimated Unemployment Rate (%)', y='Region', palette='Set2')
plt.title('Regional Distribution of Unemployment Rates')
plt.xlabel('Estimated Unemployment Rate (%)')
plt.ylabel('Region')
plt.tight_layout()
plt.show()