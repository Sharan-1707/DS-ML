import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Understanding dataset")

# Define the absolute path of the dataset relative to this script
file_name = os.path.join(os.path.dirname(__file__), "sales_data.csv")
if not os.path.exists(file_name):
    print(f"File {file_name} not found")
    exit()

df = pd.read_csv(file_name)
print("Successfully loaded dataset")
print(f"Shape of the dataset: Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- Dataset Head ---")
print(df.head())
print("\n--- Dataset Tail ---")
print(df.tail())
print("\n--- Description ---")
print(df.describe())

print("\n--- Handling missing values ---")
print("Missing values per column:")
print(df.isnull().sum())

print("\nRows with missing 'Spending' values:")
print(df[df['Spending'].isnull()])

# Impute missing Age using median 
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(f"\nImputed missing Age with median: {median_age}")

# Impute missing Spending using mean 
mean_spending = df['Spending'].mean()
df['Spending'] = df['Spending'].fillna(mean_spending)
print(f"Imputed missing Spending with mean: {mean_spending}")

print("\nMissing values after imputation:")
print(df.isnull().sum())

# Plot the distribution of Spending
plt.figure(figsize=(7, 4))
df['Spending'].hist(bins=10, edgecolor='black', color='skyblue')
plt.title("Distribution of Spending")
plt.xlabel("Spending Amount")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()
