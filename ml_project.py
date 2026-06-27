import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest,mutual_info_regression
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import os 

#Category_encoders is needed for Target Encoding text 

try:
    from category_encoders import TargetEncoder
except ImportError:
    print("Warning: category_encoders not installed. Target Encoding will not be used")

def main():
    print("Loading Datasets...")
    file_path = os.path.join(os.path.dirname(__file__), "train.csv")
 
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return

    df=pd.read_csv(file_path)
    print(f"Dataset Loaded Successfully.\nRows:{df.shape[0]}\nFeatures:{df.shape[1]}")

    #Handling missing values 
    print("\n--- Handling Missing Values ---")
    print("Artificially deleting some 'Age' data to demonstrate.")

    # Artificially create missing data on an existing column 'Age'
    df.loc[0:5, 'Age'] = np.nan
 
    imputer=SimpleImputer(strategy='median')

    df['Age'] = imputer.fit_transform(df[['Age']])
    print(f"Imputation complete. 'Age' now has {df['Age'].isnull().sum()} missing values.")

    #SKEWED Distribution
    print("\n--- Evaluating Skewed Distribution ---")
    print("Evaluating the skewness of the Spending distribution...")

    #Apply np.log1p (logarithmic + 1) to compress the distribution of Spending
    df['LogSpending'] = np.log1p(df['Spending'])
    print(f"Log Transformation applied. New skewness: {df['LogSpending'].skew():.2f} (closer to 0 is more balanced).\n")

    
if __name__=='__main__':
    main()
