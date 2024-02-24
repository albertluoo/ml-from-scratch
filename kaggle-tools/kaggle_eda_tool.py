import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def identification(df):
  print(f'Head:\n {df.head()}')
  sl()
  print(f'Columns:\n {df.columns}')
  sl()
  print(f'Dimensions:\n {df.shape}')
  sl()
  print(f'Types:\n {df.dtypes}')
  sl()
  print(f'Number of Null Values: {df.isnull().sum().sum()}')
  sl()

def drop_correlation(df, threshold, max_features_removed):
    correlated_features = set()
    print(f'Number of Features Before: {len(df.columns)}')
    sl()
    feats_removed = 0
    
    # Filter out non-numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            correlation = df[numeric_columns[i]].corr(df[numeric_columns[j]])
            if abs(correlation) > threshold and feats_removed < max_features_removed:
                feats_removed += 1
                column_to_drop = numeric_columns[j]
                correlated_features.add(numeric_columns[j])
                print(f"Dropped column '{column_to_drop}' due to high correlation")

    df.drop(list(correlated_features), axis=1, inplace=True)
    sl()
    
    # Create a heatmap using the numeric columns
    heatmap = sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 5})
    
    print(f'Number of Features After: {len(df.columns)}')
    sl()
    print(f'\nUpdated Columns: {df.columns}')
    sl()

def sl():
  print('__'*40)

def __eda__(df, threshold, max_features_removed):
   df.dropna(inplace = True)
   identification(df)
   drop_correlation(df, threshold, max_features_removed)