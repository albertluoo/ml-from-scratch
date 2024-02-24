import sys
import pandas as pd
sys.path.append('/Users/albertluo/version-control/kaggle/kaggle-tools')
from classifiers.rf_cls import RandomForestClassifier
from kaggle_eda_tool import identification, drop_correlation, sl, __eda__

# load data
train_df = pd.read_csv('/Users/albertluo/version-control/kaggle/kaggle-tools/titanic/titanic-train.csv')
test_df = pd.read_csv('/Users/albertluo/version-control/kaggle/kaggle-tools/titanic/titanic-test.csv')

# explore the data and make neccessary changes
__eda__(train_df, 0.95, 4)

# preprocessing
train_df.drop(['Name'], inplace = True, axis = 1)
train_df['Sex'] = train_df['Sex'].map({'female' : 0, 'male' : 1})
train_df['Embarked'] = train_df['Embarked'].map({'C' : 0, 'S' : 1})

encoded_train_df = pd.get_dummies(train_df, columns = ['Sex', 'Embarked', 'Cabin', 'Ticket'])
encoded_train_df = encoded_train_df.astype(float)

X_train = encoded_train_df.drop(['Survived'], axis = 1)
y_train = encoded_train_df['Survived'].values

test_df.drop(['Name'], inplace = True, axis = 1)
encoded_test_df = pd.get_dummies(test_df, columns = ['Sex', 'Embarked', 'Cabin', 'Ticket'])
encoded_test_df = encoded_test_df.astype(float)

# training on RF
rf = RandomForestClassifier(100, 9, 10, 2)
rf.random_forest_classifier(X_train, y_train)

# testing
preds = rf.predict_rf(encoded_test_df)
