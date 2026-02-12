# %% [markdown]
## Part 6 : Functions 

# %% 
# Importing libraries 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


# %%
# cleans the data & splits into training|test

def clean_and_split_data(df, target_variable, threshold, test_size=0.4, val_size=0.5, random_state=1984):
   
    # Drop specified columns
    to_drop = list(range(39, 56))
    to_drop.extend([27, 9, 10, 11, 28, 36, 60, 56])
    df = df.drop(df.columns[to_drop], axis=1)

    drop_more = [0,2,3,6,8,11,12,14,15,18,21,23,29,32,33,34,35]
    df = df.drop(df.columns[drop_more], axis=1)

    # Clean values

    df.replace('NULL', np.nan, inplace=True)

    df['hbcu'] = np.where(df['hbcu'] == 'X', 1, 0)
    df['hbcu'] = df['hbcu'].astype('category')

    df[['level', 'control']] = df[['level', 'control']].astype('category')

    df = df.drop(['med_sat_value'], axis=1, errors='ignore')
    df.dropna(axis=0, how='any', inplace=True)

    # Normalize numeric data

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = preprocessing.MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Create target variable

    df[target_variable] = pd.cut(
        df[target_variable],
        bins=[-1, threshold, 1],
        labels=[0, 1]
    )

    # One-hot encode categories

    cat_cols = df.select_dtypes(include='category').columns

    # Remove target from categorical columns
    cat_cols = cat_cols.drop(target_variable)

    encoded = pd.get_dummies(df[cat_cols])
    df = df.drop(cat_cols, axis=1).join(encoded)

    df = df.dropna(axis=0)
    df = df.drop(['chronname'], axis=1, errors='ignore')

    # Split data

    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_variable],
        random_state=random_state
    )

    test, val = train_test_split(
        test,
        test_size=val_size,
        stratify=test[target_variable],
        random_state=random_state
    )

    return train, test, val

# %% 
# train and test the model with different k and threshold values

def evaluate_knn(k, threshold, train_features, train_labels, test_features, test_labels):
    """
    k: number of neighbors
    threshold: probability cutoff for positive class
    train_features: training predictors
    train_labels: training outcomes
    test_features: testing predictors
    test_labels: testing outcomes
    """

    random.seed(1)
    print("Calculating for k =", k)

    # Fit model
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_features, train_labels)

    # Predicted probabilities for positive class
    probabilities = model.predict_proba(test_features)[:, 1]

    # Apply threshold function
    predictions = (probabilities > threshold).astype(int)

    # Compute metrics
    cm = confusion_matrix(test_labels, predictions)
    acc = accuracy_score(test_labels, predictions)

    print("Threshold:", threshold)
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)

    return acc, cm

# %% [markdown]
## Part 8 : Another kNN 

grad_data = pd.read_csv('https://query.data.world/s/qpi2ltkz23yp2fcaz4jmlrskjx5qnp', encoding="cp1252")
grad_data.info()

# %% 

train, test, val = clean_and_split_data(grad_data, target_variable='hbcu', threshold=0.5)

# %%

X_train = train.drop(columns=['hbcu'])
y_train = train['hbcu']

X_test = test.drop(columns=['hbcu'])
y_test = test['hbcu']

# %%

acc, cm = evaluate_knn(
    k=3,
    threshold=0.5,
    train_features=X_train,
    train_labels=y_train,
    test_features=X_test,
    test_labels=y_test
)

# %%
