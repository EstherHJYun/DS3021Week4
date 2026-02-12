# %% [markdown]
## Part 1 : Question

# Assuming that higher aid value, results in higher graduation rates, can we predict which institutions will have higher graudatation rates? 

# %%
## Importing libraries

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
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

# %%
## Data Cleaning 

grad_data = pd.read_csv('https://query.data.world/s/qpi2ltkz23yp2fcaz4jmlrskjx5qnp', encoding="cp1252")
grad_data.info()

# Dropping columns 
to_drop = list(range(39, 56))
to_drop.extend([27, 9, 10, 11, 28, 36, 60, 56])

grad_data1 = grad_data.drop(grad_data.columns[to_drop], axis=1)
grad_data1.info()

drop_more = [0,2,3,6,8,11,12,14,15,18,21,23,29,32,33,34,35]
grad_data2 = grad_data1.drop(grad_data1.columns[drop_more], axis=1)
grad_data2.info()

# Filling in missing values for the hbcu column
grad_data2.replace('NULL', np.nan, inplace=True)
grad_data2['hbcu'] = [1 if grad_data2['hbcu'][i]=='X' else 0 for i in range(len(grad_data2['hbcu']))]
grad_data2['hbcu'].value_counts()

# Changing data types
grad_data2['hbcu'] = grad_data2.hbcu.astype('category')
grad_data2[['level', 'control']] = grad_data2[['level', 'control']].astype('category')

# More dropping
grad_data2 = grad_data2.drop(grad_data2[['med_sat_value']], axis=1)
grad_data2.dropna(axis = 0, how = 'any', inplace = True)

# Checking for missing values
sns.displot(
    data=grad_data2.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25
)

# %% 
## Normalizing the data 

# Selecting numeric columns
numeric_cols = grad_data2.select_dtypes(include=['int64', 'float64']).columns

# Normalizing the numeric columns using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
d = scaler.fit_transform(grad_data2[numeric_cols])   # conduct data transformation
scaled_df = pd.DataFrame(d, columns=numeric_cols)   # convert back to pd df; transformation 
grad_data2[numeric_cols] = scaled_df   # put data back into the main df
grad_data2.describe()

# %%
## One Hot Encoding the data

# Selecting categorical columns
cat_cols = grad_data2.select_dtypes(include='category').columns

encoded = pd.get_dummies(grad_data2[cat_cols])
encoded.head()

# Dropping the original categorical columns and joining the encoded columns back to the main df
grad_data2 = grad_data2.drop(cat_cols, axis=1)
grad_data2 = grad_data2.join(encoded)
grad_data2.info()

# %%
## Creating the target variable 

# Visualizing the distribution of the aid_value variable to determine a threshold for categorization
print(grad_data2.boxplot(column='aid_value', vert=False, grid=False))
print(grad_data2['aid_value'].describe())

# 1 = High graduation rate (aid > 0.254348), 0 = Low graduation rate (aid <= 0.254348)
grad_data2['aid_value_f'] = pd.cut(grad_data2.aid_value,
                                    bins=[-1, 0.254348, 1],
                                    labels=[0, 1])

# Dropping more rows and columns
grad_data2 = grad_data2.dropna(axis=0)
grad_data2 = grad_data2.drop(['chronname'], axis=1) # drop chronname column
grad_data2.info()

# %%
## Calculating prevalence 

grad_data2['aid_value_f'].value_counts()[1] / grad_data2['aid_value_f'].count()

# %% [markdown]
## Part 2 : Building a kNN model 

# %%
## Splitting the data 

train, test = train_test_split(grad_data2,  test_size=0.4, stratify = grad_data2['aid_value_f']) 
test, val = train_test_split(test, test_size=0.5, stratify=test['aid_value_f'])

# %%
import random
random.seed(1984)

# Setting up the training data for the model
X_train = train.drop(['aid_value_f'], axis=1).values
y_train = train['aid_value_f'].values

# Creates and fits the KNN model with k=3
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# %%
# Checking the accuracy of the model 

# Setting up the testing data for the model
X_test = test.drop(['aid_value_f'], axis=1).values
y_test = test['aid_value_f'].values

# Returns the mean accuracy on the given test data and labels
neigh.score(X_test, y_test)

# %%
# Checking the accuracy of the model on the validation set

# Setting up the validation data for the model
X_val = val.drop(['aid_value_f'], axis=1).values
y_val = val['aid_value_f'].values

# Returns the mean accuracy on the given validation data and labels
neigh.score(X_val, y_val)

# %%
# Creating a confusion matrix 

# Generate predictions for the validation set
y_val_pred = neigh.predict(X_val)

# Display the confusion matrix
cm = confusion_matrix(y_val,y_val_pred, labels=neigh.classes_)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=neigh.classes_)  
disp.plot()
plt.show()

# %%
# Creating a classification report

y_val_pred = neigh.predict(X_val)
print(classification_report(y_val_pred, y_val))

# %%
# Calculating sensitivity and specificity

sensitivity = 131/(131+24)   # = TP/(TP+FN)
specificity = 435/(435+27)   # = TN/(TN+FP)
print(sensitivity, specificity)

# %%
# Calculating permutation importance for the neigh model

perm_importance = permutation_importance(neigh, X_test, y_test, n_repeats=10)
importances = perm_importance.importances_mean

# %%
# Selecting the best k 

def chooseK(k, X_train, y_train, X_test, y_test):
    random.seed(1)
    print("calculating... ", k, "k")    
    class_knn = KNeighborsClassifier(n_neighbors=k)
    class_knn.fit(X_train, y_train)
    
    # calculate accuracy
    accu = class_knn.score(X_test, y_test)
    return accu

# Dataframe to store the accuracy for different k values
test = pd.DataFrame({'k':list(range(1,22,2)), 
                     'accu':[chooseK(x, X_train, y_train, X_test, y_test) for x in list(range(1, 22, 2))]})

# %%

# Sort the dataframe by accuracy in descending order
test = test.sort_values(by=['accu'], ascending=False)
test

# %%
# Adjusting the threshold 

# Probabilities and predictions for the test set
test_probs = neigh.predict_proba(X_test)
test_preds = neigh.predict(X_test)

# Converting probabilities to pandas Dataframe
test_probabilities = pd.DataFrame(test_probs, columns = ['low_aid', 'high_aid'])
test_probabilities

# %% [markdown]
## Part 3 : Dataframe 
# The Dataframe includes the test target values, test predicted values, and test probabilities of the positive class

# %%
# Creating a Dataframe
final_model = pd.DataFrame({'actual_class': y_test.tolist(),
                           'pred_class': test_preds.tolist(),
                           'pred_prob': [test_probabilities['high_aid'][i] if test_preds[i]==1 else test_probabilities['low_aid'][i] for i in range(len(test_preds))]})

final_model.head()

# %%

final_model['pos_pred'] = [final_model.pred_prob[i] if final_model.pred_class[i]==1 else 1-final_model.pred_prob[i] for i in range(len(final_model.pred_class))]
final_model.head()

# %%
# Converting classes to categories
final_model.actual_class = final_model.actual_class.astype('category')
final_model.pred_class = final_model.pred_class.astype('category')

# %%
# Creating probability distribution graph

sns.displot(final_model, x="pos_pred", kind="kde")

# %%

final_model.pos_pred.value_counts()

# %%

# Function to adjust threshold and display confusion matrix
def adjust_thres(x, y, z):
    """
    x=pred_probabilities
    y=threshold
    z=tune_outcome
    """
    thres = pd.DataFrame({'new_preds': [1 if i > y else 0 for i in x]})
    thres.new_preds = thres.new_preds.astype('category')
    con_mat = confusion_matrix(z, thres)  
    print(con_mat)

# %%

confusion_matrix(final_model.actual_class, final_model.pred_class)   # original model

# %%

adjust_thres(final_model.pos_pred, .90, final_model.actual_class)   # raise threshold 

# %%

adjust_thres(final_model.pos_pred, .3, final_model.actual_class)   # lower threshold

# %% [markdown]
## Part 4 : No Code Question

# If you adjusted the k hyperparameter what do you think would
# happen to the threshold function? Would the confusion look the same at the same threshold 
# levels or not? Why or why not?

# %% [markdown]
# Adjusting the k hyperparameter changes the distribution of the probabilities, 
# which would change the number of cases that fall above or below a given threshold. 
# Therefore, the confusion matrix would not look the same at same threshold levels 
# because the predicted classes would change based on the new probabilities.

# %% [markdown]
## Part 5 : Evaluate the results using the confusion matrix

# Evaluate the results using the confusion matrix. Then "walk" through your question, summarize what 
# concerns or positive elements do you have about the model as it relates to your question? 

# %% [markdown]
# The model has a sensitivity of 0.845 and a specificity of 0.941, which indicates that 
# it is good at correctly identifying both high and low graduation rates based on aid value. 

# %% [markdown]
## Part 7 : Model Performance

# How well does the model perform? Did the interaction of the adjusted thresholds and k values help the model? Why or why not? 

# %% [markdown]
# The model performs well overall, with a high accuracy and good sensitivity and specificity. 
# Adjusting the thresholds and k values can help improve the model's performance by fine-tuning 
# the balance between sensitivity and specificity. For example, increasing k may reduce overfitting 
# but could also reduce sensitivity, while adjusting thresholds can help optimize the trade-off 
# between false positives and false negatives.

# %% [markdown]
## Part 6 : Functions 

# %%
# Cleans the data & splits into training|test

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
# Train and test the model with different k and threshold values

def train_and_test(k, threshold, train_features, train_labels, test_features, test_labels):
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
# Choose another variable as the target in the dataset and 
# create another kNN model using the two functions you created in step 7. 

# %%
# Importing the data

grad_data = pd.read_csv('https://query.data.world/s/qpi2ltkz23yp2fcaz4jmlrskjx5qnp', encoding="cp1252")
grad_data.info()

# %% 
# Running the clean_and_split data function

train, test, val = clean_and_split_data(grad_data, target_variable='hbcu', threshold=0.5)

# %%

# Setting up the training and testing data for the model
X_train = train.drop(columns=['hbcu'])
y_train = train['hbcu']

X_test = test.drop(columns=['hbcu'])
y_test = test['hbcu']

# %%
# Running the train_and_test function

acc, cm = train_and_test(
    k=3,
    threshold=0.5,
    train_features=X_train,
    train_labels=y_train,
    test_features=X_test,
    test_labels=y_test
)
