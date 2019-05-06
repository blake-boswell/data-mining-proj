import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


dfTrain = pd.read_csv('./output/modified_train.csv')
dfTest = pd.read_csv('./output/modified_test.csv')

dfTrain.set_index('id', inplace=True)
dfTest.set_index('id', inplace=True)

# X_train,Y_train = dfTrain.loc[:, dfTrain.columns != 'country_destination'].values, dfTrain.loc[:, 'country_destination'].values
# X_test, Y_test = dfTest.loc[:, dfTest.columns != 'country_destination'].values, dfTest.loc[:, 'country_destination'].values

# labels = dfTrain['country_destination']
# le = LabelEncoder()
# y = le.fit_transform(labels)
y_train = dfTrain['country_destination']
X_train = dfTrain.drop('country_destination', axis=1, inplace=False)

randomForest = RandomForestClassifier(n_estimators=100)
randomForest.fit(X_train, y_train)
 

testIds = dfTest.index.values

# labels = dfTest['country_destination']
# Y_test = le.fit_transform(labels)
y_test = dfTest['country_destination']
X_test = dfTest.fillna(-1)
X_test = X_test.loc[:, dfTest.columns != 'country_destination'].values


predictions = randomForest.predict(X_test)
print('Shape of predictions: {}'.format(predictions.shape))
print('Shape of y_test: {}'.format(y_test.shape))

print('Predictions: {}'.format(predictions))
print(set(predictions))
sub = pd.DataFrame(np.column_stack((testIds, predictions)), columns=['id', 'country'])
sub.to_csv('./submission.csv',index=False)


X = dfTrain.loc[:, dfTrain.columns != 'country_destination'].values
y = dfTrain['country_destination']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print('Classification Report: ')
print(classification_report(y_test, y_pred))

print('Accuracy score:')
print(accuracy_score(y_test, y_pred))