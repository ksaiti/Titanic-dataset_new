import pandas as pd

#load data
df = pd.read_csv('tested.csv')
print(df.head())

# handle missing data
df.info()
df.isnull().sum()
df.fillna({'Age': df['Age'].median()}, inplace=True)

#print(df.columns)
# Visualize data
import matplotlib.pyplot as plt
plt.hist(df['Age'].dropna(), bins=20, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

# train a decision tree model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#df = pd.get_dummies(df, drop_first=True)
print(df.columns)

df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

X = df.drop('Survived', axis=1)

# Exercise in order to understand the metrics of Accuracy and Confusion Matrix
# we will discuss the steps in class
#X = df[['Age', 'Survived']]
#X = df[['Age', 'PassengerId']]
#X = df[['Age']]
y = df['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, predictions)}')

import matplotlib.pyplot as plt
import numpy as np

feature_importances = model.feature_importances_
features = X.columns
indices = np.argsort(feature_importances)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), feature_importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



