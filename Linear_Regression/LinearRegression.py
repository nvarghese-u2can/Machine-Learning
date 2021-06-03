import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


housing_data = pd.read_csv('USA_Housing.csv')
print(housing_data.head())
print('\n')
print(housing_data.info())
print('\n')
print(housing_data.columns)

# Exploratory Data Analysis

sns.set_style('darkgrid')

sns.distplot(housing_data['Price'])
plt.show()

print(housing_data.corr())

sns.heatmap(housing_data.corr(), annot=True, linewidths=3, linecolor='blue')
plt.show()

sns.scatterplot(x='Avg. Area Income', y='Price', data= housing_data)
plt.show()

sns.heatmap(housing_data.isnull())
plt.show()

X = housing_data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms','Area Population']]
print(X)
y = housing_data['Price']

print(y)

# Train Test Data Split

X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.40, random_state=101)

# Create the Estimator object

lm = LinearRegression()

lm.fit(X_Train, y_Train)

print('\n Intercept value is {val}'.format(val=lm.intercept_))
print(lm.coef_)

coef = pd.DataFrame(lm.intercept_, index=X.columns, columns=['Coefficents'])
print(coef)

print('\n')
pred = lm.predict(X_Test)
print(pred)
sns.distplot(pred-y_Test)
plt.show()
sns.scatterplot(x=y_Test, y=pred)
plt.show()
print('\n Regression Statistics')
print(metrics.mean_absolute_error(y_Test, pred))
print(metrics.mean_squared_error(y_Test, pred))
print(np.sqrt(metrics.mean_squared_error(y_Test, pred)))

