# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_absolute_percentage_error,r2_score as r2_score_func, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score as cvs
import seaborn as sns

#data loading
data = pd.read_csv('car_data.csv')

data.info()

data.Make.value_counts()

data.value_counts()

data.describe()

data.corr(numeric_only=True)

data.drop(columns='Price Category', inplace=True)

data.info()

data.isnull().sum()

data.dtypes

object_data = data.dtypes == 'object'
object_data

import seaborn as sns
import matplotlib.pyplot as plt

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

sns.lineplot(x='Year of manufacture', y='Price', data=data[:40])
plt.title('Year of manufacture vs Price')
plt.xlabel('Year of manufacture')
plt.ylabel('Price')
plt.show()

data.describe()

#dealing with the outliers in the engine size
sns.histplot(data['Engine Size'], kde=True)
plt.title('Engine size distribution')
plt.show()

sns.boxplot(x=data['Engine Size'], y=data['Price'])
plt.title('Engine size distribution')
plt.figure(figsize=(10,10))
plt.show()

#detecting and eliminating the outliers
Q1 = data['Engine Size'].quantile(0.25)
Q3 = data['Engine Size'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['Engine Size'] >= lower_bound) & (data['Engine Size'] <= upper_bound)]

#dealing with the outliers in the  price data
q1 = data['Price'].quantile(0.25)
q3 = data['Price'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
data = data[(data['Price'] >= lower_bound) & (data['Price'] <= upper_bound)]

data.shape

data

data.info()

data.isnull().sum()

data['Condition']

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
data['Condition'] = imputer.fit_transform(data[['Condition']]).flatten()
data['Mileage'] = data['Mileage'].fillna(data['Mileage'].mean())
data['Fuel'] = imputer.fit_transform(data[['Fuel']]).flatten()
data['Transmission'] = imputer.fit_transform(data[['Transmission']]).flatten()

current_year = 2026
data['Car_age'] = current_year - data['Year of manufacture']
data.info()

data.drop_duplicates()

data.info()

data.columns

data.drop(columns='Year of manufacture', inplace=True)

data.corr(numeric_only=True)

sns.boxplot(x=data['Price'], y=data['Engine Size'])
plt.title('Engine size distribution')
plt.figure(figsize=(10,10))
plt.show()



data.info()

make_counts = data['Make'].value_counts()
makes_to_replace = make_counts[make_counts < 7].index
data['Make'] = data['Make'].replace(makes_to_replace, 'Other')

data.Make.value_counts()

data

data.info()

x = data.drop(columns='Price')
y = data['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42,test_size=0.2)

x_train

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(drop='first', sparse_output=False)
cat_columns_list = ['Make', 'Condition', 'Fuel', 'Transmission']
x_train_cat = cat_encoder.fit_transform(x_train[cat_columns_list])
x_test_cat = cat_encoder.transform(x_test[cat_columns_list])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_num = scaler.fit_transform(x_train.drop(columns=cat_columns_list))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_num = scaler.fit_transform(x_train.drop(columns=cat_columns_list))
x_test_num = scaler.transform(x_test.drop(columns=cat_columns_list))

x_train_processed = np.hstack((x_train_num, x_train_cat))
x_test_processed = np.hstack((x_test_num, x_test_cat))

"""Now, `x_train_processed` and `x_test_processed` are NumPy arrays containing all your features, with numerical columns scaled and categorical columns one-hot encoded. These are ready for model training."""

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20],
    'min_samples_split': [5,10],
    'min_samples_leaf': [1, 2, 5]
}
random = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=2,n_jobs=-1)

import numpy as np
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

random = random.fit(x_train_processed, y_train_log)

estimator = random.best_estimator_

pred_log = estimator.predict(x_test_processed)

normal_pred = np.expm1(pred_log)

mse_score = mean_squared_error(y_test, normal_pred)
mae_score = mean_absolute_error(y_test, normal_pred)
mape_score = mean_absolute_percentage_error(y_test, normal_pred)
r2_score_value = r2_score_func(y_test, normal_pred)

print(f"The mse is {mse_score}")
print(f"The mae is {mae_score}")
print(f"The mape is {mape_score}")
print(f'The r2 score is {r2_score_value}')

"""TRYING LIGHT GBM
**bold text**
"""

import lightgbm as light

light_model = light.LGBMRegressor(random_state=42)

param_grid_light = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20],
    'learning_rate': [0.01, 0.1],
    'min_child_samples': [20, 30, 50]
}

light_grid = GridSearchCV(estimator=light_model, param_grid=param_grid_light, cv=4, verbose=2, n_jobs=-1, scoring='neg_mean_squared_error')

light_grid.fit(x_train_processed, y_train_log)

light_best_estimator = light_grid.best_estimator_

light_pred_log = light_best_estimator.predict(x_test_processed)
normal_light_pred = np.expm1(light_pred_log)

light_mse_score = mean_squared_error(y_test, normal_light_pred)
light_mae_score = mean_absolute_error(y_test, normal_light_pred)
light_mape_score = mean_absolute_percentage_error(y_test, normal_light_pred)
light_r2_score = r2_score_func(y_test, normal_light_pred)

print(f"LightGBM MSE: {light_mse_score}")
print(f"LightGBM MAE: {light_mae_score}")
print(f"LightGBM MAPE: {light_mape_score}")
print(f"LightGBM R2 Score: {light_r2_score}")

#saving the necessary model and the preprocessor for further usage
import joblib

joblib.dump(estimator, 'light_best_estimator.joblib')
joblib.dump(cat_encoder, 'cat_encoder.joblib')
joblib.dump(scaler, 'scaler.joblib')




plt.tight_layout()
plt.show()
