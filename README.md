# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy and confusion matrix.
5. Display the results.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: M.Dhanush.
RegisterNumber:  25009955
*/
```
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
x= data.data[:,:3]
y=np.column_stack((data.target,data.data[:,6]))
x_train, x_test, y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state =42)
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)

Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])

## Output:

![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)

<img width="755" height="130" alt="Screenshot 2025-11-27 083519" src="https://github.com/user-attachments/assets/4024376e-6ea8-4a0d-b800-021f514b3686" />

<img width="284" height="149" alt="Screenshot 2025-11-27 083527" src="https://github.com/user-attachments/assets/2f651c05-b019-427d-90dd-e51b1b766f9c" />


<img width="344" height="133" alt="Screenshot 2025-11-27 083544" src="https://github.com/user-attachments/assets/5c2cca22-84a4-4e29-961f-abd3c4b52601" />


<img width="373" height="178" alt="Screenshot 2025-11-27 083558" src="https://github.com/user-attachments/assets/11f8ad0d-1c3d-4e25-b13a-35d2e8403260" />




## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
