# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Prepare Data: Read the dataset and extract X (input) and y (output).
2. Split Data: Split the data into training and testing sets using train_test_split.
3. Train Model: Instantiate LinearRegression, then fit it to the training data.
4. Predict: Use the model to predict outputs for the test data.
5. Visualize: Plot the training and testing data with the regression line for visualization.
6. Evaluate: Compute and display MSE, MAE, and RMSE to assess model performance. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Pradeep E
RegisterNumber:  212223230149
*/
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="green")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:
![Screenshot 2024-09-01 170207](https://github.com/user-attachments/assets/80fa8e88-a0a1-4f69-a0a5-a1174cc6768e)
![Screenshot 2024-09-01 170216](https://github.com/user-attachments/assets/f045509d-c01c-48a7-9b7e-8f69b52b1858)
![Screenshot 2024-09-01 170241](https://github.com/user-attachments/assets/31b99149-6b02-4f14-a762-37289056a9ec)
![Screenshot 2024-09-01 170250](https://github.com/user-attachments/assets/b6fae17b-ea10-44e9-bcb4-14991f7042b3)
![Screenshot 2024-09-01 170223](https://github.com/user-attachments/assets/e8e42ef6-9887-4d16-80f5-e63735a20728)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
