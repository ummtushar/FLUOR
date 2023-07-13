import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

# Loading the Dataset and Removing Outliers
df = pd.read_csv("fluor_data.csv")
df['z_score'] = stats.zscore(df['Volume'])
df = df.loc[df['z_score'].abs()<=3]

# Plotting the Data of volume of stocks traded vs the Open Price
df.plot(kind='scatter', x = 'Open', y = 'Volume')
plt.show()


# Training and Testing Data Split - Supervised Training
X_train, X_test, y_train, y_test = train_test_split(df['Open'], df['Volume'])

# Test train split visualisation
plt.scatter(X_train, y_train, label = 'Training Data', color = 'r', alpha = .7)
plt.scatter(X_test, y_test, label = 'Testing Data', color = 'g', alpha = .7)
plt.legend()
plt.title("Test Train Split")
plt.show()

# Creating Linear Model and Train it to FIT
LR = LinearRegression()
LR.fit(X_train.values.reshape(-1,1), y_train.values)

# Use Model to PREDICT Y on Test Data (X)
prediction = LR.predict(X_test.values.reshape(-1,1))

# Plot prediction Line against actual test data
plt.plot(X_test, prediction, label = 'Linear Regression', color = 'b', alpha = .7)
plt.scatter(X_test, y_test, label = 'Actual Test Data', color = 'g', alpha = .7)
plt.legend()
plt.show()

# R2 Score
print(LR.score(X_test.values.reshape(-1,1), y_test.values))

# ----------------------------------------------------------------------
#Multiple Variable Regression
X = df[['Open', 'High', 'Low']]
y = df['Volume']

# Training and Testing Data Split - Supervised Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

# Creating Linear Model and Train it to FIT
LR = LinearRegression()
LR.fit(X_train, y_train)

# Use Model to PREDICT Y on Test Data (X)
# model = sm.OLS(y_test, X_test).fit()
predictions = LR.predict(X_test)

# Plot prediction Line against actual test data
plt.plot(X_test, predictions, label = 'Multiple Regression', color = 'b', alpha = .7)
plt.show()

# R2 score
print(r2_score(y_test, predictions))
