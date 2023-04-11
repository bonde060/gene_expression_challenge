import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error

data = pd.read_table("Challenge_GIN_release_profile_17804library_181query.txt", delimiter="\t", header=0, index_col=0)

model = RandomForestRegressor(max_depth=5, max_features=0.2, bootstrap=True, random_state=0, n_jobs=-2)

# take rows that are present in the columns
X = data.loc[data.columns]
print(X)

# transpose is the output
Y = data.T
print(Y.shape)

#fit model
model.fit(X, Y)


print(f"Original Y interaction for gene {data.columns[0]}:\n{data.loc[:, data.columns[0]]}")
#result = pd.DataFrame(model.predict([X.iloc[0]]))
#print(f"Predicted values: \n{result}")
#print(result.shape)

#evaluate
y_pred = model.predict(X.iloc[:10, :])
mse = mean_squared_error(Y.iloc[:10, :], y_pred)
print("Mean Squared Error: ", mse)





