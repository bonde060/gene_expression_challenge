import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from testing import test
import pickle

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - abs(mape)
    print('Model Performance')
    print(f'Average Error: {np.mean(np.mean(errors))}')
    print(f'Accuracy = {np.mean(accuracy)}')

    return accuracy

#################### Read and format data ########################
data = pd.read_table("Challenge_GIN_release_profile_17804library_181query.txt", delimiter="\t", header=0, index_col=0)
# take rows that are present in the columns
X = data.loc[data.columns]
print(X)
# transpose is the output
Y = data.T
print(Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train, X_test, y_train, y_test)

#make model with params, comment out if already saved
base_model = RandomForestRegressor(n_estimators= 1000, min_samples_split= 10, min_samples_leaf= 4, max_features= 0.1, max_depth= 70, bootstrap= True, n_jobs = -2)
base_model.fit(X_train, y_train)

#save model, comment out if already saved
model_file = "../rf_model.pickle"
pickle.dump(base_model, open(model_file, "wb"))

#load model
base_model = pickle.load(open(model_file, 'rb'))

#test on training set
base_accuracy = evaluate(base_model, X_train, y_train)

#test on validation set
test(base_model, X_test, y_test)

#make predictions for 40 withheld genes
cols = pd.read_table("./ChallengeB/ChallengeB_release_hold_out_40_query_genes.txt")
cols = cols['hold_out_query_genes']
print(cols)
X_withheld = data.loc[cols]
preds = base_model.predict(X_withheld)

#save predictions
predictions = pd.DataFrame(preds, columns = data.index, index = cols).T.to_csv("randomforest_preds.csv")
print(f"Predictions: {preds}")

'''
#################### Random grid search to find decent parameters #################
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = [0.1, 0.2, 0.3]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X, Y)
print(rf_random.best_params_)

base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X, Y)
base_accuracy = evaluate(base_model, X, Y)
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X, Y)
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


################# Grid search with cross val #############
# Change this parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)
'''






