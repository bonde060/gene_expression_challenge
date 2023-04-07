import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

data = pd.read_table("Challenge_GIN_release_profile_17804library_181query.txt", delimiter="\t", header=0, index_col=0)
#data = np.loadtxt("Challenge_GIN_release_profile_17804library_181query.txt", delimiter="\t", dtype="str")
print(data)

model = RandomForestClassifier()

X = data.iloc[0:data.shape[1]]
print(X.shape)
Y = data.T
print(Y.shape)

#evaluate
crossVal = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
nScores = cross_val_score(model, X, y=Y, scoring='accuracy', cv=crossVal, n_jobs=1, error_score='raise')