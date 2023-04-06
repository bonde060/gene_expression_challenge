import numpy as np
import pandas as pd

data = pd.read_table("Challenge_GIN_release_profile_17804library_181query.txt", delimiter="\t", header=0, index_col=0)
#data = np.loadtxt("Challenge_GIN_release_profile_17804library_181query.txt", delimiter="\t", dtype="str")
print(data)