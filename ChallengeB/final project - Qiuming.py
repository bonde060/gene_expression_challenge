import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the evaluation functions
def get_averageAUROC_binary_GI(df_qGI_label, df_qGI_score, group='random'):
    scores = np.zeros(df_qGI_score.shape[1])
    for ix, query in enumerate(df_qGI_score.columns):
        #print(query)
        non_zeros_genes = df_qGI_label[query][df_qGI_label[query]!=0].index
        #y_pred = df_qGI_score_order.loc[non_zeros_genes, query] # Predicted labels
        #y_true = df_qGI_label.loc[non_zeros_genes, query] # True labels
        auroc_query = roc_auc_score(df_qGI_label.loc[non_zeros_genes, query], df_qGI_score.loc[non_zeros_genes, query])
        scores[ix] = auroc_query
        #print(auroc_query)
    mean_auroc = np.mean(scores)
    #print("Average AUROC:", mean_auroc)
    plt.hist(scores, bins=len(scores))
    title_string ='Distribution of AUROC (mean:'+str(round(mean_auroc,4))+')   ' + 'Group: '+ group
    plt.title(title_string)
    plt.xlabel('AUROC')
    plt.ylabel('Frequency')
    plt.savefig('auroc_distribution.pdf', dpi=300)
    plt.show()
    return scores


def get_average_correlation(df_true, df_score, method='pearson', group='random'):
    score_list = []
    for col in df_score.columns:
        y_pred = df_score[col]
        y_true = df_true[col]
        #score_list.append(pearsonr(y_true, y_pred)[0])
        score_list.append(y_true.corr(y_pred, method=method))
    mean_corr = np.mean(score_list)
    #print("Average correlation ("+method+"):", mean_corr)
    plt.hist(score_list, bins=len(score_list))
    title_string ='Distribution of '+ method + ' correlation (mean:'+str(round(mean_corr,4))+')   ' + 'Group: '+ group
    plt.title(title_string)
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.savefig('correlation_distribution.pdf', dpi=300)
    plt.show()
    return score_list


def multiple_evaluation_Challenge_B(df_true_continuous, df_true_binary, df_submission, correlation_method='pearson', group='random'):
    # Keep the same order of index and columns as the submission file
    df_true_continuous_submission = df_true_continuous.loc[df_submission.index, df_submission.columns]
    df_true_binary_submission = df_true_binary.loc[df_submission.index, df_submission.columns]
    
    average_pcc = get_average_correlation(df_true_continuous_submission, df_submission, method='pearson', group=group)
    average_auroc = get_averageAUROC_binary_GI(df_true_binary_submission, df_submission, group=group)
    return [average_pcc, average_auroc]

# Load the dataset
data = pd.read_csv('Challenge_GIN_release_profile_17804library_181query.txt', delimiter='\t', header=0, index_col=0, low_memory=False)
interaction_class = pd.read_csv('Challenge_GIN_release_profile_17804library_181query_interaction_class.txt', delimiter='\t', header=0, index_col=0, low_memory=False)

# Preprocess data
X = data.values.T
y = interaction_class.values.T

# Normalize input data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define neural network architecture
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=17804))
model.add(Dense(256, activation='relu'))
model.add(Dense(17804, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Save the model predictions
y_pred = model.predict(X_val)
df_submission = pd.DataFrame(y_pred, columns=data.index[2:], index=interaction_class.columns[2:])

# Save the predictions to a file
df_submission.to_csv('model_predictions.csv')

# Evaluate the model
work_dir = 'C:/Users/User.DESKTOP-PC970HM/Desktop/2nd semester/CSCI5461/Final project'  # Update this path to your working directory
os.chdir(work_dir)
group_name = 'Qiuming'  # Update this with your group name

df_GIN_release = pd.read_csv('Challenge_GIN_release_profile_17804library_181query.txt', index_col=0, sep='\t')
df_GIN_binary = pd.read_csv('Challenge_GIN_release_profile_17804library_181query_interaction_class.txt', index_col=0, sep='\t')

# Make predictions on the test set
y_pred = model.predict(X_val)

# Calculate correlation and AUROC values
corr_values = []
auroc_values = []
for i in range(y_pred.shape[1]):
    corr = np.corrcoef(y_val[:, i], y_pred[:, i])[0, 1]
    corr_values.append(corr)
    auroc = roc_auc_score(y_val[:, i], y_pred[:, i])
    auroc_values.append(auroc)

# Evaluate the model using the provided evaluation functions
[average_pcc, average_auroc] = multiple_evaluation_Challenge_B(df_true_continuous=df_GIN_release,
                                                                df_true_binary=df_GIN_binary,
                                                                df_submission=df_submission, group=group_name)


