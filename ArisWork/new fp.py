from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score


def load_and_preprocess_data(file_path):
    data = pd.read_table("Challenge_GIN_release_profile_17804library_181query.txt",
                         delimiter="\t", header=0, index_col=0)
    # take rows that are present in the columns
    # print(data.columns)
    X = data.loc[data.columns]
    print(X)
    # transpose is the output
    Y = data.T
    # print(Y.shape)
    # print(Y)
    return data, X, Y


def prepare_data(data, X):
    # Prepare the input and output data
    # X = data.iloc[:, 1:].values

    # Scale the input data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Normalize the input data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_scaled)
    X_scaled = X_scaled.astype('float32')

    # Check for NaN or infinite values
    assert not np.any(np.isnan(X_scaled))
    assert not np.any(np.isinf(X_scaled))

    return X_scaled


def create_model(input_shape):
    # Create the neural network model
    model = Sequential()
    model.add(Dense(128, input_dim=input_shape, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(17804))

    # Use a smaller learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def save_predicted_interactions(predictions, file_name="predicted_interactions.csv"):
    # Save the predicted interactions to an Excel file
    predicted_interactions_df = pd.DataFrame(predictions).T
    predicted_interactions_df.to_csv(file_name, index=True)


if __name__ == "__main__":
    file_path = "Challenge_GIN_release_profile_17804library_181query.txt"
    data, X, Y = load_and_preprocess_data(file_path)
    X_scaled = prepare_data(data, X)
    model = create_model(X_scaled.shape[1])
    
    print(type(X_scaled), type(Y))

    # Train the model
    print("Training the model...")
    history = model.fit(X_scaled, Y, epochs=150, batch_size=32, validation_split=0.1, verbose=1)

    # Make predictions for all genes
    predicted_interactions = model.predict(X_scaled[:2])
    print(predicted_interactions)
    print(X_scaled[:1])
    corr, pval = pearsonr(predicted_interactions[0], Y.iloc[0])

    print(corr)

    # Save the predicted interactions to an Excel file
    save_predicted_interactions(predicted_interactions, "all_predicted_interactions.csv")
    print("Predicted interactions for all genes saved to 'all_predicted_interactions.csv'")
    
# _________________________________________________________________________________________________

def get_averageAUROC_binary_GI(df_qGI_label, df_qGI_score, group='random'):
    scores = np.zeros(df_qGI_score.shape[1])
    for ix, query in enumerate(df_qGI_score.columns):
        # print(query)
        non_zeros_genes = df_qGI_label[query][df_qGI_label[query] != 0].index
        # y_pred = df_qGI_score_order.loc[non_zeros_genes, query] # Predicted labels
        # y_true = df_qGI_label.loc[non_zeros_genes, query] # True labels
        auroc_query = roc_auc_score(
            df_qGI_label.loc[non_zeros_genes, query], df_qGI_score.loc[non_zeros_genes, query])
        scores[ix] = auroc_query
        # print(auroc_query)
    mean_auroc = np.mean(scores)
    # print("Average AUROC:", mean_auroc)
    plt.hist(scores, bins=len(scores))
    title_string = 'Distribution of AUROC (mean:'+str(
        round(mean_auroc, 4))+')   ' + 'Group: ' + group
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
        # score_list.append(pearsonr(y_true, y_pred)[0])
        score_list.append(y_true.corr(y_pred, method=method))
    mean_corr = np.mean(score_list)
    # print("Average correlation ("+method+"):", mean_corr)
    plt.hist(score_list, bins=len(score_list))
    title_string = 'Distribution of ' + method + \
        ' correlation (mean:'+str(round(mean_corr, 4)) + \
        ')   ' + 'Group: ' + group
    plt.title(title_string)
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.savefig('correlation_distribution.pdf', dpi=300)
    plt.show()
    return score_list


def multiple_evaluation_Challenge_B(df_true_continuous, df_true_binary, df_submission, correlation_method='pearson', group='random'):
    # Keep the same order of index and columns as the submission file
    df_true_continuous_submission = df_true_continuous.loc[df_submission.index,
                                                           df_submission.columns]
    df_true_binary_submission = df_true_binary.loc[df_submission.index,
                                                   df_submission.columns]

    average_pcc = get_average_correlation(
        df_true_continuous_submission, df_submission, method='pearson', group=group)
    average_auroc = get_averageAUROC_binary_GI(
        df_true_binary_submission, df_submission, group=group)
    return [average_pcc, average_auroc]


if __name__ == '__main__':
    # Set working directory if you need. Otherwise it will read in the data from the current directory
    #work_dir = '/Users/zhangxiang/Documents/PhD/Spring2023/CSCI5461/PredictionChallenge_2023/ChallengeB/Evaluation_code_data/'
    #os.chdir(work_dir)

    # Please chaneg this variable to your group's name.
    # You can also input your name (e.g., Xiang Zhang) and it will be displayed in the figures.
    group_name = 'Xiang Zhang'

    # Please change the following file to your test data in csv format (separated by comma, with index and header))
    # You can include any number of columns in your test submission and the code will automatically evaluate them.
    # Just make sure these column (query gene) names are included by our released data instead of the held-out set
    submission_file_name = 'all_predicted_interactions.csv'
    df_submission = pd.read_csv(submission_file_name, index_col=0)

    # Provided data. Please do not change the following file names besides correcting your input directory to them.
    df_GIN_release = pd.read_csv(
        'Challenge_GIN_release_profile_17804library_181query.txt', index_col=0, sep='\t')
    df_GIN_binary = pd.read_csv(
        'Challenge_GIN_release_profile_17804library_181query_interaction_class.txt', index_col=0, sep='\t')

    # The following code will generate two evaluation figures for your reference
    # 1. Distribution of the Pearson correlation based on the continuous ground truth data
    # 2. Distribution of the AUROC values based on the ground truth binary labels (only for 1s and -1s, ignoring 0s)
    [average_pcc, average_auroc] = multiple_evaluation_Challenge_B(df_true_continuous=df_GIN_release,
                                                                   df_true_binary=df_GIN_binary,
                                                                   df_submission=df_submission, group=group_name)







