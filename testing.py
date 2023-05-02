import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def make_preds(model, X_test):
    preds  = model.predict(X_test)
    return preds

def get_averageAUROC_binary_GI(df_qGI_label, df_qGI_score, group='random'):
    scores = np.zeros(df_qGI_score.shape[1])
    for ix, query in enumerate(df_qGI_score.columns):
        # print(query)
        non_zeros_genes = df_qGI_label[query][df_qGI_label[query] != 0].index
        # y_pred = df_qGI_score_order.loc[non_zeros_genes, query] # Predicted labels
        # y_true = df_qGI_label.loc[non_zeros_genes, query] # True labels
        auroc_query = roc_auc_score(df_qGI_label.loc[non_zeros_genes, query], df_qGI_score.loc[non_zeros_genes, query])
        scores[ix] = auroc_query
        # print(auroc_query)
    mean_auroc = np.mean(scores)
    # print("Average AUROC:", mean_auroc)
    plt.hist(scores, bins=len(scores))
    title_string = 'Distribution of AUROC (mean:' + str(round(mean_auroc, 4)) + ')   ' + 'Group: ' + group
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
    title_string = 'Distribution of ' + method + ' correlation (mean:' + str(
        round(mean_corr, 4)) + ')   ' + 'Group: ' + group
    plt.title(title_string)
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.savefig('correlation_distribution.pdf', dpi=300)
    plt.show()
    return score_list


def multiple_evaluation_Challenge_B(df_true_continuous, df_true_binary, df_submission, correlation_method='pearson',
                                    group='random'):
    # Keep the same order of index and columns as the submission file
    df_true_continuous_submission = df_true_continuous.loc[df_submission.index, df_submission.columns]
    df_true_binary_submission = df_true_binary.loc[df_submission.index, df_submission.columns]

    average_pcc = get_average_correlation(df_true_continuous_submission, df_submission, method='pearson', group=group)
    average_auroc = get_averageAUROC_binary_GI(df_true_binary_submission, df_submission, group=group)
    return [average_pcc, average_auroc]

def test(model, X_test, y_test):
    preds = make_preds(model, X_test)
    print(preds.shape, X_test.columns)
    preds = pd.DataFrame(preds, index = X_test.index, columns = y_test.columns).T
    print(preds)
    df_GIN_release = pd.read_csv('Challenge_GIN_release_profile_17804library_181query.txt', index_col=0, sep='\t')
    df_GIN_binary = pd.read_csv('Challenge_GIN_release_profile_17804library_181query_interaction_class.txt',
                                index_col=0, sep='\t')
    group_name = "Tanner/Annie"
    [average_pcc, average_auroc] = multiple_evaluation_Challenge_B(df_true_continuous=df_GIN_release,
                                                                   df_true_binary=df_GIN_binary,
                                                                   df_submission=preds, group=group_name)
    return average_pcc, average_auroc
