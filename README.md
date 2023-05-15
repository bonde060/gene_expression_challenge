# Gene Interaction Prediction
Predict interactions between genes in human genome. Given known interactions for 181 genes, predict the interaction profile for query gene. Output is ~17,000 interaction values for the query gene with all other genes. Max AUC of ~0.543. 

Models tested: Sklearn Random Forest Regressor, Keras Sequential Neural Net, Keras Autoencoder

Input matrix: ~17,000 x 181 known interaction values
Output: ~17,000 x 1 interaction predictions for query gene
