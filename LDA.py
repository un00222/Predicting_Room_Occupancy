import numpy as np;
from utils import getNormalizedDataSets, calcDisplayMetrics, print_confusion

source_file_name = "Occupancy_Estimation.csv"

# Calculate mean and covariance for each class
def calcMeanCov(X_trg, Y_trg):

    classes = np.unique(Y_trg)
    means = []
    covariances = []
    for cls in classes:
        X_cls = X_trg[Y_trg.flatten() == cls]
        means.append(np.mean(X_cls, axis=0))
        covariances.append(np.cov(X_cls, rowvar=False))
    return means, covariances, classes

# Find the predicted values from Z and mean of each class
def findPredictedValues(Z, mean_class, classes):
    dists = np.array([np.linalg.norm(Z - mean, axis=1) for mean in mean_class]).T
    preds = np.argmin(dists, axis=1)
    Y_pred = np.array(classes)[preds].reshape(-1, 1)
    return Y_pred
class lda:
    def __init__(self):
        self.W_lda = []
        self.mean_class = 0
        self.classes = []

    def fit(self, X_trg, Y_trg, X_valid, Y_valid):
        # Fit the LDA model

        # Calculate mean and covariance for each class
        means, covariances, self.classes = calcMeanCov(X_trg, Y_trg)
        # Within-class scatter matrix (Sw)
        Sw = np.zeros_like(covariances[0])
        for cov in covariances:
            Sw += cov

        # Overall mean of the training data set
        overall_mean = np.mean(X_trg, axis=0)

        # Between-class scatter matrix (Sb)
        Sb = np.zeros((X_trg.shape[1], X_trg.shape[1])) # shape will be a square matrix feature_size x feature_size
        for i, cls in enumerate(self.classes):
            n_cls_i = np.sum(Y_trg.flatten() == cls) # |C_i|
            mean_diff = (means[i] - overall_mean).reshape(-1, 1)
            Sb += n_cls_i * (mean_diff @ mean_diff.T)

        # Eigen decomposition of Sw^-1 Sb
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
        # # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigvals)[::-1]
        self.W_lda = eigvecs[:, sorted_indices[:len(self.classes)-1]].real  # (n_features, n_classes-1)

        # Project training dataset for calculating mean_class
        Z_trg = X_trg @ self.W_lda
        # Compute class means in projected space     
        mean_class = []
        for i, cls in enumerate(self.classes):
            mean_class.append(Z_trg[Y_trg.flatten() == cls].mean(axis=0))
        self.mean_class = mean_class
        
    def predict(self, X):
        # Predict using the LDA model
        Z = X @ self.W_lda
        return findPredictedValues(Z, self.mean_class, self.classes)

def main():
    # Main function to execute the LDA algorithm
    # 1. Reads in the data
    # # Normalize the features
    # # 2. Random shuffle the data
    # # 3. split the data into 2/3 for training, and 1/3 for validation
    X_trg, Y_trg, X_valid, Y_valid = getNormalizedDataSets(csvFileName=source_file_name, flgSkipHeader=1, try_extra=False)
    lda_obj = lda()
    lda_obj.fit(X_trg, Y_trg, X_valid, Y_valid)

    # Predict by assigning to nearest class mean in projected space
    Y_pred_trg = lda_obj.predict(X_trg)
    Y_pred_valid = lda_obj.predict(X_valid)
    # # Display metrics
    calcDisplayMetrics(Y_trg, Y_pred_trg, Y_valid, Y_pred_valid)
    print_confusion(Y_valid, Y_pred_valid)

if __name__ == "__main__":
    main()
