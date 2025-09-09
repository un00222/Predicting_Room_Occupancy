import numpy as np;
from utils import getNormalizedDataSets, calcDisplayMetrics, print_confusion

# KNNClassification.py
class KNNClassifier:
    def __init__(self, k=5, weighting="uniform", eps=1e-12):
        """
        k: number of neighbors
        weighting: "uniform" or "distance"
        eps: small constant for numerical stability with distance weighting
        """
        assert weighting in ("uniform", "distance")
        self.k = int(k)
        self.weighting = weighting
        self.eps = eps
        self.X_train = None
        self.y_train = None
        self.n_classes = None

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train, dtype=float)
        # Handle both 1D and 2D y_train arrays
        if y_train.ndim > 1:
            self.y_train = np.asarray(y_train, dtype=int).ravel()
        else:
            self.y_train = np.asarray(y_train, dtype=int)
        self.n_classes = int(np.max(self.y_train)) + 1
        return self

    def _pairwise_distances(self, A, B):
        # Euclidean distances using vectorized expansion:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        A2 = np.sum(A*A, axis=1, keepdims=True)          # (nA,1)
        B2 = np.sum(B*B, axis=1, keepdims=True).T        # (1,nB)
        d2 = A2 + B2 - 2.0 * (A @ B.T)
        np.maximum(d2, 0.0, out=d2)
        return np.sqrt(d2, out=d2)                       # (nA,nB)

    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=float)
        dists = self._pairwise_distances(X_test, self.X_train)       # (n_test, n_train)
        # indices of k nearest neighbors (no full sort)
        nn_idx = np.argpartition(dists, self.k - 1, axis=1)[:, :self.k]
        nn_labels = self.y_train[nn_idx]                              # (n_test, k)

        if self.weighting == "uniform":
            preds = np.empty(nn_labels.shape[0], dtype=int)
            for i in range(nn_labels.shape[0]):
                counts = np.bincount(nn_labels[i], minlength=self.n_classes)
                preds[i] = int(np.argmax(counts))  # tie -> smallest label
            return preds
        else:
            # distance-weighted vote: weight = 1 / (distance + eps)
            nn_d = np.take_along_axis(dists, nn_idx, axis=1)
            w = 1.0 / (nn_d + self.eps)
            preds = np.empty(nn_labels.shape[0], dtype=int)
            for i in range(nn_labels.shape[0]):
                weights_per_class = np.zeros(self.n_classes, dtype=float)
                for lbl, ww in zip(nn_labels[i], w[i]):
                    weights_per_class[lbl] += ww
                preds[i] = int(np.argmax(weights_per_class))
            return preds

# Try a range of k values
def tune_k(X_trg, Y_trg, X_valid, Y_valid, k_values=None, weighting="uniform"):
    if k_values is None:
        k_values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] 
    best_k, best_acc = None, -1.0
    results = []
    for k in k_values:
        model = KNNClassifier(k=k, weighting=weighting).fit(X_trg, Y_trg)
        y_pred = model.predict(X_valid)
        # Handle both 1D and 2D Y_valid arrays
        Y_valid_flat = Y_valid.flatten() if Y_valid.ndim > 1 else Y_valid
        acc = float(np.mean(y_pred == Y_valid_flat))
        results.append((k, acc))
        if acc > best_acc:
            best_k, best_acc = k, acc
    return best_k, best_acc, results

def knn_train(X_trg, Y_trg, X_valid, Y_valid, verbose=True):
    # Tune k with uniform weighting
    if verbose:
        print("\n")
        print("K-VALUE TUNING (Uniform Weighting)")
    best_k, best_acc, results = tune_k(X_trg, Y_trg, X_valid, Y_valid,
                                        k_values=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
                                        weighting="uniform")
    if verbose:
        print("k tuning (uniform weighting):")
        for k, acc in results:
            print(f"  k={k:>2}: val acc = {acc:.4f}")
    print(f"\nBest k = {best_k}  (val acc = {best_acc:.4f})")
    if verbose:
        print("\n")
        print("K-VALUE TUNING (Distance Weighting)")
    best_k_dw, best_acc_dw, results_dw = tune_k(X_trg, Y_trg, X_valid, Y_valid, 
                                                    k_values=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
                                                    weighting="distance")
    if verbose:
        print("\n")
        print("k tuning (distance weighting):")
        for k, acc in results_dw:
            print(f"  k={k:>2}: val acc = {acc:.4f}")
    print(f"Best (distance) k = {best_k_dw}  (val acc = {best_acc_dw:.4f})")

    # Choose the better weighting method
    if best_acc_dw > best_acc:
        final_k = best_k_dw
        final_weighting = "distance"
        print(f"\nUsing distance weighting with k={final_k}")
    else:
        final_k = best_k
        final_weighting = "uniform"
        print(f"\nUsing uniform weighting with k={final_k}")

    # Train final model and report metrics
    if verbose:
        print("\n")
        print("FINAL MODEL EVALUATION")
    knn = KNNClassifier(k=final_k, weighting=final_weighting).fit(X_trg, Y_trg)
    Y_pred_trg = knn.predict(X_trg)
    Y_pred_valid = knn.predict(X_valid)
    return Y_pred_trg, Y_pred_valid, final_k, final_weighting

def main():
    """
    Main function to run KNN classifier on Room Occupancy dataset
    """
    # Dataset configuration
    csv_path = "Occupancy_Estimation.csv"
    skip_header = 1

    try:
        # Prepare data
        X_trg, Y_trg, X_valid, Y_valid = getNormalizedDataSets(csv_path, skip_header)
        print(f"Train: {X_trg.shape}, Valid: {X_valid.shape}  |  #features = {X_trg.shape[1]}")
        
        # Check class distribution
        print("\nClass distribution in training set:")
        unique_trg, counts_trg = np.unique(Y_trg.flatten(), return_counts=True)
        for u, c in zip(unique_trg, counts_trg):
            print(f"Class {int(u)}: {c} samples")
        
        print("\nClass distribution in validation set:")
        unique_val, counts_val = np.unique(Y_valid.flatten(), return_counts=True)
        for u, c in zip(unique_val, counts_val):
            print(f"Class {int(u)}: {c} samples")

        Y_pred_trg, Y_pred_valid, final_k, final_weighting = knn_train(X_trg, Y_trg, X_valid, Y_valid)
        
        calcDisplayMetrics(Y_trg, Y_pred_trg, Y_valid, Y_pred_valid)
        print_confusion(Y_valid, Y_pred_valid)
        
        print(f"\nFinal Model Configuration:")
        print(f"- Algorithm: K-Nearest Neighbors")
        print(f"- k value: {final_k}")
        print(f"- Weighting: {final_weighting}")
        print(f"- Distance metric: Euclidean")

    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_path}'")
        print("Please ensure the dataset file is in the correct location.")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your data format and file path.")

if __name__ == "__main__":
    main()