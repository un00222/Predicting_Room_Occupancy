import utils as u
import numpy as np

# sigmoid that does not cause overflow issues
def sigmoid(z):
    positives = (z >= 0)
    negatives = (z < 0)
    result = np.zeros_like(z, dtype=float)
    result[positives] = 1 / (1 + np.exp(-z[positives]))
    result[negatives] = np.exp(z[negatives]) / (1 + np.exp(z[negatives]))
    
    return result

def probabilities(X, alpha, beta, num_classes):
    num_thresholds = num_classes - 1
    samples = len(X)
    # P(Y≤j)
    threshold_probs = sigmoid(alpha - (X@beta).reshape(-1, 1))
    equals_probs = np.zeros((samples, num_classes))
    # P(Y <= 0)
    equals_probs[:, 0] = threshold_probs[:, 0]
    # P(Y <= j) - P(Y <= j-1)
    for j in range(1, num_thresholds):
        equals_probs[:, j] = threshold_probs[:, j] - threshold_probs[:, j-1]
    # P(Y = K)
    equals_probs[:, num_thresholds] = 1 - threshold_probs[:, -1]
    # for preventing log(0)
    return np.clip(equals_probs, 1e-15, 1-1e-15)

class ProportionalOdds:
    def __init__(self, learning_rate=1e-5, max_epochs=5000, tolerance=1e-10):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.alpha = None
        self.beta = None
        self.num_classes = 0
        
    def fit(self, training_X, training_Y, validation_X, validation_Y):
        training_Y = training_Y.flatten().astype(int)
        validation_Y = validation_Y.flatten().astype(int)
        # set the initial beta and alpha
        training_samples, feature_count = training_X.shape
        self.beta = np.zeros(feature_count)
        
        # calculate weights to help account for unbalanced labels
        counts = np.bincount(training_Y)
        self.num_classes = len(np.unique(training_Y))
        num_thresholds = self.num_classes - 1
        weights = (len(training_Y) / (self.num_classes * counts))[training_Y]
        
        p = np.cumsum(counts)[:-1] / counts.sum()
        self.alpha = np.log(p / (1 - p))
        
        tolerance = 1e-10
        prev_loss = float('inf')
        
        # train the data
        epochs = 0
        while epochs < self.max_epochs:
            epochs += 1
            # get proabilities for all classes
            training_prob = probabilities(training_X, self.alpha, self.beta, self.num_classes)
            
            # calculate beta gradient
            sig = sigmoid(self.alpha - (training_X@self.beta).reshape(-1, 1))
            # d/dx of sigmoid = sig(z)(1-sig(z))
            sig_deriv = sig * (1 - sig)
            error_beta = np.zeros((training_samples,self.num_classes))
            # for y=0
            error_beta[:, 0] = -sig_deriv[:, 0]
            # for y=1,2
            for j in range(1,num_thresholds):
                error_beta[:, j] = sig_deriv[:, j-1] - sig_deriv[:, j]
            # for y=3
            error_beta[:, -1] = sig_deriv[:, -2]
            
            pi_y = training_prob[np.arange(training_samples), training_Y]
            error_beta = (error_beta[np.arange(training_samples), training_Y] / pi_y) * weights
            
            beta_gradient = np.sum(error_beta.reshape(-1,1) * training_X, axis=0)
            
            # calculate alpha gradient
            # reuse sig_deriv and pi_y
            alpha_gradient = np.zeros(num_thresholds)
            for j in range(num_thresholds):
                y_equals_j = (training_Y == j)
                left = np.sum(sig_deriv[y_equals_j, j] / pi_y[y_equals_j] * weights[y_equals_j])
                y_equals_j_plus_1 = (training_Y == j+1)
                right = np.sum(sig_deriv[y_equals_j_plus_1, j] / pi_y[y_equals_j_plus_1] * weights[y_equals_j_plus_1])
                alpha_gradient[j] = left - right
            
            self.beta  += self.learning_rate * beta_gradient
            self.alpha += self.learning_rate * alpha_gradient
            self.alpha.sort()
            
            # check if accuracy has hit threshold
            if epochs % 1000 == 0:
                validation_prob = probabilities(validation_X, self.alpha, self.beta, self.num_classes)
                Y_pred_valid = np.argmax(validation_prob, axis=1)
                accuracy = np.mean(Y_pred_valid == validation_Y)
                validation_loss = -np.mean(np.log(validation_prob[np.arange(len(validation_Y)), validation_Y]))
                print(f"{epochs}: {accuracy*100:.2f}%, {validation_loss:.7f}")
                if abs(prev_loss - validation_loss) < tolerance:
                    print(f"Converged at epoch {epochs}.")
                    break
                prev_loss = validation_loss
        
        return self.alpha, self.beta

    def predict(self, X):
        return np.argmax(probabilities(X, self.alpha, self.beta, self.num_classes), 1)

def main():
    X_trg, Y_trg, X_valid, Y_valid = u.getNormalizedDataSets("Occupancy_Estimation.csv", 1)
    po = ProportionalOdds(max_epochs=20000)
    alpha, beta = po.fit(X_trg, Y_trg, X_valid, Y_valid)
    Y_pred_trg = po.predict(X_trg)
    Y_pred_valid = po.predict(X_valid)
    u.calcDisplayMetrics(Y_trg, Y_pred_trg, Y_valid, Y_pred_valid)
    u.print_confusion(Y_valid, Y_pred_valid)
    return

if __name__ == "__main__":
    main()