import numpy as np
import pandas as pd
import utils as u

def get_training_and_testing_data():
    data = pd.read_csv("occupancy_estimation.csv")
    df = pd.DataFrame(data)
    string_columns = df.select_dtypes(exclude=['number']).columns
    shuffled_data = data.sample(frac=1, random_state=np.random.RandomState(0)).reset_index(drop=True)

    X = shuffled_data.drop(string_columns, axis=1).values
    X = shuffled_data.drop("Room_Occupancy_Count", axis=1).values
    y = shuffled_data['Room_Occupancy_Count'].values
    test_train_split = int(len(shuffled_data) * (2 / 3))
    
    x_train = X[:test_train_split]
    x_test = X[test_train_split:]
    y_train = y[:test_train_split]
    y_test = y[test_train_split:]
    return x_train, x_test, y_train, y_test

class ClosedFormRegression:
    def __init__(self, clip_min=0, clip_max=3):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.theta = None

    def fit(self, x_train, y_train, verbose=False):
        x_train_processed = np.column_stack([np.ones(len(x_train)), x_train])

        self.theta = np.linalg.inv(x_train_processed.T @ x_train_processed) @ x_train_processed.T @ y_train
        if verbose:
            print("Y-intercept/Bias =", self.theta[0])
            
        return self.theta
        
    def predict(self, X):
        X = np.column_stack([np.ones(len(X)), X])
        y_predicted = X @ self.theta
        return np.clip(np.round(y_predicted, 0), self.clip_min, self.clip_max).astype(int)


def get_r_squared(y_true, y_pred):
    return 1 - (
            np.sum((y_true - y_pred) ** 2) /
            np.sum((y_true - np.mean(y_true)) ** 2)
    )

def calculate_smape(y_true, y_pred):
    return 100 / len(y_true) * np.sum(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-15)
    )

def main():
    x_train, y_train, x_test, y_test = u.getNormalizedDataSets("occupancy_estimation.csv", 1)
    cfr = ClosedFormRegression()
    cfr.fit(x_train, y_train, verbose=True)
    y_train_predicted = cfr.predict(x_train)
    y_test_predicted = cfr.predict(x_test)
    train_deter_coeff = get_r_squared(y_train, y_train_predicted)
    test_deter_coeff = get_r_squared(y_test, y_test_predicted)

    print("Train r-squared =", train_deter_coeff)
    print("Test r-squared =", test_deter_coeff)

    rmse_test = np.sqrt(np.mean((y_test - y_test_predicted) ** 2))
    rmse_train = np.sqrt(np.mean((y_train - y_train_predicted) ** 2))
    print("Test RMSE =", rmse_test)
    print("Train RMSE =", rmse_train)
    
    smape_train = calculate_smape(y_train, y_train_predicted)
    smape_test = calculate_smape(y_test, y_test_predicted)
    print("Train SMAPE =", smape_train)
    print("Test SMAPE =", smape_test)
    
    print("Test accuracy =", np.mean(y_test == y_test_predicted))
    print("Training accuracy =", np.mean(y_train == y_train_predicted))
    u.calcDisplayMetrics(y_train, y_train_predicted, y_test, y_test_predicted)
    u.print_confusion(y_test, y_test_predicted)

if __name__ == "__main__":
    main()
