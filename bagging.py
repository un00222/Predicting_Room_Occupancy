import utils as u 
from closed_form_linear_regression import ClosedFormRegression
from KNNClassification import KNNClassifier 
from proportional_odds import ProportionalOdds 
from LDA import lda
import numpy as np 
from concurrent.futures import ThreadPoolExecutor 
import os

def vote(preds): 
    preds = preds.astype(int)
    return np.array([ np.bincount(col).argmax() for col in preds.T ]) 

def train_subset_model(base_model_class, X_train, Y_train, X_val, kwargs_con={}, kwargs_fit={}): 
    indices = np.random.choice(len(X_train), size=len(X_train), replace=True) 
    X_train_subset = X_train[indices] 
    Y_train_subset = Y_train[indices] 
    model = base_model_class(**kwargs_con)
    model.fit(X_train_subset, Y_train_subset, **kwargs_fit) 
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    return np.ravel(train_pred), np.ravel(val_pred) 

def bag(base_model_class, num_subsets, X_train, Y_train, X_val, max_workers, kwargs_con={}, kwargs_fit={}): 
    def worker(_):
        return train_subset_model(base_model_class, X_train, Y_train, X_val, kwargs_con=kwargs_con, kwargs_fit=kwargs_fit)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor: 
        parallel_results = list(executor.map(worker, range(num_subsets))) 
        
    train_preds = np.array([res[0] for res in parallel_results]) 
    val_preds = np.array([res[1] for res in parallel_results]) 
    train_pred_overall = vote(train_preds) 
    val_pred_overall = vote(val_preds)
        
    return train_pred_overall, val_pred_overall 
    
def determine_best_subset_count(base_model_class, X_train, Y_train, X_val, Y_val, max_workers, min_subsets=1, max_subsets=100, kwargs_con={}, kwargs_fit={}): 
    best_train_pred = [] 
    best_val_pred = [] 
    best_metrics = -1 
    best_subset_count = 1 
    print(f"Deciding between number of subsets in range of {min_subsets} to {max_subsets}") 
    Y_train_comp = Y_train.flatten()
    Y_val_comp = Y_val.flatten()
    for i in range(min_subsets, max_subsets + 1): 
        if len(best_train_pred) != 0 and i % 50 == 1: 
            print(f"Completed checking subsets {min_subsets}-{i-1}") 
            print(f"best number so far = {best_subset_count}")
            print(f"training accuracy: {np.mean(best_train_pred == Y_train_comp)*100:.2f}%\nvalidation accuracy: {np.mean(best_val_pred == Y_val_comp)*100:.2f}%\n") 
        
        train_pred, val_pred = bag(base_model_class, i, X_train, Y_train, X_val, max_workers, kwargs_con, kwargs_fit) 
        metrics = u.calcMetrics(Y_val, val_pred,verbose=False) 
        metrics_sum = np.sum(metrics) 
        
        if metrics_sum > best_metrics: 
            best_metrics = metrics_sum 
            best_subset_count = i 
            best_train_pred = train_pred 
            best_val_pred = val_pred 
            
    print(f"Best number of subsets in range of {min_subsets} to {max_subsets} is {best_subset_count}") 
    u.calcDisplayMetrics(Y_train, best_train_pred, Y_val, Y_pred_valid=best_val_pred) 

# unused here because over+undersampling did not improve results, similar to how SMOTE did not
def balance_training_data(X_train, Y_train):    
    # Separate majority and minority classes
    X_0 = X_train[Y_train == 0]
    Y_0 = Y_train[Y_train == 0]
    X_not0 = X_train[Y_train != 0]
    Y_not0 = Y_train[Y_train != 0]
    
    # Compute new size for majority class
    percent_to_keep = 0.2
    num_0s = int(len(Y_0) * percent_to_keep)
    
    # Randomly sample majority class
    indices = np.random.choice(len(Y_0), size=num_0s, replace=False)
    X_0 = X_0[indices]
    Y_0 = Y_0[indices]
    
    # randomly copy from minority class to match number of majority samples
    X_us = []
    Y_us = []
    for c in range(1,4):
        X_c = X_not0[Y_not0 == c]
        Y_c = Y_not0[Y_not0 == c]
        indices = np.random.choice(len(Y_c), size=num_0s, replace=True)
        X_c = X_c[indices]
        Y_c = Y_c[indices]
        X_us.append(X_c)
        Y_us.append(Y_c)
    
    # Combine
    X_balanced = np.vstack([X_0] + X_us)
    Y_balanced = np.hstack([Y_0] + Y_us)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(Y_balanced))
    X_balanced = X_balanced[shuffle_idx]
    Y_balanced = Y_balanced[shuffle_idx]
    
    return X_balanced, Y_balanced
            
def main(): 
    X_train, Y_train, X_val, Y_val = u.getNormalizedDataSets("occupancy_estimation.csv", 1, try_extra=False) 
    
    # over and undersample the minority classes and majority class
    # X_train, Y_train = balance_training_data(X_train, Y_train.flatten()) # did not improve results
    
    num_cores = os.cpu_count()
    max_workers = max(1, num_cores - 2)
    print(f"using up to {max_workers} cores\n")
    
    """Find Best Number of Subsets in a Range (change model as needed)"""
    #determine_best_subset_count(KNNClassifier, X_train, Y_train.flatten(), X_val, Y_val, max_workers)
    
    """Closed Form Regression Bagging"""
    # X_train, Y_train, X_val, Y_val = u.getNormalizedDataSets("occupancy_estimation.csv", 1, try_extra=False) # either handle non-invertible case, or avoid)
    # pred_train, pred_val = bag(ClosedFormRegression, 45, X_train, Y_train, X_val, max_workers) 
    
    """Proportional Odds Bagging"""
    # kwargs_con = {'max_epochs': 2000}
    # kwargs_fit = {'validation_X': X_val ,'validation_Y': Y_val}
    # pred_train, pred_val = bag(ProportionalOdds, 10, X_train, Y_train, X_val, max_workers, kwargs_con, kwargs_fit) 
    
    """KNN Bagging"""
    kwargs_con = {'k': 7, 'weighting': 'distance'}
    pred_train, pred_val = bag(KNNClassifier, 55, X_train, Y_train, X_val, max_workers) 
    
    """LDA Bagging"""
    # kwargs_fit = {'X_valid': X_val ,'Y_valid': Y_val}
    # pred_train, pred_val = bag(lda, 100, X_train, Y_train, X_val, max_workers, kwargs_fit=kwargs_fit)
    
    u.calcDisplayMetrics(Y_train, pred_train, Y_val, pred_val)
    u.print_confusion(Y_val, pred_val)
    return 

if __name__ == "__main__":
    main()