import numpy as np;
import pandas as pd
from imblearn.over_sampling import SMOTE
# Read data from a file
def readFromFile(fileName,flgSkipHeader):
    data = np.loadtxt(fileName, delimiter=',', skiprows=flgSkipHeader, 
                      dtype=str, encoding='utf-8')
    return np.array(data)

# Normalize all the features

def normalize_data(data, try_extra=False):
    # attempt to do more feature engineering on date and time
    if try_extra:
        df = pd.DataFrame(data)
        # remove CO2 slope feature
        #df = df.drop(15)
        
        # covert time to hours
        for col in df.columns:
            if df[col].str.match(r'\d{2}:\d{2}:\d{2}').all():
                df[col] = pd.to_datetime(df[col], format='%H:%M:%S').dt.hour
                # remove early morning and late night hours
                # df = df[~df[col].between(0,4)]
                # df = df[~df[col].between(21,24)]
            
        # # one-hot-encode days of the week
        one_hots = []
        for col in df.columns:
                                                                    # yyyy/mm/dd
            if df[col].dtype == object and df[col].str.match(r'^\d{4}/\d{2}/\d{2}$').all():
                df[col] = pd.to_datetime(df[col], format='%Y/%m/%d')
                # remove weekends
                # df = df[~df[col].dt.dayofweek.isin(values=[5, 6])]
                df[col] = df[col].dt.day_name()
                dummies = pd.get_dummies(df[col])
                one_hots.append(dummies)
                df = df.drop(columns=[col])
        data = df.to_numpy()

    #dk989 - can think of a better technique here, but it will work generically 
    
    # Identify columns that can be converted to float
    float_cols = []
    for i in range(data.shape[1]):
        try:
            data[:, i].astype(float)
            float_cols.append(i)
        except ValueError:
            print(f"non-float column: {data[:, i][0]}")
            continue
    # Keep columns that can be converted to float
    cols_to_keep = float_cols

    data = data[:, cols_to_keep]

    # Calculate unique values of last column
    # unique_values_last_col = np.unique(data[:, -1])
    # print("Unique values in last column:", unique_values_last_col)

    data = data.astype(float)
    
    features = data[:,:-1]
    labels = data[:,-1]
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0, ddof=1)
    features = (features - mean) /std
    if try_extra and one_hots:
        one_hots = [oh.to_numpy() for oh in one_hots]
        features = np.hstack([features] + one_hots)
    data = np.hstack((features, labels.reshape(-1,1)))
    # Note: Applying SMOTE to the entire dataset before splitting can lead to
    # data leakage from the validation set into the training set.
    # It is generally recommended to apply SMOTE only to the training data after the split.
    # try:
    #     smote = SMOTE(random_state=42)
    #     features, labels = smote.fit_resample(data[:, :-1], data[:, -1])
    #     data = np.hstack((features, labels.reshape(-1, 1)))
    # except ImportError:
    #     print("imblearn is not installed. Skipping SMOTE.")
    #     print("Install it with: pip install imbalanced-learn")
        
    return data

# Split the data into training (2/3) and validation (1/3)
def splitTrgValidation(data):

    np.random.seed(0)
    # Use this seed to shuffle the data for reproducibility`` and compare results with others
    np.random.shuffle(data)

    split_at= int(2/3 * data.shape[0])
    data_training_X = data[:, range(data.shape[1] - 1)] [:split_at]
    data_training_Y = data[:, data.shape[1] - 1 ].astype(float)[:split_at].reshape(-1,1)
    data_validation_X = data[:, range(data.shape[1] - 1)] [split_at:]
    data_validation_Y = data[:, data.shape[1] - 1].astype(float)[split_at:].reshape(-1,1)
    return data_training_X, data_training_Y, data_validation_X, data_validation_Y

# calculate the metrics
def calcMetrics(Y, Y_predicted, verbose=True):
    Y = Y.flatten().astype(int)
    Y_predicted = Y_predicted.flatten().astype(int)
    
    num_classes = len(np.unique(Y))
    epsilon = 1e-15 # To avoid divide by 0 issues
    TP = FP = TN = FN = precision = recall = f_measure = 0
    for i in np.unique(Y):
        TP = np.sum((Y == i) & (Y_predicted == i))
        FP = np.sum((Y != i) & (Y_predicted == i))
        TN = np.sum((Y != i) & (Y_predicted != i))
        FN = np.sum((Y == i) & (Y_predicted != i))
        precision += TP / (TP + FP + epsilon)
        recall += TP / (TP + FN + epsilon)
        if verbose:
            print(f"{i}: {TP:>4}/{TP+FN:<4}={TP / (TP + FN)*100:.2f}%")
    precision /= num_classes
    recall /= num_classes
    f_measure += 2 * precision * recall / (precision + recall + epsilon)
    accuracy = np.mean(Y_predicted == Y)

    return precision, recall, f_measure, accuracy

# Display the metrics in a table format
def calcDisplayMetrics(Y_trg, Y_pred_trg, Y_valid, Y_pred_valid):
    # # Training data set metrics
    trg_precision, trg_recall, trg_f_measure, trg_accuracy = calcMetrics(Y_trg, Y_pred_trg.astype(float))
    # # Validation data set metrics
    valid_precision, valid_recall, valid_f_measure, valid_accuracy = calcMetrics(Y_valid, Y_pred_valid.astype(float))

    # Display the metrics in the table
    headers = ["Metric", "Training Set", "Validation Set"]
    rows = [
        ["Precision", trg_precision, valid_precision ],
        ["Recall", trg_recall, valid_recall ],
        ["F_Measure", trg_f_measure, valid_f_measure ],
        ["Accuracy", "{:.2f}%".format(trg_accuracy * 100), "{:.2f}%".format(valid_accuracy * 100)]
    ]

    # Print header
    print(f"{headers[0]:<10} | {headers[1]:<10} | {headers[2]:<10}")
    print("_" * 45)

    # Print rows
    for row in rows:
        if row[0] == "Accuracy":
            print(f"{row[0]:<10} | {row[1]:<15} | {row[2]:<15}")
        else :
            print(f"{row[0]:<10} | {row[1]:<15.4f} | {row[2]:<15.4f}")

def getNormalizedDataSets(csvFileName, flgSkipHeader, try_extra=False):
    # 1. Reads in the data
    data = readFromFile(csvFileName, flgSkipHeader)

    # Normalize the features
    data = normalize_data(data, try_extra)

    # 2. Random shuffle the data
    # 3. split the data into 2/3 for training, and 1/3 for validation
    X_trg, Y_trg, X_valid, Y_valid = splitTrgValidation(data)
    return X_trg, Y_trg, X_valid, Y_valid

def print_confusion(Y, predictions):
    Y = Y.flatten().astype(int)
    predictions = predictions.flatten().astype(int)
    
    classes = np.unique(Y)
    print("     ", end='')
    for c in classes:
        print(f"{c:>4}", end=' ')
    print(" |    T")
    print("-"*((len(classes)+2)*5+2))
    
    sums = np.zeros(len(classes)).astype(int)
    for i in classes:
        print(f"{i:>2} | ", end='')
        sum = 0
        for idx, j in enumerate(classes):
            count = np.sum((Y == i) & (predictions == j))
            print(f"{count:>4}", end=' ')
            sum += count
            sums[idx] += count
        print(f" | {sum:>4}")
        
    
    # print predicted class totals + total guesses
    print("-"*((len(classes)+2)*5+2))
    print(" T | ", end='')    
    for sum in sums:
        print(f"{sum:>4}", end=' ') 

    print(f" | {np.sum(sums):>4}")
