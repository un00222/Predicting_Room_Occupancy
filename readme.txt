Required Python Libraries via pip
---------------------------------------------------------------------------------------------------------------------
Ensure the following libraries are installed. If not, run pip install [library]:
    numpy, pandas, imbalanced-learn

minimum Python version: 3.2 (required to use concurrent.futures in bagging.py)

###################################################################################################################################################################
Utils (utils.py) - required libraries: numpy, pandas, imbalanced-learn
---------------------------------------------------------------------------------------------------------------------
This file is not meant to be run independently. It contains common functions used to preprocess 
data and calculate metrics to avoid repeated code.

normalize_data - returns normalized data
    ingests all features that can be converted from a string to a floating point number
        all floating point numbers are normalized
        for each feature that cannot be converted to a float, a sample value is printed to the console
    optional boolean try_extra will convert dates to one-hot-encoded days of the week and times to integer hour
    Uncomment lines 75-81 to use smote (left commented out because it was ineffective)

splitTrgValidation - returns data_training_X, data_training_Y, data_validation_X, data_validation_Y
    splits the input data into training x and y and validation x and Y
    the split is done at 2/3 training data, 1/3 validation data

calcMetrics - returns precision, recall, f_measure, accuracy
    calculates prceision, recall, f measure, and accuracy for the input predictions against the input true classifications
    
    console output:
        for each of the unique classes in the input true classifications, the class accuracy is printed
            format: {class}: {true positive}/{true positive + false negative}={accuracy}%

calcDisplayMetrics - returns nothing
    Prints precision, recall, f measure, and accuracy for the training predictions and the validation predictions input
    The metrics are printed in a table format for clear comparison between the training and validation results

    example table layout:
    Metric     | Training Set | Validation Set
    _____________________________________________
    Precision  | 0.1234          | 0.1234
    Recall     | 0.5678          | 0.5678
    F_Measure  | 0.9876          | 0.9876
    Accuracy   | 54.32%          | 54.32%

getNormalizedDataSets - returns X_trg, Y_trg, X_valid, Y_valid
    Generates the training and validation data

print_confusion - returns nothing
    prints a confusion matrix for the actual Y value vs. the predicted
    The columns are the predictions, while the rows are the true values.
    An extra column and row are display summing the totals.

    example matrix format:
            0    1    2    3  |    T
    --------------------------------
     0 |    4    2    0    0  |    6
     1 |    1    3    2    0  |    6
     2 |    0    0    5    1  |    6
     3 |    0    0    3    3  |    6
    --------------------------------
     T |    5    5   10    4  |   24

###################################################################################################################################################################
KNN (KNNClassification.py) - required libraries: numpy, utils
---------------------------------------------------------------------------------------------------------------------
To run KNNClassification.py place the "occupancy_estimation.csv" and "utils.py" files in the same directory as KNNClassification.py. 
The "metrics.py" file must be in the same directory as well.

No parameters are required to run KNNClassification.py.

How to use the class to for making predictions:
    1. Import KNNClassification into a Python script.
    2. Create an object of the KNNClassifier class. 
        2a. Provide the constructor with the desired k (nearest neighbors to compare to), weighting option, and stability constant.
            Default constructor parameters are k=5, weighting="uniform", and eps=1e-12
                weighting has 2 options: "uniform" and "distance"
    3. Call the fit function of the class and input the training X and Y data, and the validation X and Y data.
    4. After running fit, predict can be run for any data subset of the same dataset training was conducted on.

main function:
    Finds the best k for both uniform and distance weighting.
    Selects the combo of k and weighting that produced the best results.
    Trains the model using the selected k and weighting and generates the predictions.
    
    outputs on console:
        output from normalize_data
        class distributions for training and validation subsets
            example format for 1 class: "Class 1: 295 samples"
        validation accuracy for each k checked in the tuning stage
            best k for the tuning and it's validation accuracy
        *tuning stage is done twice (once for tuning using each of the weighting options)
        output from calcMetrics and calcDisplayMetrics

###################################################################################################################################################################
Closed Form Linear Regression (closed_form_linear_regression.py) - required libraries: numpy, utils
---------------------------------------------------------------------------------------------------------------------
To run closed_form_linear_regression.py place the "occupancy_estimation.csv" and "utils.py" files in the same directory as Pythonclosed_form_linear_regression.py. 
The "metrics.py" file must be in the same directory as well.

No parameters are required to run closed_form_linear_regression.py.

How to use the class to for making predictions:
    1. Import closed_form_linear_regression into a Python script.
    2. Create an object of the lda class. 
        There are no parameters for the constructor.
    3. Call the fit function of the class and input the training X and Y data, and the validation X and Y data.
    4. After running fit, predict can be run for any data subset of the same dataset training was conducted on.

main function:
    Trains the Closed Form Linear Regression model on the normalized data from utils for room_occupancy.csv.
    Uses the trained model to predict the classifications for each row of the dataset.
    Calculates metrics specific to linear regression.
    Outputs the metrics from utils for the prediction.
    
    outputs on console:
        output from normalize_data
        linear regression-specific metrics
        output from calcMetrics and calcDisplayMetrics

###################################################################################################################################################################
Linear Discriminant Analysis (LDA.py) - required libraries: numpy, utils
---------------------------------------------------------------------------------------------------------------------
To run LDA.py place the "occupancy_estimation.csv" and "utils.py" files in the same directory as LDA.py. 
The "metrics.py" file must be in the same directory as well.

No parameters are required to run LDA.py.

How to use the class to for making predictions:
    1. Import LDA into a Python script.
    2. Create an object of the lda class. 
        There are no parameters for the constructor.
    3. Call the fit function of the class and input the training X and Y data, and the validation X and Y data.
    4. After running fit, predict can be run for any data subset of the same dataset training was conducted on.

main function:
    Trains the LDA model on the normalized data from utils for room_occupancy.csv.
    Uses the trained model to predict the classifications for each row of the dataset.
    Outputs the metrics from utils for the predictions.

    outputs on console:
        output from normalize_data
        output from calcMetrics and calcDisplayMetrics

###################################################################################################################################################################
Proportional Odds (proportional_odds.py) - required libraries: numpy, utils
---------------------------------------------------------------------------------------------------------------------
To run proportional_odds.py place the "occupancy_estimation.csv" and "utils.py" files in the same directory as proportional_odds.py. 
The "metrics.py" file must be in the same directory as well.

No parameters are required to run proportional_odds.py.

How to use the class to for making predictions:
    1. Import proportional_odds into a Python script.
    2. Create an object of the ProportionalOdds class. 
        2a. Provide the constructor with the desired learning rate, maximum epochs, and desired tolerance (checked every 1000 epochs to see if loss difference was lower than it).
            Default constructor parameters are learning_rate=1e-5, max_epochs=5000, and tolerance=1e-10
    3. Call the fit function of the class and input the training X and Y data, and the validation X and Y data.
            Validation data during training is used to compute accuracy per 1000 epochs and check loss.
    4. After running fit, predict can be run for any data subset of the same dataset training was conducted on.
        4a. Running predict produces a vector of the predicted classes for each feature row in the input feature matrix.
        4b. predict cannot be run before fit, as alpha and beta have not been generated.

main function:
    A proportional odds model is trained with a learning rate of 1e-5 and maximum epochs of 20000.
    Periodic updates on provided every 1000 epochs on validation accuracy and loss.

    outputs on console:
        output from normalize_data
        Accuracy and loss per 1000 epochs:
            example format: 2000: 87.36%, 0.4797889
        output from calcMetrics and calcDisplayMetrics

###################################################################################################################################################################
Bagging Ensemble Model (bagging.py) - required libraries: numpy, utils, and one of the following: KNNClassification, Pythonclosed_form_linear_regression, LDA, proportional_odds
---------------------------------------------------------------------------------------------------------------------
To run bagging.py place the "occupancy_estimation.csv" and "utils.py" files in the same directory as bagging.py.
Ensure that the model that is being used as the base's file is also located in the same directory. 
The "metrics.py" file must be in the same directory as well.

Hardware note: The training is done in parallel using all but 2 CPU cores, or 1 if the CPU has 1 or 2. It is necessary for the training to occur on multiple thread to reduce the 
    overall execution time and allow for larger subset counts.

No parameters are required to run bagging.py. Instead, select a model by uncommenting just the lines under the comment for the model and ensure that the base python file
    is in the same directory. 

2 options for running the bagging:
    1. bag - returns pred_train, pred_val
        Trains the bagging model and produces predictions for the input training and validation data.
        inputs: 
            base_model_class: The class being trained for each subset (KNNClassifier, ClosedFormRegression, lda, ProportionalOdds)
            num_subsets: The number of subsets to train that will be the voters
            X_train: The training data
            Y_train: The training true classifications
            X_val: The validation data
            max_workers: The number of cores to use for training subset models in parallel
            kwargs_con: dictionary of additional arguements used in the constructor of the base_model_class
            kwargs_fit: dictionary of additional arguements used in the fit function of the base_model_class

    2. determine_best_subset_count - returns nothing
        Performs bagging using the selected base model for all subsets in the input range.
        Prints the number of subsets that produced the highest accuracy and displays the metrics for that bagging model.
        inputs:
            base_model_class: The class being trained for each subset (KNNClassifier, ClosedFormRegression, lda, ProportionalOdds)
            X_train: The training data
            Y_train: The training true classifications
            X_val: The validation data
            Y_val: The validation true classifications
            max_workers: The number of cores to use for training subset models in parallel
            min_subsets: Minimum number of subsets to test
            max_subsets: Maximum number of subsets to test
            kwargs_con: dictionary of additional arguements used in the constructor of the base_model_class
            kwargs_fit: dictionary of additional arguements used in the fit function of the base_model_class
        In the main function of bagging.py, comment out lines 135 and 136 when running this function, as it does not
            return predictions.

outputs on console:
    output from normalize_data
    number of CPU cores being used to run model training in parallel
    model-specific training information
    output from calcMetrics and calcDisplayMetrics

unused code:
    balance_training_data:
        oversamples minority classes and undersamples the majority class
        returns a new data set that has an even distribution across all classes
        removed because it made bagging perform worse with every single base model