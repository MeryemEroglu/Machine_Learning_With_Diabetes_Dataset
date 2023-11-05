##############################
# Diabete Feature Engineering
##############################

# Problem: It is requested to develop a machine learning model that can predict whether individuals have diabetes or not when their features are specified.
# Before developing the model, it is expected that you perform the necessary data analysis and feature engineering steps.

# The dataset is a part of the large dataset maintained at the National Institute of Diabetes and Digestive and Kidney Diseases in the United States.
# These data are used for a diabetes study conducted on Pima Indian women aged 21 and above, living in Phoenix, the fifth-largest city in the state of Arizona in the United States.
# The dataset consists of 768 observations and 8 numerical independent variables.
# The target variable is specified as 'outcome,' where 1 indicates a positive diabetes test result, and 0 indicates a negative result.

#Pregnancies: Number of pregnancies
#Glucose: Glucose
#BloodPressure: Blood pressure (Diastolic)
#SkinThickness: Skin Thickness
#Insulin: Insulin
#BMI: Body Mass Index
#DiabetesPedigreeFunction: A function that calculates the probability of having diabetes based on family history.
#Age: Age (years)
#Outcome: Information about whether a person has diabetes. Afflicted (1) or not (0).

#TASK 1: EXPLORATORY DATA ANALYSIS
       # Step 1: Examine the overall picture.
       # Step 2: Capture numeric and categorical variables.
       # Step 3: Analyze numeric and categorical variables.
       # Step 4: Perform target variable analysis. (Mean of the target variable by categorical variables, mean of numeric variables by the target variable)
       # Step 5: Perform outlier analysis.
       # Step 6: Perform missing data analysis.
       # Step 7: Perform correlation analysis.

#TASK 2: FEATURE ENGINEERING
       # Step 1: Perform necessary operations for missing and outlier values. There are no missing observations in the dataset, but variables like Glucose, Insulin, etc.
                # containing 0 values may indicate missing values. For example, a person's glucose or insulin value cannot be 0.
                # You can consider replacing zero values with NaN in the relevant variables and then apply operations for missing values.
       # Step 2: Create new variables.
       # Step 3: Perform encoding operations.
       # Step 4: Standardize numeric variables.
       # Step 5: Build a model.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv('diabetes.csv')
df.head()
df.dtypes
df.isnull().sum()/df


######################################
#TASK 1: EXPLORATORY DATA ANALYSIS
######################################

# Step 1: Examine the overall picture.

def check_df(dataframe, head= 5):
    """
    This function provides an overview of a DataFrame including its shape, data types,
    the first 'head' rows, the last 'head' rows, the count of missing values, and selected quantiles.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame to be analyzed.
    head : int, optional
        Number of rows to display from the beginning and end of the DataFrame (default is 5).

    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.99, 1]).T)

check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Provides the names of categorical, numeric, and categorical but cardinal variables in the dataset.
    Note: Numeric-appearing categorical variables are also included in the categorical variables.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame to be analyzed.
    cat_th : int, optional
        The threshold value for variables that are numerical but categorical (default is 10).
    car_th : int, optional
        The threshold value for categorical but cardinal variables (default is 20).

    Returns
    -------
    cat_cols : list
        List of categorical variables.
    num_cols : list
        List of numeric variables.
    cat_but_car : list
        List of categorical-appearing cardinal variables.

    Notes
    -----
    cat_cols + num_cols + cat_but_car = total number of variables
    num_but_cat is within cat_cols.

    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

##################################
#ANALYSIS OF CATEGORICAL VARIABLES
##################################

def cat_summary(dataframe, categorical_col, plot=False):
    """
    Display a summary of a categorical variable in a DataFrame, including value counts and ratios.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame containing the categorical variable.
    categorical_col : str
        The name of the categorical column to be analyzed.
    plot : bool, optional
        If True, display a countplot to visualize the distribution (default is False).

    """
    print(pd.DataFrame({categorical_col: dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts()/len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[categorical_col], data=dataframe)
        plt.show()


cat_summary(df, "Outcome")


##################################
#ANALYSIS OF NUMERICAL VARIABLES
##################################

def num_summary(dataframe, numerical_col, plot=False):
    """
    Display a summary of a numerical variable in a DataFrame, including descriptive statistics and an optional histogram.

    Parameters
    ----------
    dataframe (DataFrame): The DataFrame containing the numerical variable.
    numerical_col (str): The name of the numerical column to be analyzed.
    plot (bool, optional): If True, display a histogram to visualize the distribution (default is False).

    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)


########################################
#ANALYSIS OF NUMERIC VARIABLES BY TARGET
########################################

def target_summary_with_num(dataframe, target, numerical_col):
    """
    Display the mean of a numerical variable grouped by the target variable in a DataFrame.

    Parameters
    ----------
    dataframe (DataFrame): The DataFrame containing the data.
    target (str): The name of the target variable.
    numerical_col (str): The name of the numerical column to be analyzed.

    """
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n#################################\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


#############################
#ANALYSIS OF MISSING VALUES
#############################

# Check if there are any missing values in the DataFrame
df.isnull().values.any()
# Returns False, there are no missing values in the data set.

# Get the count of missing values for each column
df.isnull().sum()

# Get the count of non-missing (not null) values for each column
df.notnull().sum()

# Get the total count of missing values in the entire DataFrame
df.isnull().sum().sum()

# Filter rows containing at least one missing value (NaN) in any column
var = df[df.isnull().any(axis=1)]

# Filter rows containing at least one non-missing (not null) value in any column
var1 = df[df.notnull().any(axis=1)]



###################
# CORRELATION
##################

# Correlation indicates the direction and strength of the linear relationship
# between two random variables in probability theory and statistics.

df.corr()

#Correlation Matrix

# Create a heatmap of the correlation matrix
f, ax = plt.subplots(figsize=[18, 13])
# Use the `sns.heatmap` function to display the correlation matrix with annotations
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
# Set the title of the heatmap
ax.set_title("Correlation Matrix", fontsize=20)
# Display the plot
plt.show(block=True)


##################
#BASE MODEL SETUP
##################

# Split the data into independent variables(X) and target variable(y).
y = df["Outcome"]
X = df.drop("Outcome", axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

# Create a Random Forest Classifier model and fit it to the training data
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")


# Accuracy: Proportion of correct predictions (TP+TN) / (TP+TN+FP+FN)
# Recall: Proportion of true positive predictions for the positive class TP / (TP+FN)
# Precision: Proportion of true positive predictions among all positive predictions TP / (TP+FP)
# F1 Score: A balance between precision and recall, calculated as 2 * (Precision * Recall) / (Precision + Recall)
# AUC (Area Under the Curve): Measures the classifier's ability to distinguish between positive and negative classes.

########################################
#(df)
#Accuracy: 0.77
#Recall: 0.706
#Precision: 0.59
#F1: 0.64
#Auc: 0.75

#after removing outliers part (new_df):
#Accuracy: 0.8
#Recall: 0.73
#Precision: 0.62
#F1: 0.67
#Auc: 0.78
########################################

def plot_importance(model, features, num=len(X), save=False):
    """
    Plot feature importance for a machine learning model.

    Parameters:
    -----------
    model : object
        The trained machine learning model for which feature importance is to be visualized.
    features : pd.DataFrame
        The DataFrame containing the features used for modeling.
    num : int, optional
        The number of top important features to display. Default is the total number of features in 'X'.
    save : bool, optional
        Whether to save the plot as 'importances.png'. Default is False.

    """
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)



###########################################
# TASK 2: FEATURE ENGINEERING
##########################################


##################################
# ANALYSIS OF MISSING VALUES
#################################

# Check if there are any missing values in the DataFrame
df.isnull().values.any()
# Get the count of missing values for each column
df.isnull().sum()
# Get the count of non-missing (not null) values for each column
df.notnull().sum()
# Get the total count of missing values in the entire DataFrame
df.isnull().sum().sum()
# Filter rows containing at least one missing value (NaN) in any column
var = df[df.isnull().any(axis=1)]
# Filter rows containing at least one non-missing (not null) value in any column
var1 = df[df.notnull().any(axis=1)]


# In a human, features cannot be zero other than 'Pregnancies' and 'Outcome',
# therefore NaN should be entered for the values entered as min value zero.

zero_cols = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
for col in zero_cols:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

df.isnull().sum()

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

def missing_values_table(dataframe, na_name=False):
    """

    Generate a summary of missing values in a DataFrame and optionally return
    column names with missing values.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame to analyze for missing values.
    na_name : bool, optional
        If True, return a list of column names with missing values. Default is False.

    Returns
    -------
    None or list of str
        If na_name is False (default), no return value.
        If na_name is True, a list of column names with missing value.

    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe.isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
missing_values_table(df)


na_columns = missing_values_table(df, na_name=True)

# Examining the Relationship Between Missing Values and the Dependent Variable
def missing_vs_target(dataframe, target, na_columns):
    """
    Calculate the mean of the 'target' variable grouped by the presence
     or absence of missing values in the specified columns.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The input DataFrame containing the data.
    target : str
        The name of the target variable for which the mean is calculated.
    na_columns : list of str
        A list of column names to analyze for missing values.

    Prints:
    -------
    For each column in 'na_columns', this function prints a DataFrame containing:
        - 'TARGET_MEAN': The mean of the 'target' variable for rows with missing and non-missing values in the column.
        - 'Count': The count of rows with missing and non-missing values in the column.
    """
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_columns)

# Filling Missing Values
for col in zero_cols:
    df.loc[df[col].isnull(), col] = df[col].median()

df.isnull().sum()


###############################
# ANALYSIS OF OUTLIER
###############################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """

    Calculate the lower and upper outlier detection thresholds for a numerical column in a DataFrame.

    Parameters
    ----------
    dataframe (DataFrame): The DataFrame containing the data.
    col_name (str): The name of the numerical column for which outlier thresholds are calculated.
    q1 (float, optional): The lower quartile (default is 0.25).
    q3 (float, optional): The upper quartile (default is 0.75).

    Returns
    -------
    low_limit (float, optional): The lower threshold for detecting outliers.
    up_limit (float, optional): The upper threshold for detecting outliers.

    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

for col in num_cols:
    print(col, outlier_thresholds(df, col))


def check_outlier(dataframe, col_name):
    """

    Check for outliers in a specified numerical column of a DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the data.
    col_name : str
        The name of the numerical column to check for outliers.

    Returns
    -------
    bool
        True if outliers are found, False otherwise.

    Notes
    -----
    This function checks for outliers in the specified numerical column of the DataFrame by comparing the values
    to the lower and upper outlier thresholds. If any values in the column fall outside of these thresholds,
    the function returns True, indicating the presence of outliers. Otherwise, it returns False.

    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    """

    Replace outliers in a DataFrame column with specified lower and upper limits.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the data.
    variable : str
        The name of the column in the DataFrame where outliers should be replaced.
    q1 : float, optional
        The lower quantile used to calculate the lower threshold. Default is 0.05.
    q3 : float, optional
        The upper quantile used to calculate the upper threshold. Default is 0.95.

    Returns
    -------
    None
        This function modifies the input DataFrame in place.

    Notes
    -----
    Outliers are replaced with the calculated lower and upper limits as follows:
    - Values less than the lower limit are set to the lower limit.
    - Values greater than the upper limit are set to the upper limit.

    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in df.columns:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in df.columns:
    print(col, check_outlier(df, col))



def remove_outlier(dataframe, col_name):
    """
    Remove outliers from a specified numerical column of a DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the data.
    col_name : str
        The name of the numerical column from which outliers should be removed.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the outliers from the specified column removed.

    Notes
    -----
    This function removes rows containing outliers in the specified numerical column
    of the DataFrame based on the lower and upper outlier thresholds. The thresholds
    are calculated using the `outlier_thresholds` function.

    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

for col in num_cols:
    new_df = remove_outlier(df, col)

#df.shape[0] - new_df.shape[0]


"""
Accuracy: 0.77
Recall: 0.706
Precision: 0.59
F1: 0.64
Auc: 0.75
--------------------------
after removing outliers part:
Accuracy: 0.8
Recall: 0.73
Precision: 0.62
F1: 0.67
Auc: 0.78
"""

################################
# FEATURE EXTRACTION
################################

# Deriving Variables from Raw Data

# Categorize individuals' ages into "mature" for ages 21 to 49, and "senior" for ages 50 and above.
df.loc[(df["Age"] >= 21) & (df["Age"] <= 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] > 50), "NEW_AGE_CAT"] = "senior"

#BMI below 18.5: Underweight
#BMI between 18.5 and 24.9: Normal
#BMI between 24.9 and 29.9: Overweight
#BMI over 30: Obese
df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=["Underweight", "Healthy", "Overweight", "Obese"])

#Converting Glucose Value into a Categorical Variable
df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])

# Categorizing individuals as "underweight" based on BMI < 18.5, and distinguishing between "mature" (ages 21-49)
# and "senior" (age 50 and above) age groups. Assigning appropriate labels to the "NEW_AGE_BMI_NOM" column.
df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightMature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightSenior"

# Categorizing individuals as "healthy" based on BMI between 18.5 and 25, and distinguishing between "mature" (ages 21-49)
# and "senior" (age 50 and above) age groups. Assigning appropriate labels to the "NEW_AGE_BMI_NOM" column.
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthyMature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthySenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthyMature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthySenior"

# Categorizing individuals as "overweight" based on BMI between 25 and 30, and distinguishing between "mature" (ages 21-49)
# and "senior" (age 50 and above) age groups. Assigning appropriate labels to the "NEW_AGE_BMI_NOM" column.
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightMature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightSenior"

# Categorizing individuals as "obese" based on BMI greater than 18.5, and distinguishing between "mature" (ages 21-49)
# and "senior" (age 50 and above) age groups. Assigning appropriate labels to the "NEW_AGE_BMI_NOM" column.
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obeseMature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obeseSenior"

# Categorizing individuals with "low" Glucose levels (less than 70) and distinguishing between "mature" (ages 21-49)
# and "senior" (age 50 and above) age groups. Assigning appropriate labels to the "NEW_AGE_GLUCOSE_NOM" column.
df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowMature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowSenior"

# Categorizing individuals with "normal" Glucose levels (between 70 and 99) and distinguishing between "mature" (ages 21-49)
# and "senior" (age 50 and above) age groups. Assigning appropriate labels to the "NEW_AGE_GLUCOSE_NOM" column.
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalMature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalSenior"

# Categorizing individuals with "hidden" Glucose levels (between 100 and 125) and distinguishing between "mature" (ages 21-49)
# and "senior" (age 50 and above) age groups. Assigning appropriate labels to the "NEW_AGE_GLUCOSE_NOM" column.
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenMature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddenSenior"

# Categorizing individuals with "high" Glucose levels (above 125) and distinguishing between "mature" (ages 21-49)
# and "senior" (age 50 and above) age groups. Assigning appropriate labels to the "NEW_AGE_GLUCOSE_NOM" column.
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highMature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highSenior"

# Categorize Insulin values and add a new column "NEW_INSULIN_SCORE"
df["NEW_INSULIN_SCORE"] = np.where((df["Insulin"] >= 16) & (df["Insulin"] <= 166), "Normal", "Abnormal")

# Calculate the product of Glucose and Insulin values and add a new column "NEW_GLUCOSE*INSULIN"
df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]

df.columns = [col.upper() for col in df.columns]

df.head()


##################################
# ENCODING
##################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Label Encoding & Binary Encoding
def label_encoder(dataframe, binary_col):
    """
    Encode binary categorical data using LabelEncoder.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The input DataFrame containing the data.
    binary_col : str
        The name of the binary categorical column to be encoded.

    Returns:
    --------
    pd.DataFrame
        The input DataFrame with the specified binary column encoded using LabelEncoder.

    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes not in [int, float] and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

# One-Hot Encoding
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Outcome"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """
    Apply one-hot encoding to categorical columns in the DataFrame.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The input DataFrame containing the data.
    categorical_cols : list of str
        A list of column names to be one-hot encoded.
    drop_first : bool, optional
        Whether to drop the first category in each encoded column to prevent multicollinearity. Default is False.

    Returns:
    --------
    pd.DataFrame
        The input DataFrame with specified categorical columns one-hot encoded.

    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

##################################
# STANDARDIZATION
##################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)
#num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape

##################################
# MODELING
##################################

# Split the data into independent variables(X) and target variable(y).
y = df["Outcome"]
X = df.drop("Outcome", axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

# Create a Random Forest Classifier model and fit it to the training data
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")


# Accuracy: Proportion of correct predictions (TP+TN) / (TP+TN+FP+FN)
# Recall: Proportion of true positive predictions for the positive class TP / (TP+FN)
# Precision: Proportion of true positive predictions among all positive predictions TP / (TP+FP)
# F1 Score: A balance between precision and recall, calculated as 2 * (Precision * Recall) / (Precision + Recall)
# AUC (Area Under the Curve): Measures the classifier's ability to distinguish between positive and negative classes.


##################################
# FEATURE IMPORTANCE
##################################

def plot_importance(model, features, num=len(X), save=False):
    """
    Plot feature importance for a machine learning model.

    Parameters:
    -----------
    model : object
        The trained machine learning model for which feature importance is to be visualized.
    features : pd.DataFrame
        The DataFrame containing the features used for modeling.
    num : int, optional
        The number of top important features to display. Default is the total number of features in 'X'.
    save : bool, optional
        Whether to save the plot as 'importances.png'. Default is False.
    """
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)
