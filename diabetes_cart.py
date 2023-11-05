################################################
# Decision Tree Classification: CART
################################################


# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling using CART
# 4. Hyperparameter Optimization with GridSearchCV
# 5. Final Model
# 6. Feature Importance
# 7. Analyzing Model Complexity with Learning Curves (BONUS)
# 8. Visualizing the Decision Tree
# 9. Extracting Decision Rules
# 10. Extracting Python/SQL/Excel Codes of Decision Rules
# 11. Prediction using Python Codes
# 12. Saving and Loading Model



import warnings

import joblib
import numpy as np
import pandas as pd
import pydotplus
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from skompiler import skompile

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

################################################
# 1. EXPLORATORY DATA ANALYSIS
################################################

# Step 1: Examine the overall picture.
df = pd.read_csv("diabetes.csv")
def check_df(dataframe, head=5):
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





################################################
# 2. DATA PREPROCESSING & FEATURE ENGINEERING
################################################


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





################################################
# 3. Modeling using CART
################################################

df = pd.read_csv("diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

#y_pred for confusion matrix
y_pred = cart_model.predict(X)

#Y_prob for AUC
y_prob = cart_model.predict_proba(X)[:, 1]

#Confusion Matrix
print(classification_report(y, y_pred))

#AUC
roc_auc_score(y, y_prob)

################################################
# Model validation with Holdout Method
################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)
df.shape
#(768, 9)

X_train.shape
#(537, 8)
X_test.shape
#(231, 8)
y_train.shape
#(537,)
y_test.shape
#(231,)

# Train a Decision Tree model using the training data (X_train, y_train) with a random state of 17.
cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# Evaluate the model's performance on the training set and display the results.
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# Evaluate the model's performance on the test set and display the results.
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_pred)

################################################
# Model validation with Cross Validation
################################################

# Train a Decision Tree model using the entire dataset (X, y) with a fixed random state of 17.
cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)

# Perform cross-validation with k-fold (k=5) and evaluate the model's performance using accuracy, F1 score, and ROC AUC scoring.
cv_results = cross_validate(cart_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

# Calculate the mean accuracy, F1 score, and ROC AUC score from cross-validation results.
cv_results['test_accuracy'].mean()
# 0.70
cv_results['test_f1'].mean()
# 0.57
cv_results['test_roc_auc'].mean()
# 0.67

####################################################
# 4. Hyperparameter Optimization with GridSearchCV
###################################################
# Get the current parameters of the Decision Tree model.
cart_model.get_params()

# Define a parameter grid to search for the best hyperparameters.
cart_params = {'max_depth': range(1, 11), "min_samples_split": range(2, 20)}


# Perform hyperparameter optimization using GridSearchCV with 5-fold cross-validation.
cart_best_grid = GridSearchCV(cart_model, cart_params, cv=5, n_jobs=-1, verbose=1).fit(X, y)


# Perform hyperparameter optimization using GridSearchCV with 5-fold cross-validation.
best_params = cart_best_grid.best_params_

# Display the best mean cross-validated score obtained with the best hyperparameters.
best_score = cart_best_grid.best_score_
# Best Score: 0.75

# Generate a random sample from the input data 'X' for prediction.
random_user = X.sample(1, random_state=45)

# Make predictions on the random sample using the model with optimized hyperparameters.
cart_best_grid.predict(random_user)

# Get the probability of class 1 (positive class) for the random sample.
cart_best_grid.predict_proba(random_user)[:, 1]


################################################
# 5. Final Model
################################################

# Create a new Decision Tree model ('cart_final') with the best hyperparameters found by GridSearchCV.
cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)
# Get the parameters of the 'cart_final' model.
cart_final.get_params()

# Update the 'cart_final' model's hyperparameters with the best parameters found by GridSearchCV and fit the model again.
cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)

# Perform cross-validation on the 'cart_final' model to evaluate its performance.
cv_result = cross_validate(cart_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

# Calculate the mean accuracy, F1 score, and ROC AUC score from cross-validation results.
cv_results['test_accuracy'].mean()
#0.70
cv_results['test_f1'].mean()
#0.57
cv_results['test_roc_auc'].mean()
#0.67

################################################
# 6. Feature Importance
################################################

# Retrieve the feature importances from the 'cart_final' Decision Tree model.
cart_final.feature_importances_
def plot_importance(model, features, num=len(X), save=False):
    """

    Plot feature importances using a bar plot.

    Parameters
    ----------
    model : object
        The machine learning model (e.g., DecisionTreeClassifier) from which you want to extract feature importances.

    features : DataFrame
        The dataset's feature columns for which you want to plot importances.

    num : int, optional
        The number of top features to display on the plot. Default is 5.

    save : bool, optional
        If True, save the plot as 'importances.png' in the current working directory. Default is False.

    """
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(cart_final, X, num=5)

######################################################
# 7. Analyzing Model Complexity with Learning Curves
######################################################

# Calculate training and testing scores for different max_depth values
train_score, test_score = validation_curve(cart_final, X, y,
                                           param_name="max_depth",
                                           param_range=range(1, 11),
                                           scoring="roc_auc",
                                           cv=10)

# Calculate the mean training and testing scores across different max_depth values.
mean_train_score = np.mean(train_score, axis=1)
mean_test_score = np.mean(test_score, axis=1)

# Plot the training and testing scores against max_depth values.
plt.plot(range(1, 11), mean_train_score,
         label="Training Score", color='b')

plt.plot(range(1, 11), mean_test_score,
         label="Validation Score", color='g')

# Add titles and labels to the plot.
plt.title("Validation Curve for CART")
plt.xlabel("Number of max_depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc='best')
plt.show()



def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    """
Generate a validation curve to visualize how a model's performance changes with different hyperparameter values.

    Parameters:
    -----------
    model: Estimator object
        The machine learning model for which the validation curve is created.

    X: array-like or pd.DataFrame
        The feature dataset.

    y: array-like or pd.Series
        The target variable dataset.

    param_name: str
        The name of the hyperparameter to be tuned (e.g., "max_depth").

    param_range: iterable
        A range of hyperparameter values to be explored.

    scoring: str, optional
        The scoring metric used for evaluation (default is "roc_auc").

    cv: int, optional
        The number of cross-validation folds (default is 10).

    """
    train_score, test_score = validation_curve(model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


val_curve_params(cart_final, X, y, "max_depth", range(1, 11), scoring="f1")
# List of hyperparameters and their ranges
cart_val_params = [["max_depth", range(1, 11)], ["min_samples_split", range(2, 20)]]

# Generate validation curves for each hyperparameter
for i in range(len(cart_val_params)):
    val_curve_params(cart_model, X, y, cart_val_params[i][0], cart_val_params[i][1])


################################################
# 8. Visualizing the Decision Tree
################################################

def tree_graph(model, col_names, file_name):
    """

    Generate a graphical representation of the decision tree model and save it as a PNG file.

    Parameters:
    -----------
    model: Estimator object
        The decision tree model to be visualized.

    col_names: list
        A list of feature (column) names used in the model.

    file_name: str
        The name of the PNG file to save the visualization.

    """
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)

# Visualize the decision tree model 'cart_final' and save it as a PNG file.
tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")

# Retrieve and display the parameters of the 'cart_final' model.
cart_final.get_params()

################################################
# 9. Extracting Decision Rules
################################################

# Export the decision rules from the 'cart_final' model and print them as text.
tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)


################################################
# 10. Extracting Python Codes of Decision Rules
################################################
# sklearn '0.23.1' version is more optimum.

# Transform the prediction operation into Python code.
print(skompile(cart_final.predict).to('python/code'))

# Transform the prediction operation into SQLAlchemy/SQLite query.
print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))

# Transform the prediction operation into Excel code.
print(skompile(cart_final.predict).to('excel'))


################################################
# 11. Prediction using Python Codes
################################################

def predict_with_rules(x):
    """
    Make binary predictions based on a set of nested conditions.

    Parameters
    ----------
    x : list
        A list representing the input feature vector used for making predictions.

    Returns
    -------
    int
        The binary prediction (0 or 1) based on the input conditions.

    """
    return ((((((0 if x[6] <= 0.671999990940094 else 1 if x[6] <= 0.6864999830722809 else
        0) if x[0] <= 7.5 else 1) if x[5] <= 30.949999809265137 else ((1 if x[5
        ] <= 32.45000076293945 else 1 if x[3] <= 10.5 else 0) if x[2] <= 53.0 else
        ((0 if x[1] <= 111.5 else 0 if x[2] <= 72.0 else 1 if x[3] <= 31.0 else
        0) if x[2] <= 82.5 else 1) if x[4] <= 36.5 else 0) if x[6] <=
        0.5005000084638596 else (0 if x[1] <= 88.5 else (((0 if x[0] <= 1.0 else
        1) if x[1] <= 98.5 else 1) if x[6] <= 0.9269999861717224 else 0) if x[1
        ] <= 116.0 else 0 if x[4] <= 166.0 else 1) if x[2] <= 69.0 else ((0 if
        x[2] <= 79.0 else 0 if x[1] <= 104.5 else 1) if x[3] <= 5.5 else 0) if
        x[6] <= 1.098000019788742 else 1) if x[5] <= 45.39999961853027 else 0 if
        x[7] <= 22.5 else 1) if x[7] <= 28.5 else (1 if x[5] <=
        9.649999618530273 else 0) if x[5] <= 26.350000381469727 else (1 if x[1] <=
        28.5 else ((0 if x[0] <= 11.5 else 1 if x[5] <= 31.25 else 0) if x[1] <=
        94.5 else (1 if x[5] <= 36.19999885559082 else 0) if x[1] <= 97.5 else
        0) if x[6] <= 0.7960000038146973 else 0 if x[0] <= 3.0 else (1 if x[6] <=
        0.9614999890327454 else 0) if x[3] <= 20.0 else 1) if x[1] <= 99.5 else
        ((1 if x[5] <= 27.649999618530273 else 0 if x[0] <= 5.5 else (((1 if x[
        0] <= 7.0 else 0) if x[1] <= 103.5 else 0) if x[1] <= 118.5 else 1) if
        x[0] <= 9.0 else 0) if x[6] <= 0.19999999552965164 else ((0 if x[5] <=
        36.14999961853027 else 1) if x[1] <= 113.0 else 1) if x[0] <= 1.5 else
        (1 if x[6] <= 0.3620000034570694 else 1 if x[5] <= 30.050000190734863 else
        0) if x[2] <= 67.0 else (((0 if x[6] <= 0.2524999976158142 else 1) if x
        [1] <= 120.0 else 1 if x[6] <= 0.23899999260902405 else 1 if x[7] <=
        30.5 else 0) if x[2] <= 83.0 else 0) if x[5] <= 34.45000076293945 else
        1 if x[1] <= 101.0 else 0 if x[5] <= 43.10000038146973 else 1) if x[6] <=
        0.5609999895095825 else ((0 if x[7] <= 34.5 else 1 if x[5] <=
        33.14999961853027 else 0) if x[4] <= 120.5 else (1 if x[3] <= 47.5 else
        0) if x[4] <= 225.0 else 0) if x[0] <= 6.5 else 1) if x[1] <= 127.5 else
        (((((1 if x[1] <= 129.5 else ((1 if x[6] <= 0.5444999933242798 else 0) if
        x[2] <= 56.0 else 0) if x[2] <= 71.0 else 1) if x[2] <= 73.0 else 0) if
        x[5] <= 28.149999618530273 else (1 if x[1] <= 135.0 else 0) if x[3] <=
        21.0 else 1) if x[4] <= 132.5 else 0) if x[1] <= 145.5 else 0 if x[7] <=
        25.5 else ((0 if x[1] <= 151.0 else 1) if x[5] <= 27.09999942779541 else
        ((1 if x[0] <= 6.5 else 0) if x[6] <= 0.3974999934434891 else 0) if x[2
        ] <= 82.0 else 0) if x[7] <= 61.0 else 0) if x[5] <= 29.949999809265137
         else ((1 if x[2] <= 61.0 else (((((0 if x[6] <= 0.18299999833106995 else
        1) if x[0] <= 0.5 else 1 if x[5] <= 32.45000076293945 else 0) if x[2] <=
        73.0 else 0) if x[0] <= 4.5 else 1 if x[6] <= 0.6169999837875366 else 0
        ) if x[6] <= 1.1414999961853027 else 1) if x[5] <= 41.79999923706055 else
        1 if x[6] <= 0.37299999594688416 else 1 if x[1] <= 142.5 else 0) if x[7
        ] <= 30.5 else (((1 if x[6] <= 0.13649999350309372 else 0 if x[5] <=
        32.45000076293945 else 1 if x[5] <= 33.05000114440918 else (0 if x[6] <=
        0.25599999725818634 else (0 if x[1] <= 130.5 else 1) if x[0] <= 8.5 else
        0) if x[0] <= 13.5 else 1) if x[2] <= 92.0 else 1) if x[5] <=
        45.54999923706055 else 1) if x[6] <= 0.4294999986886978 else (1 if x[5] <=
        40.05000114440918 else 0 if x[5] <= 40.89999961853027 else 1) if x[4] <=
        333.5 else 1 if x[2] <= 64.0 else 0) if x[1] <= 157.5 else ((((1 if x[7
        ] <= 25.5 else 0 if x[4] <= 87.5 else 1 if x[5] <= 45.60000038146973 else
        0) if x[7] <= 37.5 else 1 if x[7] <= 56.5 else 0 if x[6] <=
        0.22100000083446503 else 1) if x[6] <= 0.28849999606609344 else 0) if x
        [6] <= 0.3004999905824661 else 1 if x[7] <= 44.0 else (0 if x[7] <=
        51.0 else 1 if x[6] <= 1.1565000414848328 else 0) if x[0] <= 6.5 else 1
        ) if x[4] <= 629.5 else 1 if x[6] <= 0.4124999940395355 else 0)

X.columns

x = [12, 13, 20, 23, 4, 55, 12, 7]

predict_with_rules(x)

x = [6, 148, 70, 35, 0, 30, 0.62, 50]

predict_with_rules(x)



################################################
# 12. Saving and Loading Model
################################################
# Serialize the trained decision tree model 'cart_final' using joblib and save it to a file.
joblib.dump(cart_final, "cart_final.pkl")

# Load the serialized model 'cart_final' from the file.
cart_model_from_disc = joblib.load("cart_final.pkl")

# Define an input feature vector 'x' for making predictions.
x = [12, 13, 20, 23, 4, 55, 12, 7]

# Make predictions using the deserialized model 'cart_model_from_disc' on the input data.
cart_model_from_disc.predict(pd.DataFrame(x).T)

