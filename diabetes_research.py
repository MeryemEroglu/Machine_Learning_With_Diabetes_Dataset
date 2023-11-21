################################################
# End-to-End Diabetes Machine Learning Pipeline I
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function

import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################

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
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):
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
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

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
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    """
        Display the mean of a numerical variable grouped by the target variable in a DataFrame.

        Parameters
        ----------
        dataframe (DataFrame): The DataFrame containing the data.
        target (str): The name of the target variable.
        numerical_col (str): The name of the numerical column to be analyzed.

        """
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    """
        Calculate the mean of the target variable grouped by the specified categorical column.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The input DataFrame containing the data.
        target : str
            The name of the target variable for which the mean will be calculated.
        categorical_col : str
            The name of the categorical column used for grouping.

        """
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(dataframe, cols):
    """
    Generate and display a correlation matrix heatmap for the specified columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input DataFrame containing the data.
    cols : list of str
        The list of column names for which the correlation matrix will be calculated and visualized.
    """
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(dataframe[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

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
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]


    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

check_df(df)
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

# Display categorical column summaries using cat_summary function.
for col in cat_cols:
    cat_summary(df, col)

# Display descriptive statistics for numerical columns.
df[num_cols].describe().T

# Display numerical column summaries using num_summary function with plots.
for col in num_cols:
    num_summary(df, col, plot=True)

# Generate and display a correlation matrix heatmap for numerical columns.
correlation_matrix(df, num_cols)

# Display target variable summaries with numerical columns using target_summary_with_num function.
for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

################################################
# 2. Data Preprocessing & Feature Engineering
################################################

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


def replace_with_thresholds(dataframe, variable):
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
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
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
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

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

df.head()

df.columns = [col.upper() for col in df.columns]

df['NEM_GLUCOSE_CAT'] = pd.cut(x=df['GLUCOSE'], bins=[-1, 139, 200], labels=["normal", "prediabetes"])

df.loc[(df['AGE'] < 35), "NEW_AGE_CAT"] = 'young'
df.loc[(df['AGE'] >= 35) & (df['AGE'] <= 55), "NEW_AGE_CAT"] = 'middle_age'
df.loc[(df['AGE'] > 55), "NEW_AGE_CAT"] = 'old'

df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[-1, 18.5, 24.9, 29.9, 100], labels=["underweight", "healthy", "overweight", "obese"])

df['NEW_BLOOD_PRESSURE'] = pd.cut(x=df['BLOODPRESSURE'], bins=[-1, 79, 89, 123], labels=["normal", "hypertension_stages_1", "hypertension_stages_2"])

check_df(df)
