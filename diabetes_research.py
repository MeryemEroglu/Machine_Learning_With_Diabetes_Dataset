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

df = pd.read_csv("diabetes.csv")

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
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

for col in cat_cols:
    cat_summary(df, col)

for col in cat_cols:
    target_summary_with_cat(df, "OUTCOME", col)

cat_cols = [col for col in cat_cols if "OUTCOME" not in col]

df = one_hot_encoder(df, cat_cols, drop_first=True)

check_df(df)
df.columns = [col.upper() for col in df.columns]

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
cat_cols = [col for col in cat_cols if "OUTCOME" not in col]

for col in num_cols:
    print(col, check_outlier(df, col, 0.05, 0.95))

replace_with_thresholds(df, "INSULIN")

X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)
check_df(X)

def diabetes_data_prep(dataframe):
    dataframe.columns = [col.upper() for col in dataframe.columns]

    dataframe['NEW_GLUCOSE_CAT'] = pd.cut(x=dataframe['GLUCOSE'], bins=[-1, 139, 200], labels=["normal", "prediabetes"])

    dataframe.loc[(dataframe['AGE'] < 35), "NEW_AGE_CAT"] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 35) & (dataframe['AGE'] <= 55), "NEW_AGE_CAT"] = 'middleage'
    dataframe.loc[(dataframe['AGE'] > 55), "NEW_AGE_CAT"] = 'old'

    dataframe['NEW_BMI_RANGE'] = pd.cut(x=dataframe['BMI'], bins=[-1, 18.5, 24.9, 29.9, 100],
                                        labels=["underweight", "healthy", "overweight", "obese"])

    dataframe['NEW_BLOOD_PRESSURE'] = pd.cut(x=dataframe['BLOODPRESSURE'], bins=[-1, 79, 89, 123],
                                           labels=["normal", "hypertension_stages_1", "hypertension_stages_2"])
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)
    cat_cols = [col for col in cat_cols if "OUTCOME" not in col]
    df = one_hot_encoder(dataframe, cat_cols, drop_first=True)
    df.columns = [col.upper() for col in df.columns]
    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
    cat_cols = [col for col in cat_cols if "OUTCOME" not in col]
    replace_with_thresholds(df, "INSULIN")
    X_scaled = StandardScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)
    y = df["OUTCOME"]
    X = df.drop(["OUTCOME"], axis=1)
    return X, y

check_df(df)

X, y = diabetes_data_prep(df)

check_df(X)

####################################
# 3. Base Models
####################################

def base_models(X, y, scoring="roc_auc"):
    print("Base Models...")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ('SVC', SVC()),
                   ('CART', DecisionTreeClassifier()),
                   ('RF', RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   ('CatBoost', CatBoostClassifier(verbose=False))]
    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}")
base_models(X, y, scoring="accuracy")

############################################
# 4. Automated Hyperparameter Optimization
############################################

# Define hyperparameter grids for each classifier
knn_parameters = {"n_neighbors": range(2, 50)}

cart_parameters = {"max_depth": range(1, 20),
                   "min_samples_split": range(2, 30)}

rf_parameters = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_parameters = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_parameters = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}

# Define a list of classifiers with their respective hyperparameter grids
classifiers = [('KNN', KNeighborsClassifier(), knn_parameters),
               ('CART', DecisionTreeClassifier(), cart_parameters),
               ('RF', RandomForestClassifier(), rf_parameters),
               ('XGBOOST', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_parameters),
               ('LightGBM', LGBMClassifier(), lightgbm_parameters)]

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    """

    Perform hyperparameter optimization for a list of classifiers.
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        The input data.

    y : array-like of shape (n_samples,)
        The target labels.

    cv : int, default=3
        Number of folds for cross-validation.

    scoring : str, default="roc_auc"
        The scoring metric for evaluation.

    Returns:
    --------
    best_models : dict
        A dictionary containing the best models for each classifier.

    """
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"~~~~~~~~~~~~~~~{name}~~~~~~~~~~~~~~~~~~")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")
        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)
        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)

#############################################
# 5. Stacking & Ensemble Learning
############################################

def voting_classifier(best_models, X, y):
    """
    Creates a voting classifier using the best models obtained from hyperparameter optimization.

    Parameters:
    -----------
    best_models : dict
        A dictionary containing the best models for each classifier.

    X : array-like of shape (n_samples, n_features)
        The input data.

    y : array-like of shape (n_samples,)
        The target labels.

    Returns:
    --------
    voting_clf : VotingClassifier
        A voting classifier fitted on the input data.

    """
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)

######################################################
# 6. Prediction for a New Observation
######################################################

X.columns
random_user = X.sample(1, random_state=45)
voting_clf.predict(random_user)

joblib.dump(voting_clf, "voting_clf2.pkl")

new_model = joblib.load("voting_clf2.pkl")
new_model.predict(random_user)