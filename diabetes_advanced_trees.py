####################################################
# Random Forests, GBM, XGBoost, LightGBM, CatBoost
####################################################


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv('diabetes/diabetes.csv')
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)


################################################
# Random Forests
################################################
# Create a Random Forest Classifier model with a specified random seed (random_state).
rf_model = RandomForestClassifier(random_state=17)

# Retrieve the model's parameters to inspect their current values.
rf_model.get_params()

# Perform cross-validation with the Random Forest model, evaluating it with accuracy, F1 score, and ROC AUC scoring metrics.
cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

#Calculate the mean accuracy score, mean F1 score, and mean ROC AUC score from the cross-validation results using the final model.
cv_results['test_accuracy'].mean()
# 0.753896103896104
cv_results['test_f1'].mean()
#0.6190701534636385
cv_results['test_roc_auc'].mean()
#0.8233960113960114

# Define a dictionary of Random Forest parameters to be used in a grid search.
rf_params = {"max_depth": [4, 5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 300]}

# Perform a grid search (GridSearchCV) with the Random Forest model to find the best combination of parameters using 5-fold cross-validation.
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Get the best parameter values from the grid search.
rf_best_grid.best_params_

# Create a final Random Forest model using the best parameters obtained from the grid search and set the random seed.
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

# Perform cross-validation with the final Random Forest model using accuracy, F1 score, and ROC AUC scoring metrics.
cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

# Calculate the mean accuracy score, mean f1 score, mean roc auc score from the cross-validation results using the final model.
cv_results['test_accuracy'].mean()
# 0.7669172932330828
cv_results['test_f1'].mean()
# 0.6287772622372885
cv_results['test_roc_auc'].mean()
# 0.8334928774928775

def plot_importance(model, features, num=len(X), save=False):
    """
    Plot feature importances for a given model.

    Parameters
    ----------
    model : object
        The trained machine learning model.
    features : DataFrame
        The DataFrame containing the feature names.
    num : int, optional
        Number of top features to display (default is all features).
    save : bool, optional
        If True, save the plot as 'importances.png' (default is False).
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

plot_importance(rf_final, X)

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    """
    Plot a validation curve for a given model and hyperparameter.

    Parameters
    ----------
    model : object
        The machine learning model.
    X : array-like or pd.DataFrame
        The feature matrix.
    y : array-like
        The target values.
    param_name : str
        The name of the hyperparameter to tune.
    param_range : array-like
        The range of hyperparameter values to test.
    scoring : str, optional
        The scoring metric to evaluate (default is "roc_auc").
    cv : int, optional
        The number of cross-validation folds (default is 10).

    """
    train_score, test_score = validation_curve(model, X=X, y=y,
                                               param_name=param_name,
                                               param_range=param_range,
                                               scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score, label="Training Score", color='b')
    plt.plot(param_range, mean_test_score, label="Test Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")



################################################
# GBM
################################################
# Create a GradientBoostingClassifier model with a random state
gbm_model = GradientBoostingClassifier(random_state=17)

# Retrieve the model's parameters to inspect their current values.
gbm_model.get_params()

# Perform cross-validation with the Gradient Boosting Machine model, evaluating it with accuracy, F1 score, and ROC AUC scoring metrics.
cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.7591715474068416
cv_results['test_f1'].mean()
#0.634235802826363
cv_results['test_roc_auc'].mean()
#0.8254867225716283

# Define a dictionary of Gradient Boosting Machine model parameters to be used in a grid search.
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

# Perform a grid search (GridSearchCV) with the Gradient Boosting Machine model to find the best combination of parameters using 5-fold cross-validation.
gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)

cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


################################################
# XGBoost
################################################
xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
xgboost_model.get_params()
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7800186741363212
cv_results['test_f1'].mean()
# 0.668605747317776
cv_results['test_roc_auc'].mean()
# 0.8257784765897973

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}


xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)


cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)
cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7474492827434004
cv_results['test_f1'].mean()
# 0.624110522144179
cv_results['test_roc_auc'].mean()
# 0.7990293501048218

lgbm_model.get_params()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

#colsample_bytree=1, learning_rate=0.01, n_estimators=300, random_state=17

lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
               "n_estimators": [200, 250, 300, 350, 400],
               "colsample_bytree": [0.9, 0.8, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)
#colsample_bytree=0.9, learning_rate=0.01, n_estimators=200, random_state=17

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7643833290892115
cv_results['test_f1'].mean()
# 0.6193071162618689
cv_results['test_roc_auc'].mean()
# 0.8227931516422082

#Hyperparameter optimization just for n_estimators
lgbm_model = LGBMClassifier(colsample_bytree=0.9, learning_rate=0.01, random_state=17)
lgbm_params = {"n_estimators": [200, 400, 1000, 5000, 8000, 9000, 10000]}
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

################################################
# CatBoost
################################################

catboost_model = CatBoostClassifier(random_state=17, verbose=False)
cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7735251676428148
cv_results['test_f1'].mean()
# 0.6502723851348231
cv_results['test_roc_auc'].mean()
# 0.8378923829489867

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7721755368814192
cv_results['test_f1'].mean()
# 0.6322580676028952
cv_results['test_roc_auc'].mean()
# 0.842001397624039

################################################
# Feature Importance
################################################
def plot_importance(model, features, num=len(X), save=False):
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

plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)