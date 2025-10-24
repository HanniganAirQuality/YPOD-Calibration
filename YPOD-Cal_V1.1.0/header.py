### importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import importlib
os.chdir("/Users/Chiara/Documents/Documents - Chiara’s MacBook Air/Research/YPOD Calibration/Model Improvement")
import config
importlib.reload(config)

def setup_plots(pollutants): # sets up figure for pod visualization
    figures, axes = {}, {}
    with plt.style.context("ggplot"): # plt style
        for pollutant in pollutants:
            fig, ax = plt.subplots()
            figures[pollutant] = fig
            axes[pollutant] = ax
    return figures, axes

def process_pod_data(pod):
    filename = f"YPOD{pod}{config.filename_date}" # filename, needs to be changed 
    df = pd.read_csv(filename)
    df.columns = config.pod_headers # sets columns
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format = "mixed") # datetime format for timestamps
    df = df[(df["Timestamp"] >= config.start_cutoff) & (df["Timestamp"] <= config.end_cutoff)] # crops data
    pod_name = df["PodID"].iloc[0] # finds pod ID, temp variable
    df.drop(columns = ["PodID"], inplace = True) # drops column for numeric data
    df.set_index("Timestamp", inplace = True) # sets indexer to timestamp
    df = df.resample(config.sample_time).median().reset_index() # resamples to sample time with median
    df["PodID"] = pod_name # adds podid to df 
    df.drop(columns = config.pod_col_drop, inplace = True) # drops columns not needed
    return df

def process_reference_file(filename, header, drop_rows, drop_cols, numeric_cols):
    df = pd.read_csv(filename)
    df.columns = header # heads file
    df.drop(df.head(drop_rows).index, inplace = True) # drops rows not needed
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format = "mixed") # datetime format for timestamps
    df = df[(df["Timestamp"] >= config.start_cutoff) & (df["Timestamp"] < config.end_cutoff)] # crops data
    df.drop(columns = drop_cols, inplace = True) # drops columns not needed
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors = "coerce") # converts to numeric
        df.loc[df[col] == 99999, col] = np.nan # sets nan values
        df.loc[df[col] < 0, col] = 0 # data filtering less than zero
    df.set_index("Timestamp", inplace = True) # sets indexer to timestamp
    df = df.resample(config.sample_time).median().reset_index() # resamples to sample time with median
    df.fillna(df.mean(numeric_only = True), inplace = True) # converts to numeric, fills data
    return df

def impute_numeric_data(df):
    numeric = df.select_dtypes(include = [np.number]) # numeric data
    imputer = SimpleImputer(strategy = "median") # median imputing
    imputed_array = imputer.fit_transform(numeric) # transforms to arr
    return pd.DataFrame(imputed_array, columns = numeric.columns)

def select_model(X, y):
    if config.linear_regression_mvlr:
        return LinearRegression()
    if config.ridge_mvlr:
        return Ridge(alpha = 1, random_state = 10)
    if config.lasso_mvlr:
        return Lasso(alpha = 1, random_state = 10)
    if config.gradient_boost_model:
        return XGBRegressor(objective = 'reg:squarederror', n_estimators = 100, random_state = 10)
    if config.random_forest_model:
        return RandomForestRegressor(n_estimators = 100, random_state = 10)
    print("No regression model chosen, please set a model = True in the 'variables to change' section.")
    return None

def print_model_summary(model, variables, pod_id_index, model_name, r2, y_test, y_pred, Type = None):
    print(f"Pod {pod_id_index} {model_name} {Type} model:\n")
    if hasattr(model, "coef_"):
        print("Coefficients:")
        for var, coef in zip(variables, model.coef_.flatten()):
            print(f"{var}: {coef:.5f}") # prints coefficients
        print(f"Intercept: {model.intercept_.item():.5f}\n")
    elif hasattr(model, "feature_importances_"):
        print("Feature Importances:")
        for var, importance in zip(variables, model.feature_importances_):
            print(f"{var}: {importance:.5f}") # prints feature importances 
    else:
        print("Model does not provide feature information")
    print(f"\nR-squared: {r2:.5f}") # r squared
    print(f"Mean absolute error: {metrics.mean_absolute_error(y_test, y_pred):.5f}") # absolute error
    print(f"Mean squared error: {metrics.mean_squared_error(y_test, y_pred):.5f}") # squared error
    print(f"Root mean squared error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.5f}\n") # root mean squared error

def plot_accuracy(y_test, y_pred, model_name, units, pod_id_index, Type = None):
    plt.style.use("ggplot")
    plt.scatter(y_test, y_pred, alpha = 0.6, s = 40, edgecolors = "black", linewidth = 1, c = "deepskyblue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color = "orangered", linewidth = 2)
    min_val = min(y_test.min(), y_pred.min()) # minimum for line
    max_val = max(y_test.max(), y_pred.max()) # maximum for line
    pad = (max_val - min_val) * 0.05
    plt.xlim(min_val - pad, max_val + pad) # sizing
    plt.ylim(min_val - pad, max_val + pad) # sizing
    plt.xlabel(f"Actual {model_name} {units}", fontweight = "bold", fontsize = 10)
    plt.ylabel(f"Predicted {model_name} {units}", fontweight = "bold", fontsize = 10)
    plt.title(f"Actual vs Predicted {model_name} Values for Pod {pod_id_index} ({Type} Data)", fontweight = "bold", fontsize = 10)
    plt.tight_layout()
    if config.save_plots: # save plots
        filename = f"{pod_id_index}_{model_name.replace(' ', '_')}_model_accuracy.png" # export filename
        path = os.path.join(config.save_dir, filename) # join path
        plt.savefig(path, dpi = 300)
        print(f"Saved: {path}")
    plt.show()

def plot_timeseries(X_, y_, y_pred, pod, model_name, units, pod_id_index, Type = None):
    plt.style.use("ggplot")
    fig, ax1 = plt.subplots()
    X_ = X_.copy()
    X_["Timestamp"] = pod["Timestamp"].iloc[X_.index] # finds timestamp
    X_["Model Prediction"] = y_pred.flatten() # model prediciton
    X_["Reference Instrument Data"] = y_.values.flatten() # data handling
    X_ = X_.sort_values("Timestamp") # sorts
    ax1.plot(X_["Timestamp"], X_["Reference Instrument Data"], label = f"Reference Instrument Data {units}", color = "orangered", linewidth = 1)
    ax1.plot(X_["Timestamp"], X_["Model Prediction"], label = "Model Prediction", color = "k", linestyle = "--", linewidth = 1)
    ax1.set_xlabel("Timestamp", fontweight = "bold", fontsize = 10)
    ax1.set_ylabel(f"{model_name} {units}", fontweight = "bold", fontsize = 10)
    ax1.tick_params(axis = "x", labelsize = 7)
    ax1.tick_params(axis = "y", labelsize = 7)
    handles, labels = ax1.get_legend_handles_labels() # handles, labels for plotting
    if config.display_pod_data_timeseries:
        ax2 = ax1.twinx() # dual axis
        ax2.plot(X_["Timestamp"], X_[model_name], label = f"{model_name} Raw Data (ADU)", linewidth = 1, color = "deepskyblue")
        ax2.set_ylabel(f"{model_name} (ADU)", fontweight = "bold", fontsize = 10)
        ax2.tick_params(axis = "y", labelsize = 7)
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2 # adds handles
        labels += l2 # adds labels
    plt.legend(handles, labels, labelcolor = "black")
    plt.title(f"Time Series of Actual vs Predicted {model_name} for Pod {pod_id_index} ({Type} Data)", fontweight = "bold", fontsize = 10)
    plt.tight_layout()
    if config.save_plots: # save plots
        filename = f"{pod_id_index}_{model_name.replace(' ', '_')}_model_timeseries.png" # export filename
        path = os.path.join(config.save_dir, filename) # join path
        plt.savefig(path, dpi = 300)
        print(f"Saved: {path}")
    plt.show()
    
def plot_residuals(X_, y_, y_pred, model_name, pod_id_index, Type = None):
    plt.style.use("ggplot")
    X_ = X_.copy() # copy
    res = y_ - y_pred # residual (actual - predicted)
    plt.scatter(X_.iloc[:, 0], res, alpha = 0.6, color = "deepskyblue", edgecolor = "k", linewidth = 1)
    plt.axhline(y = 0, color = "orangered", linestyle = "--", linewidth = 2)
    plt.xlabel(f"Raw {X_.columns[0]} Signal (ADU)", fontweight = "bold", fontsize = 10)
    plt.ylabel("Residual", fontweight = "bold", fontsize = 10)
    plt.title (f"Residual Plot for {pod_id_index} {model_name} {Type} Data", fontweight = "bold", fontsize = 10)
    plt.tight_layout()
    plt.show()