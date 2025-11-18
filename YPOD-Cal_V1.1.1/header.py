### importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import os
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.stats import zscore
import importlib
os.chdir("/Users/Chiara/Documents/Documents_Chiara_MacBook_Air/Research/YPOD Calibration/Model Improvement")
import config
importlib.reload(config)

def setup_plots(pollutants): # sets up figure for pod visualization
    figures, axes = {}, {}
    with plt.style.context(config.plt_style): # plt style
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

def impute_numeric_data(df): # AKA my nemisis
    original_index = df.index.copy() # housekeeping
    numeric = df.select_dtypes(include = [np.number]) # selects the numeric data
    numeric = numeric.dropna(axis = 1, how = "all") # drops Nan values
    imputer = SimpleImputer(strategy = "median") # imputes along the median of datapoints
    imputed_array = imputer.fit_transform(numeric) # fits imputer to old data 
    numeric_cols = list(numeric.columns) # tags numeric columns
    if imputed_array.shape[1] != len(numeric_cols):
        raise ValueError(f"Imputation column mismatch: array has {imputed_array.shape[1]}, \
                         "f"numeric has {len(numeric_cols)}, columns = {numeric_cols}")
    imputed_df = pd.DataFrame(imputed_array, index = original_index, columns = numeric_cols) # creates the imputed df
    non_numeric = df.select_dtypes(exclude = [np.number]) # bookeeps non-numeric columns i.e timestamp
    non_numeric = non_numeric.loc[:, ~non_numeric.columns.isin(numeric_cols)] # avoids duplicate column names ~ negates
    return pd.concat([imputed_df, non_numeric], axis = 1) # concatenates


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
    print("_______________________________")
    print(f"Pod {pod_id_index} {model_name} {Type} model:")
    print("_______________________________")
    if hasattr(model, "coef_") and config.print_coefficients == True:
        print("Coefficients:")
        for var, coef in zip(variables, model.coef_.flatten()):
            print(f"{var}: {coef:.5f}") # prints coefficients
        print(f"Intercept: {model.intercept_.item():.5f}\n")
    elif hasattr(model, "feature_importances_"):
        print("Feature Importances:")
        for var, importance in zip(variables, model.feature_importances_):
            print(f"{var}: {importance:.5f}\n") # prints feature importances 
    print(f"R-squared: {r2:.5f}") # r squared
    print(f"Mean absolute error: {metrics.mean_absolute_error(y_test, y_pred):.5f}") # absolute error
    print(f"Mean squared error: {metrics.mean_squared_error(y_test, y_pred):.5f}") # squared error
    print(f"Root mean squared error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.5f}\n") # root mean squared error

def plot_accuracy(y_test, y_pred, model_name, units, pod_id_index, Type = None):
    plt.style.use(config.plt_style)
    plt.figure(figsize = (5, 4))
    plt.scatter(y_test, y_pred, alpha = 0.6, s = 40, edgecolors = "white", linewidth = 1, c = "orangered")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color = "k", linewidth = 1)
    min_val = min(y_test.min(), y_pred.min()) # minimum for line
    max_val = max(y_test.max(), y_pred.max()) # maximum for line
    pad = (max_val - min_val) * 0.05
    plt.xlim(min_val - pad, max_val + pad) # sizing
    plt.ylim(min_val - pad, max_val + pad) # sizing
    plt.xlabel(f"Actual {model_name} {units}", fontweight = "bold", fontsize = 10)
    plt.ylabel(f"Predicted {model_name} {units}", fontweight = "bold", fontsize = 10)
    plt.title(f"Actual vs Predicted {model_name} Values for Pod {pod_id_index} ({Type} Data)", fontweight = "bold", fontsize = 8)
    plt.tight_layout()
    if config.save_plots: # save plots
        filename = f"{pod_id_index}_{model_name.replace(' ', '_')}_model_accuracy.png" # export filename
        path = os.path.join(config.save_dir, filename) # join path
        plt.savefig(path, dpi = 300)
        print(f"Saved: {path}")
    plt.show()

def plot_timeseries(X_, y_, y_pred, pod, model_name, units, pod_id_index, Type = None):
    plt.style.use(config.plt_style)
    fig, ax1 = plt.subplots(figsize = (9, 4))
    X_ = X_.copy()
    X_["Timestamp"] = pod["Timestamp"].iloc[X_.index] # finds timestamp
    X_["Model Prediction"] = y_pred.flatten() # model prediciton
    X_["Reference Data"] = y_.values.flatten() # data handling
    X_ = X_.sort_values("Timestamp") # sorts
    ax1.plot(X_["Timestamp"], X_["Reference Data"], "-", label = f"Reference Data {units}", color = "k", linewidth = 1, markersize = 2)
    ax1.plot(X_["Timestamp"], X_["Model Prediction"], "-", label = "Model Prediction", color = "orangered", linewidth = 1)
    ax1.set_xlabel("Timestamp", fontweight = "bold", fontsize = 10)
    ax1.set_ylabel(f"{model_name} {units}", fontweight = "bold", fontsize = 10)
    ax1.tick_params(axis = "x", labelsize = 7, rotation = 45)
    ax1.tick_params(axis = "y", labelsize = 7)
    handles, labels = ax1.get_legend_handles_labels() # handles, labels for plotting
    if config.display_pod_data_timeseries:
        ax2 = ax1.twinx() # dual axis
        ax2.plot(X_["Timestamp"], X_[model_name], "-", label = f"{model_name} Raw Data (ADU)", linewidth = 1, color = "deepskyblue", markersize = 2)
        ax2.set_ylabel(f"{model_name} (ADU)", fontweight = "bold", fontsize = 10)
        ax2.tick_params(axis = "y", labelsize = 7)
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2 # adds handles
        labels += l2 # adds labels
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d | %H:%M:%S")) # timestamp formatting
    plt.gcf().autofmt_xdate()
    plt.legend(handles, labels, labelcolor = "black", fontsize = 7, loc = "upper center")
    plt.title(f"Time Series of Actual vs Predicted {model_name} for Pod {pod_id_index} ({Type} Data)", fontweight = "bold", fontsize = 10)
    plt.tight_layout()
    if config.save_plots: # save plots
        filename = f"{pod_id_index}_{model_name.replace(' ', '_')}_model_timeseries.png" # export filename
        path = os.path.join(config.save_dir, filename) # join path
        plt.savefig(path, dpi = 300)
        print(f"Saved: {path}")
    plt.show()
    
def plot_residuals(X_, y_, y_pred, model_name, pod_id_index, Type = None):
    plt.style.use(config.plt_style)
    plt.figure(figsize = (5, 4))
    X_ = X_.copy() # copy
    res = y_ - y_pred # residual (actual - predicted)
    plt.scatter(X_.iloc[:, 0], res, alpha = 0.6, color = "orangered", edgecolor = "white", linewidth = 1)
    plt.axhline(y = 0, color = "k", linestyle = "-", linewidth = 1)
    plt.xlabel(f"Raw {X_.columns[0]} Signal (ADU)", fontweight = "bold", fontsize = 10)
    plt.ylabel("Residual", fontweight = "bold", fontsize = 10)
    plt.title (f"Residual Plot for {pod_id_index} {model_name} {Type} Data", fontweight = "bold", fontsize = 8)
    plt.tight_layout()
    if config.save_plots: # save plots
        filename = f"{pod_id_index}_{model_name.replace(' ', '_')}_residuals.png" # export filename
        path = os.path.join(config.save_dir, filename) # join path
        plt.savefig(path, dpi = 300)
        print(f"Saved: {path}")
    plt.show()

def add_terms(df):
    df_= df.copy()
    df_ = pd.concat([df_, pd.DataFrame({
        # cross products
        "TrH": df["Temperature"] * df["Relative Humidity"],
        "TCO2": df["Temperature"] * df["CO2"],
        "TCO": df["Temperature"] * df["CO"],
        "CO2rH": df["CO2"] * df["Relative Humidity"],
        "COrH": df["CO"] * df["Relative Humidity"],
        "CO2CO": df["CO"] * df["CO2"],
        # cross divisions
        "T_rH": df["Temperature"] / df["Relative Humidity"],
        "T_CO2": df["Temperature"] / df["CO2"],
        "T_CO": df["Temperature"] / df["CO"],
        "CO2_rH": df["CO2"] / df["Relative Humidity"],
        "CO_rH": df["CO"] / df["Relative Humidity"],
        "CO2_CO": df["CO"] / df["CO2"],
        # squared terms
        "Tsq": df["Temperature"] ** 2,
        "rHsq": df["Relative Humidity"] ** 2,
        "CO2sq": df["CO2"] ** 2,
        "COsq": df["CO"] ** 2,
        # reciprocal terms
        "one_T": 1 / df["Temperature"],
        "one_rH": 1 / df["Relative Humidity"],
        "one_CO2": 1 / df["CO2"],
        "one_CO": 1 / df["CO"], 
        # exponential terms
        "eT": np.exp(df["Temperature"]),
        "erH": np.exp(df["Relative Humidity"]),
        # log terms
        "logT": np.log(df["Temperature"]),
        "logrH": np.log(df["Relative Humidity"]),
        "logCO2": np.log(df["CO2"]),
        "logCO": np.log(df["CO"]),
        # sqrt terms
        "sqrtT": df["Temperature"] ** (1 / 2),
        "sqrtrH": df["Relative Humidity"] ** (1 / 2),
        "sqrtCO2": df["CO2"] ** (1 / 2),
        "sqrtCO": df["CO"] ** (1 / 2),
        # normalized to mean
        # "Tnorm": df["Temperature"] - df["Temperature"].mean(),
        # "rHnorm": df["Relative Humidity"] - df["Relative Humidity"].mean(),
        # "CO2norm": df["CO2"] - df["CO2"].mean(),
        # "COnorm": df["CO"] - df["CO"].mean(),
        # terms addewd to get rid of CO flatline? 
        # "scaledCO": (df["CO"] - 20000),
        # "saturatedCO": (df["CO"] >= 23000 - 10).astype(int), # saturated boolean term -- can be but in firmware later?
        # "scaledCO": df["CO"] - 25000
        # "close2satCO": np.clip((df["CO"] - df["CO"].mean() - 0.9) / 0.1, 0, 1),
        })], axis = 1)
    # numeric_df = df_.select_dtypes(include = [np.number]) # debugging
    # print("Inf count:", np.isinf(numeric_df).sum().sum()) # debugging
    # print("NaN count:", numeric_df.isna().sum().sum()) # debugging
    if config.z_score:
        numeric_cols = df_.select_dtypes(include = np.number).columns # excludes timestamps or non-numeric data
        df_ = df_.dropna(subset = numeric_cols) # drops rows with NAN values 
        df_[numeric_cols] = df_[numeric_cols].apply(zscore) # z scores data
    # print(df_) # debugging
    return df_

def plot_coefficients(model, variables, pod_id_index, model_name):
    if hasattr(model, "coef_"):
        fig, ax = plt.subplots(figsize = (5, 4))
        coefs = model.coef_.flatten()
        coef_df = pd.DataFrame({"Variable": variables, "Coefficient": coefs}).sort_values("Coefficient")
        plt.style.use(config.plt_style)
        color_norm = mcolors.Normalize(vmin = coef_df["Coefficient"].min(), vmax = coef_df["Coefficient"].max()) # gets equispacing for colors
        cmap = cm.get_cmap(config.plt_colormap)
        colors = cmap(color_norm(coef_df["Coefficient"].values))
        bars = ax.barh(coef_df["Variable"], coef_df["Coefficient"], color = colors, edgecolor = "white")
        sm = [] # init
        sm = cm.ScalarMappable(cmap = cmap, norm = color_norm) # maps values to corresponding colors
        # cbar = fig.colorbar(sm, ax = ax)
        # cbar.set_label("Coefficient Magnitude", rotation = 270, labelpad = 15, fontweight = "bold")
        if config.z_score:
            ax.set_title(f"Z-Scored Model Coefficients for {pod_id_index} {model_name} Data", fontweight = "bold", fontsize = 10)
        else:
            ax.set_title(f"Model Coefficients for {pod_id_index} {model_name} Data", fontweight = "bold", fontsize = 10)
        ax.set_xlabel("Coefficient Value", fontweight = "bold", fontsize = 10)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize = 7, fontweight = "bold") # workaround for bold font
        ax.tick_params(axis = "x", labelsize = 7)
        for bar in bars:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.2f}",
                    va = "center", fontsize = 7, fontweight = "bold")
        fig.tight_layout()
        if config.save_plots: # save plots
            filename = f"{pod_id_index}_{model_name.replace(' ', '_')}_coefficients.png" # export filename
            path = os.path.join(config.save_dir, filename) # join path
            plt.savefig(path, dpi = 300)
            print(f"Saved: {path}")
        plt.show()

def plot_bias(y_test, y_pred, model_name, pod_id_index, Type = "Testing"):
    residuals = y_pred - y_test 
    plt.style.use(config.plt_style)
    fig, ax = plt.subplots(figsize = (2, 4))
    ax.boxplot(residuals, vert = True, patch_artist = True, boxprops = dict(facecolor = "lightsalmon", color = "k"),
               medianprops = dict(color = "red", linewidth = 1))
    ax.set_title(f"Residuals for Pod\n{pod_id_index} {model_name} {Type} Data", fontweight = "bold", fontsize = 7)
    ax.set_ylabel("Residual", fontweight = "bold", fontsize = 8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = 8)
    ax.axhline(0, color = "black", linestyle = "--", linewidth = 1)  # zero bias line
    plt.tight_layout()
    if config.save_plots:
        filename = f"{pod_id_index}_{model_name.replace(' ', '_')}_bias.png"
        path = os.path.join(config.save_dir, filename)
        plt.savefig(path, dpi = 300)
        print(f"Saved: {path}")
    plt.show()

def plot_feature_importance(model_info, co2_vars, co_vars):
    def agg_plot(pollutant_name, target_vars): # helper function
        coef_sums = {v: 0.0 for v in target_vars} # dictionary
        for model, variables, model_name in model_info:
            if model_name != pollutant_name or not hasattr(model, "coef_"):
                continue # ml or parametric models
            coefs = np.abs(model.coef_.flatten())
            for var, coef in zip(variables, coefs):
                if var in coef_sums:
                    coef_sums[var] += coef # sums coefficients for variables
        coef_df = pd.DataFrame(list(coef_sums.items()), columns = ["Variable", "Total Coeff"]) # df
        coef_df["Normalized"] = coef_df["Total Coeff"] / coef_df["Total Coeff"].sum() # normalized to total sum
        coef_df.sort_values("Normalized", ascending = True, inplace = True) # ascending
        cmap = cm.get_cmap(config.plt_colormap) # backup if no colormap declared
        norm = mcolors.Normalize(vmin = coef_df["Total Coeff"].min(), vmax = coef_df["Total Coeff"].max()) # norm
        colors = cmap(norm(coef_df["Total Coeff"].values)) # colors for plotting
        plt.style.use(config.plt_style)
        fig, ax = plt.subplots(figsize = (5, 4))
        bars = ax.barh(coef_df["Variable"], coef_df["Normalized"], color = colors, edgecolor = "white")
        ax.set_title(f"Normalized Feature Importance: {pollutant_name}", fontsize = 10, fontweight = "bold")
        ax.set_xlabel("Normalized Importance", fontsize = 10, fontweight = "bold")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize = 7, fontweight = "bold") # workaround for bold font
        ax.tick_params(axis = "x", labelsize = 7)
        for bar in bars:
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2, f"{(bar.get_width() * 100):.1f}%",
                    va = "center", fontsize = 7, fontweight = "bold") # bar text but make it nice
        plt.tight_layout()
        if config.save_plots:
            filename = f"{pollutant_name.replace(' ', '_')}_feature_importance.png"
            path = os.path.join(config.save_dir, filename)
            plt.savefig(path, dpi = 300)
            print(f"Saved: {path}")
        plt.show()
        return coef_df
    
    co2_importance = agg_plot("CO2", co2_vars)
    co_importance = agg_plot("CO", co_vars)
    return co2_importance, co_importance