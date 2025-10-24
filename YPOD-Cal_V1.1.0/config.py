import pandas as pd
# ****************************************************************************
# ****************************************************************************
save_dir = "/Users/Chiara/Documents/Documents - Chiara’s MacBook Air/Research/YPOD Calibration/Model Improvement" # directory for write files
pod_id = ["B8", "A1", "U4"]
#["U4", "V1", "U8", "X0", "F2", "F4", "D4", "K3", "Z2", "Z3", "U7", "U6", "U2", "U1", "A1", "T4", "T2", "O1", "L1", "B8", "G2", "D8"] # pods to look at
### YPOD header names in order of csv file outputs
pod_headers = ["Timestamp", "Longitude", "Latitude", "PodID", "T_BME", "P_BME", "Temperature", "Relative Humidity", 
               "Fig2600", "Fig2602", "Ozone", "CO", "CO2", "PM1.0", "PM2.5", "PM10.0","Blank"] # headers for YPOD firmware
pod_col_drop = ["Blank", "Longitude", "Latitude", "T_BME"] # columns to drop in ypod data 
pollutants = ["CO2", "CO", "Relative Humidity", "Temperature"] # pollutant plots to generate, needs to match header
start_cutoff = pd.to_datetime("2025-07-09 10:00:00", format = "mixed") # start time for data window
end_cutoff = pd.to_datetime("2025-07-17 07:00:00", format = "mixed") # end time for data window
filename_date = "_2025_07_09.CSV"
sample_time = "5min" # "h" for hourly, "d" for daily "xmin" for x minutes
ref2_filename = "CO_CO2_Shed_Data.CSV" # reference file name
ref2_headers = ["Timestamp UTC", "Timestamp", "CO2_adu", "H2O_adu", "CO_adu", "CO2_ppm", "CO_ppm"] # referance file headers
ref2_row_drop = 1 # reference row to drop / ignore, include header rows
ref2_col_drop = ["CO2_adu", "H2O_adu", "CO_adu", "Timestamp UTC"] # reference columns to drop / ignore
ref2_data = ["CO2_ppm", "CO_ppm"] # desired headers / data to look at and calibrate to
### initializes arrays for each variable, needs to be in order, length of number of reference files
number_of_files = 1
filenames = [ref2_filename] # filename array
headers = [ref2_headers] # header array
row_drop = [ref2_row_drop] # row drop array
col_drop = [ref2_col_drop] # column drop array
data = [ref2_data] # data storage array
### variables to run models against (parameters to take into account), must match pod headers
co_regr_var = ["CO", "Relative Humidity", "Temperature"] # carbon monoxide model
co2_regr_var = ["CO2", "Relative Humidity", "Temperature"] # carbon dioxide model
models = ["CO2", "CO"] # parameters being calibrated
calibrated_units = ["(PPM)", "(PPM)"] # units for calibrated models, must be in order of "models" array
### stores regression variables in an array
regr_var = [co2_regr_var, co_regr_var] # change when adding a model, must be in order of "models" array
ref_cal_var = ["CO2_ppm", "CO_ppm"] # reference variables used in calibration, must match ref instrument headers
### regression type toggles
linear_regression_mvlr = False # uses multivariable linear regression, least squares
ridge_mvlr = True # uses ridge linear regression, least squares with coefficient size penalties
lasso_mvlr = False # uses lasso linear regression, least squares with added regularization term, reduces features
gradient_boost_model = False # uses gradient boosting ML
random_forest_model = False # uses random forest ML
### plotting toggles
raw_pod_data_plotting = True # plots raw timeseries data from pods in pod_id array
model_accuracy_plotting = True # plots actual vs predicted model values for calibration variable (linear comparison)
model_timeseries_plotting = True # plots model timeseries data against referece timeseries data (timeseries comparison)
display_pod_data_timeseries = True # displays the pod timeseries data on the above plot
residual_plotting = True
save_plots = False # saves plots to save_dir
# ****************************************************************************
# ****************************************************************************