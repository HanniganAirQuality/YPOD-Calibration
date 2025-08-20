# YPOD-Calibration

# Python script written to run five models of regression for calibration with a reference

|Model                        |Type                                                |
|-----------------------------|----------------------------------------------------|
|Linear Regression (regular)  |Standard multivariable regression                   |
|Linear Regression (ridge)    |Multivariable with L2 penalty nonzero coefficients  |
|Linear Regression (lasso)    |Multivariable with L2 penalty zero coefficients     |
|Gradient Boosting            |Machine learning w/ outlier handling, single tree   |
|Random Forest                |Machine learning w/ outlier handling, multiple trees|

## Code is structured so that all variables that need to be changed *should* be in the top cell. However, the code may still throw errors if syntax is off or wrong names are used. 

## save_plots toggle is needed to export plots, plots include:
1. Raw data plottin: visualizes timeseries data as it comes off of the SD cards
2. Model accuracy plotting: linear scatter plot for model prediction vs actual data
3. Model timeseries plotting: timeseries data of reference vs model prediction
   - display_pod_data_timeseries toggle will display the raw pod data on this chart
