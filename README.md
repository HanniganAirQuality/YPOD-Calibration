# YPOD-Calibration

# Python script written to run five models of regression for calibration with a reference

|Model                        |Type                                                |
|-----------------------------|----------------------------------------------------|
|Linear Regression (regular)  |Standard multivariable regression                   |
|Linear Regression (ridge)    |Multivariable with L2 penalty nonzero coefficients  |
|Linear Regression (lasso)    |Multivariable with L2 penalty zero coefficients     |
|Gradient Boosting            |Machine learning w/ outlier handling, single tree   |
|Random Forest                |Machine learning w/ outlier handling, multiple trees|

## Addition of z-scoring, coefficient investigations

## Save_plots toggle is needed to export plots, plots include:
1. Raw data plotting: visualizes timeseries data as it comes off of the SD cards
2. Model accuracy plotting: linear scatter plot for model prediction vs actual data
3. Model timeseries plotting: timeseries data of reference vs model prediction
   - display_pod_data_timeseries toggle will display the raw pod data on this chart
4. Residual plotting: plots residuals for regression model
5. Bias plotting: plots bar charts for residuals to show bias
6. Z-score: z-scores all data in plots etc.
7. Coefficient plotting: plots the coefficients on each variable in the linear regression
8. Feature importance plotting: plots the normalized importance of each variable based on it's coefficient (this only applies to z-scored data)
