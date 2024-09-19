import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
# os.environ['R_HOME'] = "C:\\Users\\VW489VV\\AppData\\Local\\Programs\\R\\R-4.3.3"
import datetime
import math
from scipy.special import logit, expit # expit is the inverse of logit
import statsmodels
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from itertools import combinations
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.iolib.summary import (Summary, summary_params, table_extend)
import statsmodels.stats.diagnostic as dg
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
import statsmodels.stats.api as sms
from time import gmtime, strftime

from warnings import simplefilter, filterwarnings
simplefilter(action='ignore', category=FutureWarning)
filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=UserWarning)
from pandas.errors import PerformanceWarning
filterwarnings("ignore", category=PerformanceWarning)

import psutil
from pathlib import Path
from multiprocessing import Pool
from itertools import chain
import itertools
import timeit
from joblib import Parallel, delayed
import multiprocessing
import glob
import scipy.stats as stats
import shelve
import datetime
import sys
#pip install arch
from arch.unitroot import *
import os, shutil
#pip install rpy2
from rpy2 import robjects as r

# 1 Manually set the working directory for input files
input_dir = sys.argv[1]
os.chdir(input_dir)

# 2 The name of the file where the results will be saved 
portfolio_dir = sys.argv[2]
portfolio_abs_dir = input_dir + "\\" + portfolio_dir

# Find the Hypothesis and the dependent variable for the current running
portfolio, hypo = portfolio_dir.split("_")

# Importing the dataset
print(input_dir) # verify in the console if the path of the working drectory is correct
input_files = glob.glob("*.xlsx")
if len(input_files) > 1:
    raise Exception(f"There should only be one excel file in the base directory. Found {len(input_files)}:\n{input_files}")

dataset = pd.read_excel(input_files[0])
dataset = pd.concat([pd.DataFrame(dataset['Date']),
    pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns)==portfolio]]),
    pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('x')]])], axis=1)

# 3 Reporting Date
os.chdir(portfolio_abs_dir)
dataset_run = pd.read_excel(glob.glob("01.*")[0])
RepDate = dataset_run.dropna(subset=[portfolio])["Date"].max()

# 6 Number of lags to be used in modelling
# H0: Hypothesis 0
# no_lags = 2 # default is 2;
# H2: Hypothesis 2 if no model resulted from H1
pattern = r'lag(\d+)'
lag_numbers = []
for column in dataset_run.columns:
    match = re.search(pattern, column)
    if match:
        lag_numbers.append(int(match.group(1)))
        
if lag_numbers:
    largest_lag_number = max(lag_numbers)
    
no_lags = largest_lag_number

# Initialize the DataFrame to store BPV results
model_list_BPV = pd.DataFrame()

# Read the Excel file into a pandas DataFrame
model_list_final_candidate = pd.read_excel(glob.glob("12.*")[0])
start = timeit.default_timer()

#dataset_analiza= pd.read_excel('06.Dataset_Agro_H1.xlsx')
#dataset_analiza = pd.DataFrame(dataset_analiza.loc[dataset_analiza.Date>RepDate,:]) 

#--------------------------------------------------------------------------------------------

dir_hypo = str("BVP_Analysis_" + portfolio + "_" + hypo)
if not os.path.exists(dir_hypo):
    os.makedirs(dir_hypo) # create a folder named after the value set for parameter portfolio and current hypothesis
else: 
    filelist = glob.glob(os.path.join(dir_hypo, "*"))
    for f in filelist:
        os.remove(f)
new_path = portfolio_abs_dir + "\\" + dir_hypo # path of the output
os.chdir(str(new_path)) # change the working directory for output storage

# Determine the oldest available DR for a specific portfolio (ALL is the DR variable)
min_rep_date = pd.DataFrame(dataset.loc[pd.isnull(dataset[portfolio])==False,:]).Date.min()

# Create a different dataframe for initial macro variables
data_macro = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('x')]])
#data_macro["xCPIonIR"] = (data_macro["xCPI"] / data_macro["xIR"])/100
#data_macro["xURonGDP"] = (data_macro["xUR"] / data_macro["xGDP"])/100


def apply_percent_change(df, percent=1):
    # Calculate the increase and decrease by the given percentage
    BPV_factor = percent / 100
    
    # Create new DataFrame for storing the results
    data_macro_increased = df.loc[:, data_macro.columns != 'Date'] + BPV_factor
    data_macro_decreased = df.loc[:, df.columns != 'Date'] - BPV_factor
    
    # Add suffixes to the columns to indicate the transformation
    data_macro_increased = data_macro_increased.add_suffix(f'_BPVinc')
    data_macro_decreased = data_macro_decreased.add_suffix(f'_BPVdec')
    
    # Combine the original DataFrame with the new transformed columns
    data_macro_combined = pd.concat([data_macro, data_macro_increased, data_macro_decreased], axis=1)
    
    return data_macro_combined

percentage = int(sys.argv[4])
data_macro = apply_percent_change(data_macro, percent=percentage)


#### Create a different dataframe for dependent variables
data_dep = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('x')==False]])

# Create column x...adj by shifting the initial variable / column as to have values > 0
# list_of_neg_ind_var = list()
# for i in range(0,len(data_macro.columns)):
#     if data_macro.iloc[:,i].min() < 0:
#         list_of_neg_ind_var.append(data_macro.columns[i])
#         data_macro[data_macro.columns[i]+'adj'] = data_macro.iloc[:,i] - data_macro.iloc[:,i].min() + 0.0001

# # Drop initial negative columns from data_macro
# data_macro = data_macro.drop(list_of_neg_ind_var, axis=1)


# Concatenate the final macro variables to be used in the model in the initial dataset
dataset = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('x')==False]])
dataset = pd.concat([dataset,data_macro], axis=1)


# Create additional columns with lagged macro variables (lags up to value of the defined parameter no_lags)        
for i in range (1,no_lags+1):        
    dataset[pd.Series(data_macro.add_suffix('_lag'+str(i)).columns)] = data_macro.shift(+i)
    
# Filling missing values before reporting date only 
dataset = dataset.bfill() # each missing value is replaced with the value from the next row

# Create transformation of values
# # Moving Average of order 2
# data_ma2 = dataset.rolling(2).mean() # each value is the result of a moving average of 2
# data_ma2 = data_ma2.add_suffix('_ma2') # add a suffix to each column name from dataframe

# Relative Change of order 1 for MA(2)
data_change = dataset.loc[:, dataset.columns!='Date'].pct_change(periods=1)
data_change = data_change.add_suffix('_change')

# 1st Difference
data_diff = dataset.loc[:, dataset.columns!='Date'].diff(periods=1, axis=0)
data_diff = data_diff.add_suffix('_diff')

# reciprocal
data_reciprocal = pd.DataFrame(columns = pd.Series((dataset.loc[:,dataset.columns != 'Date']).add_suffix('_reciprocal').columns))
for i in range (0,len(data_reciprocal.columns)-1):
    data_reciprocal.iloc[:,i] = 1/(dataset.iloc[:,(i+1)]) 
# the last column is not calculated based on the loop, so it will be computed separately
data_reciprocal.iloc[:,len(data_reciprocal.columns)-1] = 1/(dataset.iloc[:,len(dataset.columns)-1])


# Cube root on initial dataframe applied on all data series
data_cuberoot = pd.DataFrame(columns = pd.Series((dataset.loc[:,dataset.columns != 'Date']).add_suffix('_cuberoot').columns))
for i in range (0,len(data_cuberoot.columns)-1):
    data_cuberoot.iloc[:,i] = np.cbrt(dataset.iloc[:,(i+1)]) 
# the last column is not calculated based on the loop, so it will be computed separately
data_cuberoot.iloc[:,len(data_cuberoot.columns)-1] = np.cbrt(dataset.iloc[:,len(dataset.columns)-1])

# Difference of cube root dataframe
data_cuberootdiff = data_cuberoot.diff(periods=1, axis=0)
data_cuberootdiff = data_cuberootdiff.add_suffix('_diff')


# Annual absolute difference applied on independent only
data_macro = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('x')]])


data_diffa = data_macro.loc[:, data_macro.columns!='Date'].diff(periods=4, axis=0)
data_diffa = data_diffa.add_suffix('_diffa')




# Square root on initial dataframe/applied only on dependent var
data_sqrt = pd.DataFrame(columns = pd.Series((data_dep.loc[:,data_dep.columns != 'Date']).add_suffix('_sqrt').columns))
for i in range (0,len(data_sqrt.columns)-1):
    data_sqrt.iloc[:,i] = np.sqrt(data_dep.iloc[:,(i+1)]) 
# the last column is not calculated based on the loop, so it will be computed separately
data_sqrt.iloc[:,len(data_sqrt.columns)-1] = np.sqrt(data_dep.iloc[:,len(data_dep.columns)-1])



# Difference of difference
# data_diff2 = (dataset.loc[:, dataset.columns!='Date'].diff(periods=1, axis=0)).diff(periods=1, axis=0)
# data_diff2 = data_diff2.add_suffix('_diff_diff')

# Logit transformation: The logit function is defined as logit(p) = log(p/(1-p)). Note that logit(0) = -inf, logit(1) = inf, and logit(p) for p<0 or p>1 yields nan.
#/applied only on dependent var
data_logit = pd.DataFrame(columns = pd.Series((data_dep.loc[:,data_dep.columns != 'Date']).add_suffix('_logit').columns))
for i in range (0,len(data_logit.columns)-1):
    data_logit.iloc[:,i] = logit(data_dep.iloc[:,(i+1)]) 
# the last column is not calculated based on the loop, so it will be computed separately
data_logit.iloc[:,len(data_logit.columns)-1] = logit(data_dep.iloc[:,len(data_dep.columns)-1])

# Difference of logit dataframe/applied only on dependent var
data_logitdiff = data_logit.diff(periods=1, axis=0)
data_logitdiff = data_logitdiff.add_suffix('_diff')


# Natural logarithm tranformation applied on dependent only
data_log = np.log(data_dep.loc[:, data_dep.columns != 'Date'] )
data_log = data_log.add_suffix('_log')


# Difference of log dataframe
data_logdiff = data_log.diff(periods=1, axis=0)
data_logdiff = data_logdiff.add_suffix('_diff')


# Merge all dataframes containing transformed variables with the initial dataset
dataset = pd.concat([dataset, data_diff, data_logit, data_logitdiff, data_change, data_sqrt, data_reciprocal, data_diffa, data_log, data_logdiff, data_cuberoot, data_cuberootdiff], axis=1) #REMOVED: diff_diff; ADDED: diffa, log, log_diff, cuberoot, cuberoot_diff

# Delete the columns which are entirely with missing values
dataset = dataset.dropna(axis='columns', how='all')

# Keep the data from the first available PD
dataset = pd.DataFrame(dataset.loc[dataset.Date>=min_rep_date,:])

# Export dataset ALL in excel
dataset.to_excel (str(new_path)+"\\"+r'01.Data_'+portfolio+'_'+hypo+'.xlsx', index = False, header=True)
# STATIONARITY ANALYSIS
dataset_train = pd.DataFrame(dataset.loc[dataset.Date<=RepDate,:]) # filter only the historical data
dataset_train = pd.DataFrame(dataset_train.loc[:,dataset_train.columns != 'Date']) # drop column Date
dataset_train = dataset_train.bfill()

# dependent variables, all columns from dataset_train not containing macro variables
dependent_var = pd.DataFrame(dataset_train.loc[:,dataset_train.columns[pd.Series(dataset_train.columns).str.startswith('x')==False]])

# independent variables, all columns from dataset_train not containing PDs
independent_var = dataset_train.drop(pd.Series(dependent_var.columns), axis=1)

#-------------------------------------------------------------------------------------------


# Calculate Lag 1 dependent variables for train dataset with stationary variables
# Create a dataset with stationary dependent variables which will be further lagged (order 1)
lagged_dependent = pd.DataFrame(dataset_train.loc[:,dataset_train.columns[pd.Series(dataset_train.columns).str.startswith('x')==False]])

# Shift the dataframe lagged_dependent with 1 lag                
lagged_dependent = lagged_dependent.shift(+1)
    
# Filling missing data on the first row for lagged dependent shifted 
lagged_dependent = lagged_dependent.bfill() # each missing value is replaced with the value from the next row

# Rename the columns from lagged_dependent dataset by adding teh suffix "lagged"
lagged_dependent = lagged_dependent.add_suffix('_lagged')

# Add the lagged_dependent dataset to dataset_train
dataset_train = pd.concat([dataset_train, lagged_dependent], axis=1)

### REMEMBER: change something only for PD LOAN product

# Sort the columns from dataset_train alphabetically and export to excel
dataset_train = dataset_train[sorted(dataset_train.columns)]
dataset_train.to_excel (str(new_path)+"\\"+r'05.Dataset_train_'+portfolio+'.xlsx', index = False, header=True)


# Calculate Lag 1 dependent variables for entire dataset
# Create a dataset with all dependent variables which will be further lagged (order 1)
lagged_dependent = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('x')==False]])
lagged_dependent = lagged_dependent.drop('Date', axis=1)

# Shift the dataframe lagged_dependent with 1 lag                
lagged_dependent = lagged_dependent.shift(+1)
    
# Rename the columns from lagged_dependent dataset by adding teh suffix "lagged"
lagged_dependent = lagged_dependent.add_suffix('_lagged')

# Add the lagged_dependent dataset to dataset_train
dataset = pd.concat([dataset, lagged_dependent], axis=1)

# Sort the columns from dataset alphabetically and export to excel
dataset = dataset[sorted(dataset.columns)]
dataset.to_excel (str(new_path)+"\\"+r'06.Dataset_'+portfolio+'_'+hypo+'.xlsx', index = False, header=True)

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

# # Store the forecasted dates
# Date = dataset_analiza['Date'].reset_index(drop=True)
# New_Date = []
# for date in Date:
#     New_Date.append(date.strftime('%Y%m%d'))
# New_Date_prefixed_BPVinc = ['Y_pred_BPVinc_'+char for char in New_Date]
# New_Date_prefixed_BPVdec = ['Y_pred_BPVdec_'+char for char in New_Date]

# model_list_BPV = pd.DataFrame()
# model_list_BPV[[New_Date_prefixed_BPVinc]] = None
# model_list_BPV[[New_Date_prefixed_BPVdec]] = None



#Find the selected model and train it
row_index = int(sys.argv[3])

print(f"Model name: {model_list_final_candidate.iloc[row_index]['Model_number']}")

Model_number=model_list_final_candidate.iloc[row_index]['Model_number']

# Prepare the list of independent variables for the selected model
temp_list: list = model_list_final_candidate.iloc[row_index]['Independent'].strip("()[]").replace("'", "").split(", ")

# Prepare the training data for the selected model
X_train = pd.DataFrame(dataset_train[temp_list])
X_train.insert(loc=0, column='const', value=1)  # Add constant for intercept

# Extract the dependent variable for the selected model
dependent_var = model_list_final_candidate.iloc[row_index]['Dependent']
Y_train = pd.DataFrame(dataset_train[dependent_var])

# Drop rows with NaN values from X_train and Y_train separately
X_train = X_train.dropna().reset_index(drop=True)
Y_train = Y_train.dropna().reset_index(drop=True)

# Perform an inner join to ensure rows in X_train and Y_train match up
# This step assumes that the index of X_train and Y_train should match after dropping NaN values
combined_train = pd.concat([X_train, Y_train], axis=1, join='inner').dropna()

# Split the combined DataFrame back into X_train and Y_train
X_train = combined_train[ ['const'] + temp_list]
Y_train = combined_train[[dependent_var]]


# Fit the linear regression model for the selected model
#print("X_train")
#print(X_train)

model = sm.OLS(Y_train, X_train).fit()

#print(model.summary())
print("MODEL DATA")
num_coefficients = len(model.params)
model_feature_names = model.params.index.tolist()
print(f"num_coef: {num_coefficients}")
print(f"feature_names: {model_feature_names}\n\n")
print(model.params)

#-------------------------------------------------------------------------------------


# create out of sample dataframe for calculating Y_pred
dataset_analiza = pd.DataFrame(dataset.loc[dataset.Date>RepDate,:]) 

# Store the forecasted dates
Date = dataset_analiza['Date'].reset_index(drop=True)
New_Date = []
for date in Date:
    New_Date.append(date.strftime('%Y%m%d'))
New_Date_prefixed_BPVinc = ['Y_pred_BPVinc_'+char for char in New_Date]
New_Date_prefixed_BPVdec = ['Y_pred_BPVdec_'+char for char in New_Date]

model_list_BPV = pd.DataFrame()
model_list_BPV[[New_Date_prefixed_BPVinc]] = None
model_list_BPV[[New_Date_prefixed_BPVdec]] = None


#----------------------------------------------------------

# Convert the forecasted Y into the original dataseries
# Define a function that calculates the reverse Y_pred

def reverse_y_pred(dependent_var_name, Y):
    char_list = list()
    char_list.append(dependent_var_name)
    if dependent_var_name.rfind('_')>portfolio.rfind('_'): 
        len_dep_variable=dependent_var_name.rfind('_')
        char = dependent_var_name[0:len_dep_variable]
        char_list.append(char)
    else:    
        char = dependent_var_name
    while char.rfind('_')>portfolio.rfind('_'):
        char = char[0:char.rfind('_')]
        char_list.append(char)
    Y_orig = Y
    Y_orig.columns = range(Y_orig.columns.size)
    Y_orig.rename(columns = {0:dependent_var_name}, inplace=True)
    Y_orig.reset_index(inplace=True)
    Y_orig.rename(columns = {"index":"Date"}, inplace=True)
    if len(char_list) > 1:
        Y_orig = pd.concat( [Y_orig,
                    ((pd.DataFrame(dataset.loc[dataset.Date<=RepDate,char_list[1:]])).tail(5)).reset_index(drop=True) ],
                  axis = 1)
        for s in range(2,len(Y_orig.columns)):
            for t in range(5,len(Y_orig)):
                suffix = Y_orig.columns[s-1]
                if (suffix[suffix.rfind('_')+1:]=='diff') | (suffix[suffix.rfind('_')+1:]=='diff_diff'):
                    Y_orig.iloc[t,s] = Y_orig.iloc[t,s-1]+Y_orig.iloc[t-1,s]
                elif suffix[suffix.rfind('_')+1:]=='logit':
                    Y_orig.iloc[t,s] = expit(Y_orig.iloc[t,s-1])
                elif suffix[suffix.rfind('_')+1:]=='ma2':
                    Y_orig.iloc[t,s] = 2*Y_orig.iloc[t,s-1]-Y_orig.iloc[t-1,s]
                elif suffix[suffix.rfind('_')+1:]=='sqrt':
                    Y_orig.iloc[t,s] = pow(Y_orig.iloc[t,s-1],2)
                elif suffix[suffix.rfind('_')+1:]=='change':
                    Y_orig.iloc[t,s] = (Y_orig.iloc[t,s-1]+1)*Y_orig.iloc[t-1,s]
                elif suffix[suffix.rfind('_')+1:]=='reciprocal':
                    Y_orig.iloc[t,s] = 1/Y_orig.iloc[t,s]
                elif suffix[suffix.rfind('_')+1:]=='cuberoot':
                    Y_orig.iloc[t,s] = pow(Y_orig.iloc[t,s-1],3)
                elif suffix[suffix.rfind('_')+1:]=='log':
                    Y_orig.iloc[t,s] = np.exp(Y_orig.iloc[t,s-1])  
    return Y_orig

def prepare_test_data(dataset_analiza, var_name, 
                      temp_list, lagged=True):
    # Create a DataFrame with the 'const' column for the intercept
    input_dict = {}
    for temp_var in temp_list:
        prefix_var_name = var_name.split("_")[0]
        prefix_temp_var = temp_var.split("_")[0]
        
        if prefix_var_name == prefix_temp_var:
            input_dict.update({var_name: dataset_analiza[var_name]})
        else:
            input_dict.update({temp_var: dataset_analiza[temp_var]})
    
    
    X_test = pd.DataFrame(input_dict, index=dataset_analiza.index)
    X_test.insert(loc=0, column='const', value=1)
    
    if lagged:
        X_test = X_test.reset_index(drop=True)
    
    return X_test

def run_predict(X_test, pref, new_date, lagged=True):
    Y_pred = pd.DataFrame(model.predict(X_test)).T
    if lagged:
        for m in range(1, len(X_test)):
            X_test.iloc[m, 1] = Y_pred.iloc[0, m - 1]
            Y_pred = pd.DataFrame(model.predict(X_test)).T
    Y_pred.columns = new_date
    Y_pred = Y_pred.add_prefix(pref)  
    return Y_pred

def string_to_list(string):
    return string.strip("()[]").replace("'", "").split(", ")

def process_sensitivity_pred(var_name, dependent_var, 
                             temp_list, dataset_analiza):
    #if var_name[0] == "x":    
    split_list = var_name.split("_")
    split_list.insert(1, "BPVinc") 
    inc_var_name = "_".join(split_list)

    split_list = var_name.split("_")
    split_list.insert(1, "BPVdec") 
    dec_var_name = "_".join(split_list)
                
    #temp_list_inc = dataset_analiza[inc_var_name]
    #temp_list_dec = dataset_analiza[dec_var_name]

    print(inc_var_name)
    print(dec_var_name)
    
    # Determine if the dependent variable is lagged and in the independent variables list
    if f"{dependent_var}_lagged" in temp_list:
        X_test_inc = prepare_test_data(
            dataset_analiza, 
            inc_var_name,
            temp_list=temp_list
            )
        # Predict the Y values iteratively to account for lagged dependent variable
        Y_pred_inc = run_predict(
            X_test=X_test_inc, 
            pref='Y_pred_BPVinc_',
            new_date=New_Date
            )

        # Prepare test data for BPVdec
        X_test_dec = prepare_test_data(
            dataset_analiza, 
            dec_var_name,
            temp_list=temp_list
            )
        # Predict the Y values iteratively to account for lagged dependent variable
        Y_pred_dec = run_predict(
            X_test_dec, 
            pref='Y_pred_BPVdec_',
            new_date = New_Date
            )
    else:
        # Prepare test data for BPVinc without iterative prediction
        X_test_inc = prepare_test_data(
            dataset_analiza, 
            inc_var_name, 
            temp_list=temp_list,
            lagged=False
            )
        # Predict the Y values
        Y_pred_inc = run_predict(
            X_test_inc, 
            pref='Y_pred_BPVinc_',
            new_date=New_Date,
            lagged=False
            )

        # Prepare test data for BPVdec without iterative prediction
        X_test_dec = prepare_test_data(
            dataset_analiza, 
            dec_var_name,
            temp_list=temp_list,
            lagged=False
            )
        # Predict the Y values
        Y_pred_dec = run_predict(
            X_test_dec, 
            pref='Y_pred_BPVdec_',
            new_date=New_Date,
            lagged=False
            )

    return Y_pred_inc, Y_pred_dec
        
        
for idx, var_name in enumerate(temp_list):
    if var_name[0] != "x":
        continue
    #try:
    Y_pred_inc, Y_pred_dec = process_sensitivity_pred(
                                var_name, 
                                dependent_var,
                                temp_list,
                                dataset_analiza
                            )

    model_list_temp_BPV = pd.concat([Y_pred_inc, Y_pred_dec], axis=1)
    # Append the results to model_list_interim dataframe
    model_list_BPV = pd.concat([model_list_BPV, model_list_temp_BPV], axis=0)


# Calculate Y_orig
Y_orig_df = pd.DataFrame()

Y = pd.DataFrame( 
        model_list_final_candidate.loc\
        [row_index, 
            list(
                model_list_final_candidate\
                .columns[(model_list_final_candidate.columns)\
                            .str.startswith('Y_orig')==True])
        ] )


dependent_var_name = dependent_var
# Y_orig = reverse_y_pred(dependent_var_name, Y)  
Y_orig = pd.DataFrame(Y.iloc[0:,-1]).T
Y_orig.columns = New_Date
Y_orig = Y_orig.add_prefix('Y_orig_')
Y_orig = Y_orig.reset_index(drop=True)
Y_orig_df = pd.concat([Y_orig_df,Y_orig], axis=0)


Y_orig_df = Y_orig_df.reset_index(drop=True)

print(Y_orig_df)


# Reset the index of model_list_BPV to align for concatenation
model_list_BPV.reset_index(drop=True, inplace=True)

# Calculate Y_orig
Y_orig_df_BPVinc = pd.DataFrame()
for i in range (0, len(model_list_BPV)):
    Y_BPVinc = pd.DataFrame( 
        model_list_BPV.loc[
            i, 
            list(
                model_list_BPV.columns[
                    (model_list_BPV.columns).str.startswith('Y_pred_BPVinc')==True])
                    ])
    # First, create the DataFrame from model_list_BPV
    Y_BPVinc_df = pd.DataFrame(model_list_final_candidate.loc\
        [row_index, 
            list(
                model_list_final_candidate\
                .columns[(model_list_final_candidate.columns)\
                            .str.startswith('Y_act')==True])
        ] )
    Y_BPVinc.columns = [0]
    Y_BPVinc_df.columns = [0]
    Y_BPVinc = pd.concat([Y_BPVinc_df, Y_BPVinc], axis=0)

    dependent_var_name = dependent_var
    Y_orig_BPV = reverse_y_pred(dependent_var_name, Y_BPVinc)  
    Y_orig_BPV = pd.DataFrame(Y_orig_BPV.iloc[5:,-1]).T
    Y_orig_BPV.columns = New_Date
    Y_orig_BPV = Y_orig_BPV.add_prefix('Y_orig_BPVinc')
    Y_orig_BPV = Y_orig_BPV.reset_index(drop=True)

    print("Y_orig_BPV increase:")
    print(Y_orig_BPV)

    Y_orig_df_BPVinc = pd.concat([Y_orig_df_BPVinc,Y_orig_BPV], axis=0)
Y_orig_df_BPVinc = Y_orig_df_BPVinc.reset_index(drop=True)


# Add the Y_orig_BPV to the model_list_BPV dataframe
model_list_BPV = pd.concat([model_list_BPV, Y_orig_df_BPVinc], axis = 1)

Y_orig_df_BPVdec = pd.DataFrame()
for i in range (0, len(model_list_BPV)):
    Y_BPVdec = pd.DataFrame( 
        model_list_BPV.loc[
            i, list(
                model_list_BPV\
                    .columns[(model_list_BPV.columns)\
                    .str.startswith('Y_pred_BPVdec')==True])] 
                    )
    # First, create the DataFrame from model_list_BPV
    Y_BPVdec_df = pd.DataFrame(model_list_final_candidate.loc\
        [row_index, 
            list(
                model_list_final_candidate\
                .columns[(model_list_final_candidate.columns)\
                            .str.startswith('Y_act')==True])
        ] )
    Y_BPVdec.columns = [0]
    Y_BPVdec_df.columns = [0]
    Y_BPVdec = pd.concat([Y_BPVdec_df, Y_BPVdec], axis=0)

    dependent_var_name = dependent_var
    Y_orig_BPV = reverse_y_pred(dependent_var_name, Y_BPVdec)  
    Y_orig_BPV = pd.DataFrame(Y_orig_BPV.iloc[5:,-1]).T
    Y_orig_BPV.columns = New_Date
    Y_orig_BPV = Y_orig_BPV.add_prefix('Y_orig_BPVdec')
    Y_orig_BPV = Y_orig_BPV.reset_index(drop=True)

    print("Y_orig_BPV decrease:")
    print(Y_orig_BPV)

    Y_orig_df_BPVdec = pd.concat([Y_orig_df_BPVdec,Y_orig_BPV], axis=0)
Y_orig_df_BPVdec = Y_orig_df_BPVdec.reset_index(drop=True)

# Add the Y_orig_BPV to the model_list_BPV dataframe
model_list_BPV = pd.concat([model_list_BPV, Y_orig_df_BPVdec], axis = 1)

# Calculate the min and max of Y_orig_BPV for each model and test if it fits into interval [0,1]
model_list_BPV['Min_Y_orig_BPVinc'] = model_list_BPV[ pd.Series(model_list_BPV.columns[((model_list_BPV.columns).str.startswith('Y_orig_BPVinc')==True)]) ].min(axis=1)
model_list_BPV['Max_Y_orig_BPVinc'] = model_list_BPV[ pd.Series(model_list_BPV.columns[((model_list_BPV.columns).str.startswith('Y_orig_BPVinc')==True)]) ].max(axis=1)
try: 
    model_list_BPV['Avg_Y_orig_BPVinc'] = np.average(model_list_BPV[ pd.Series(model_list_BPV.columns[((model_list_BPV.columns).str.startswith('Y_orig_BPVinc')==True)])[0:4] ],axis=1)
except ZeroDivisionError: 
    0
model_list_BPV['Y_pred_check_BPVinc'] = np.where( ((model_list_BPV['Min_Y_orig_BPVinc'] >= 0) & (model_list_BPV['Max_Y_orig_BPVinc'] <= 1)),
                                            'OK', 'NOK')

# Select columns that start with 'Y_orig_BPVdec'
increase_columns = model_list_BPV.columns[model_list_BPV.columns.str.startswith('Y_orig_BPVinc')]

# Calculate the average of the selected columns
model_list_BPV['Average_Increase'] = model_list_BPV[increase_columns].mean(axis=1)

Average_Increase = model_list_BPV[increase_columns].mean(axis=1)

model_list_BPV['Min_Y_orig_BPVdec'] = model_list_BPV[ pd.Series(model_list_BPV.columns[((model_list_BPV.columns).str.startswith('Y_orig_BPVdec')==True)]) ].min(axis=1)
model_list_BPV['Max_Y_orig_BPVdec'] = model_list_BPV[ pd.Series(model_list_BPV.columns[((model_list_BPV.columns).str.startswith('Y_orig_BPVdec')==True)]) ].max(axis=1)
try: 
    model_list_BPV['Avg_Y_orig_BPVdec'] = np.average(model_list_BPV[ pd.Series(model_list_BPV.columns[((model_list_BPV.columns).str.startswith('Y_orig_BPVdec')==True)])[0:4] ],axis=1)
except ZeroDivisionError: 
    0
model_list_BPV['Y_pred_check_BPVdec'] = np.where( ((model_list_BPV['Min_Y_orig_BPVdec'] >= 0) & (model_list_BPV['Max_Y_orig_BPVdec'] <= 1)),
                                            'OK', 'NOK')

# Select columns that start with 'Y_orig_BPVdec'
decrease_columns = model_list_BPV.columns[model_list_BPV.columns.str.startswith('Y_orig_BPVdec')]

# Calculate the average of the selected columns
model_list_BPV['Average_Decrease'] = model_list_BPV[decrease_columns].mean(axis=1)

Average_Decrease = model_list_BPV[decrease_columns].mean(axis=1)

average_base_columns = Y_orig_df
Average_Baseline = average_base_columns.mean(axis=1).values[0]  
model_list_BPV['Average_Baseline'] = Average_Baseline

Relative_diff_increase = (Average_Increase / Average_Baseline - 1)
Relative_diff_decrease = (Average_Decrease / Average_Baseline - 1)

# Assign to the respective columns in model_list_BPV
model_list_BPV['Relative_diff_increase'] = Relative_diff_increase
model_list_BPV['Relative_diff_decrease'] = Relative_diff_decrease
model_list_BPV['Percentage'] = percentage


# Extract the model number from 'model_list_final_candidate'
model_number = model_list_final_candidate.iloc[row_index]['Model_number']

# Create a copy of temp_list to work with
variable_bpv_analysis = temp_list.copy()

# Add original independent variables
if f"{dependent_var}_lagged" in variable_bpv_analysis:
    variable_bpv_analysis.remove(f"{dependent_var}_lagged")


# Create a DataFrame for the additional information
model_info = pd.DataFrame({
    'Model_number': [model_number] * len(model_list_BPV),
    'Dependent': [dependent_var] * len(model_list_BPV),
    'Independent': [', '.join(temp_list)] * len(model_list_BPV),  # Repeat for each row in model_list_BPV,
    'Variable BPV Analysis': variable_bpv_analysis
})

# Extract the last 5 values and their corresponding dates before RepDate
last_5_actual_values = dataset.loc[dataset['Date'] <= RepDate, ['Date', dependent_var]].tail(5)

# Create a dictionary with column names as 'Y_actual_<date>'
last_5_dict = {f'Y_act_{row.Date.strftime("%Y-%m-%d")}': [row[dependent_var]] for _, row in last_5_actual_values.iterrows()}

# Convert the dictionary into a DataFrame
last_5_df = pd.DataFrame(last_5_dict)

# Ensure that last_5_df has the same number of rows as model_list_BPV
last_5_df = pd.concat([last_5_df] * len(model_list_BPV), ignore_index=True)

# Reset the index to align correctly for concatenation
last_5_df.reset_index(drop=True, inplace=True)

# Concatenate the new DataFrame with model_list_BPV
model_list_BPV = pd.concat([model_info, last_5_df, model_list_BPV], axis=1)

# Write the updated DataFrame to Excel
excel_path = str(portfolio_abs_dir) + "\\" + f'13.Senzitivity_analysis_excel_in_{Model_number}.xlsx'
model_list_BPV.to_excel(excel_path, index=False, header=True)
print(f"Saved at: \n{excel_path}")
