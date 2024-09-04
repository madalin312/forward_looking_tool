# Import the necessary libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
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
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
from multiprocessing import Pool
from itertools import chain
import timeit
from joblib import Parallel, delayed
import glob
import scipy.stats as stats
import datetime
#pip install arch
from arch.unitroot import *
import os, shutil
import subprocess
#pip install rpy2
from rpy2 import robjects as r
import json
from wakepy import set_keepawake

set_keepawake()

pd.set_option('display.max_colwidth', None)

start = timeit.default_timer()
# set parameters

extParams = json.loads(sys.argv[1])

# 1 Manually set the working directory for input files
os.chdir(extParams['dirPath'])

# 2 The name of the file where the results will be saved 
portfolio = extParams['portfolio']

# 3 Reporting Date
RepDate = datetime.datetime.fromisoformat(extParams['reportingDate'].split('T')[0])

# 4 Set the maximum number of independent variables to be included in the linear regression
no_ind_var = extParams['independentVars']

# Set the Hypothesis for the current running
extHypo = extParams['hypo']
hypo = extHypo['name']

# 5 Set the pvalue used for model selection
# H0: Hypothesis 0
p_value_model = extHypo['p_value_model']

p_value_blue = extHypo['p_value_blue'] # this hypothesis will be changed only if the auditor requires

# 6 Number of lags to be used in modelling
no_lags = extHypo['lags']

# 7 Set the stationarity hypoyhesis for the dependent variable
# H0: Hypothesis 0
Stationarity_flag = extHypo['stationarity']
# H3: Hypothesis 3, if no model resulted from H2
#Stationarity_flag="N"

# Parameter used to establish if linear interpolated data is used; default is 0, meaning the first rows is not considered in development; options: -1,0,1
Resize_sample = extParams['resizeSample']

# eliminate models which contains GDP and Private Consumptions
check_gdp_prcons =('xGDP','xPRC')
check_prcons_gdp =('xPRC','xGDP')

# eliminate specific transformations:
elimin_trans= ['xUR_diff',
               'xUR_lag1',
               'xUR_lag1_diff',
               'xUR_lag2',
               'xUR_lag2_diff',
               'xUR_lag3',
               'xUR_lag3_diff',
               'xUR_lag4_diff',
               'xUR_lag5',
               'xUR_lag5_diff',
               'xUR_lag6',
               'xUR_lag6_diff',
               'xUR_lag7',
               'xUR_lag7_diff',
               'xUR_lag8_diff'
              ]

try:
    # Importing the dataset
    input_dir = os.getcwd() 
    print(input_dir) # verify in the console if the path of the working drectory is correct
    dataset = pd.read_excel(extParams['file']) # import the dataset from Excel
    dataset = pd.concat([pd.DataFrame(dataset['Date']),
        pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns)==portfolio]]),
        pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('x')]])], axis=1)

    if not os.path.exists(str(portfolio+"_"+hypo)):
        os.makedirs(str(portfolio+"_"+hypo)) # create a folder named after the value set for parameter portfolio and current hypothesis
    else: 
        filelist = glob.glob(os.path.join(str(portfolio+"_"+hypo), "*"))
        for f in filelist:
            os.remove(f)
    new_path = input_dir + "\\" + portfolio+"_"+hypo # path of the output
    os.chdir(str(new_path)) # change the working directory for output storage

    # Determine the oldest available DR for a specific portfolio (ALL is the DR variable)
    min_rep_date = pd.DataFrame(dataset.loc[pd.isnull(dataset[portfolio])==False,:]).Date.min()

    # Create a different dataframe for initial macro variables
    data_macro = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('x')]])

    #### Create a different dataframe for dependent variables
    data_dep = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('x')==False]])

    # Concatenate the final macro variables to be used in the model in the initial dataset
    dataset = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('x')==False]])
    dataset = pd.concat([dataset,data_macro], axis=1)

    data_macro_gdp = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('xGDP')]])
    data_macro_prcons = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('xPRCONS')]])
    data_macro_ur = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('xUR')]])
    data_macro_cpi = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('xCPI')]])
    data_macro_robor = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('xROBOR')]])
    data_macro_euribor = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('xEURIBOR')]])
    data_macro_fx = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('xFX')]])


    # Create transformation of macroeconomic series
    # YOY data
    dataset_yoy = pd.concat([data_macro_ur, data_macro_robor, data_macro_euribor, data_macro_fx], axis=1)
    data_yoy = dataset_yoy.loc[:, dataset_yoy.columns!='Date'].pct_change(periods=4)
    data_yoy = data_yoy.add_suffix('_YOY')


    # Create additional columns with lagged macro variables (lags up to value of the defined parameter no_lags)        
    for i in range (1,no_lags+1):        
        dataset[pd.Series(data_macro.add_suffix('_lag'+str(i)).columns)] = data_macro.shift(+i)

    dataset = pd.concat([dataset,data_yoy], axis=1)

    # Filling missing values before reporting date only 
    #dataset = dataset.bfill() # each missing value is replaced with the value from the next row

    # Create transformation of values
    # 1st Difference applied on all series
    data_diff = dataset.loc[:, dataset.columns!='Date'].diff(periods=1, axis=0)
    data_diff = data_diff.add_suffix('_diff')

    dataset = pd.concat([dataset,data_diff], axis=1)

    # average of four periods applied on macroeconomic variables only 
    data_macro_gdp = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('xGDP')]])
    data_macro_prcons = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('xPRCONS')]])
    data_macro_ur = pd.DataFrame(dataset.loc[:,dataset.columns[pd.Series(dataset.columns).str.startswith('xUR')]])
    dataset_avg = pd.concat([data_macro_gdp, data_macro_prcons, data_macro_ur], axis=1)

    data_avg = dataset_avg.rolling(4).mean() # each value is the result of a moving average of 4
    data_avg = data_avg.add_suffix('_avg') # add a suffix to each column name from dataframe


    #apply elimin_trans
    dataset.drop(elimin_trans, axis=1, inplace=True)

    # Difference of difference applied for dependent variables only
    data_diff2 = (data_dep.loc[:, data_dep.columns!='Date'].diff(periods=1, axis=0)).diff(periods=1, axis=0)
    data_diff2 = data_diff2.add_suffix('_diff_diff')
    
    # Relative Change of order 1 applied on dependent variables only 
    data_change = data_dep.loc[:, data_dep.columns!='Date'].pct_change(periods=1)
    data_change = data_change.add_suffix('_change')
    

    # Logit transformation: The logit function is defined as logit(p) = log(p/(1-p)). Note that logit(0) = -inf, logit(1) = inf, and logit(p) for p<0 or p>1 yields nan. applied on dependent variables only
    data_logit = pd.DataFrame(columns = pd.Series((data_dep.loc[:,data_dep.columns != 'Date']).add_suffix('_logit').columns))
    for i in range (0,len(data_logit.columns)-1):
        data_logit.iloc[:,i] = logit(data_dep.iloc[:,(i+1)]) 
    # the last column is not calculated based on the loop, so it will be computed separately
    data_logit.iloc[:,len(data_logit.columns)-1] = logit(data_dep.iloc[:,len(data_dep.columns)-1])

    data_logitdiff = data_logit.diff(periods=1, axis=0)
    data_logitdiff = data_logitdiff.add_suffix('_diff')

    # Difference of difference applied on logit transformation
    data_logitdiff2 = (data_logit.diff(periods=1, axis=0)).diff(periods=1, axis=0)
    data_logitdiff2 = data_logitdiff2.add_suffix('_diff_diff')


    # Cube root on initial dataframe applied on dependent variables only
    data_cuberoot = pd.DataFrame(columns = pd.Series((data_dep.loc[:,data_dep.columns != 'Date']).add_suffix('_cuberoot').columns))
    for i in range (0,len(data_cuberoot.columns)-1):
        data_cuberoot.iloc[:,i] = np.cbrt(data_dep.iloc[:,(i+1)]) 
    # the last column is not calculated based on the loop, so it will be computed separately
    data_cuberoot.iloc[:,len(data_cuberoot.columns)-1] = np.cbrt(data_dep.iloc[:,len(data_dep.columns)-1])

    # Difference of cube root dataframe
    data_cuberootdiff = data_cuberoot.diff(periods=1, axis=0)
    data_cuberootdiff = data_cuberootdiff.add_suffix('_diff')

    # Difference of difference applied on cube root transformation
    data_cuberootdiff2 = (data_cuberoot.diff(periods=1, axis=0)).diff(periods=1, axis=0)
    data_cuberootdiff2 = data_cuberootdiff2.add_suffix('_diff_diff')


    # Merge all dataframes containing transformed variables with the initial dataset
    dataset = pd.concat([dataset, data_diff2, data_change, data_logit, data_logitdiff, data_logitdiff2, data_cuberoot, data_cuberootdiff, data_cuberootdiff2, data_avg], axis=1)

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

    #Use updated sample in accordance with BT expectations: no linear interpolated values to be used in development sample
    dataset_train = pd.DataFrame(dataset_train.iloc[Resize_sample+1:])

    # dependent variables, all columns from dataset_train not containing macro variables
    dependent_var = pd.DataFrame(dataset_train.loc[:,dataset_train.columns[pd.Series(dataset_train.columns).str.startswith('x')==False]])

    # independent variables, all columns from dataset_train not containing PDs
    independent_var = dataset_train.drop(pd.Series(dependent_var.columns), axis=1)

    #p-values approximation for ADF according to Table 4.2, p. 103 of Banerjee et al. (1993) - adf.test method from R (tseries package)
    x=(0.01,0.025,0.05,0.1,0.9,0.95,0.975,0.99)
    adf_y25=(-4.38,-3.95,-3.6,-3.24,-1.14,-0.8,-0.5,-0.15)
    adf_y50=(-4.15,-3.8,-3.5,-3.18,-1.19,-0.87,-0.58,-0.24)
    adf_y100=(-4.04,-3.73,-3.45,-3.15,-1.22,-0.9,-0.62,-0.28)

    pp_y25=(-22.5,-19.9,-17.9,-15.6,-3.66,-2.51,-1.53,-0.43)
    pp_y50=(-25.7,-22.4,-19.8,-16.8,-3.71,-2.6,-1.66,-0.65)
    pp_y100=(-27.4,-23.6,-20.7,-17.5,-3.74,-2.62,-1.73,-0.75)

    # Calculate the STATIONARITY for INDEPENDENT variables
    # Augmented Dickey Fueller Test
    ADF_output = pd.DataFrame(columns = [0, 1, 2, 3, 4, 5, 6])
    for i in range (0,len(independent_var.columns)):
        df = pd.concat([pd.DataFrame(adfuller(independent_var.iloc[:,i], regression='ct', autolag = None, maxlag= math.trunc(pow(len(independent_var.iloc[:,i])-1,1/3))))[0:4], pd.DataFrame(list(adfuller(independent_var.iloc[:,i], regression='ct', autolag = None, maxlag= math.trunc(pow(len(independent_var.iloc[:,i])-1,1/3)))[4].items()))[1]], ignore_index=True)
        df = df.T
        ADF_output = pd.concat([ADF_output, df], ignore_index=True)
    ADF_output = ADF_output.set_axis(['ADF_Statistic','ADF_PvalueMacKinnon','ADF_lags_used','ADF_observation_used','ADF_Critical_Value_1%', 'ADF_Critical_Value_5%','ADF_Critical_Value_10%'], axis=1, inplace = False)
    ADF_output.insert(loc = 0, column = 'VAR_name', value = pd.Series(independent_var.columns))
    ADF_output.insert(loc = 2, column = 'ADF_PvalueBanerjee', value = np.interp(ADF_output['ADF_Statistic'], np.where((len(independent_var.iloc[:,i])-math.trunc(pow(len(independent_var.iloc[:,i])-1,1/3)))<=25,adf_y25,np.where((len(independent_var.iloc[:,i])-math.trunc(pow(len(independent_var.iloc[:,i])-1,1/3)))<=50,adf_y50,adf_y100)),x))
    ADF_output['ADF'] = np.where( ADF_output['ADF_PvalueBanerjee']<=p_value_blue, True, False)


    # Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test - verify the stationarity around a deterministic trend, not around a constant
    KPSS_output = pd.DataFrame(columns = [0, 1, 2, 3, 4, 5, 6])
    for i in range (0,len(independent_var.columns)):
        df = pd.concat([pd.DataFrame(kpss(independent_var.iloc[:,i], regression='ct', nlags = math.trunc(4*pow(len(independent_var.iloc[:,i])/100,0.25)))[0:3]), pd.DataFrame(list(kpss(independent_var.iloc[:,i], regression='ct', nlags = math.trunc(4*pow(len(independent_var.iloc[:,i])/100,0.25)))[3].items()))[1]], ignore_index=True)
        df = df.T
        KPSS_output = pd.concat([KPSS_output, df], ignore_index=True)
    KPSS_output = KPSS_output.set_axis(['KPSS_Statistic','KPSS_Pvalue','KPSS_lags_used','KPSS_Critical_Value_10%', 'KPSS_Critical_Value_5%','KPSS_Critical_Value_2.5%', 'KPSS_Critical_Value_1%'], axis=1, inplace = False)
    KPSS_output.insert(loc = 0, column = 'VAR_name', value = pd.Series(independent_var.columns))
    KPSS_output['KPSS'] = np.where( (KPSS_output['KPSS_Pvalue']>=0.05), True, False)

    # Philips Perron (PP) Test
    PP_output = pd.DataFrame(columns = [0, 1, 2])
    for i in range (0,len(independent_var.columns)):
        df = pd.concat([pd.DataFrame(data=PhillipsPerron(independent_var.iloc[:,i],lags=math.trunc(4*pow(len(independent_var.iloc[:,i])/100,0.25)),trend='ct',test_type='rho').stat,index=["row1"], columns=["PP_Statistic"]),
                        pd.DataFrame(data=PhillipsPerron(independent_var.iloc[:,i],lags=math.trunc(4*pow(len(independent_var.iloc[:,i])/100,0.25)),trend='ct',test_type='rho').pvalue,index=["row1"], columns=["PP_Pvalue"]),
                        pd.DataFrame(data=PhillipsPerron(independent_var.iloc[:,i],lags=math.trunc(4*pow(len(independent_var.iloc[:,i])/100,0.25)),trend='ct',test_type='rho').lags,index=["row1"], columns=["PP_Lags"])],axis=1, ignore_index=True)
        #df = df.T
        PP_output = pd.concat([PP_output, df], ignore_index=True)
    PP_output = PP_output.set_axis(['PP_Statistic','PP_PvalueMacKinnon','PP_Lags'], axis=1, inplace = False)
    PP_output.insert(loc = 0, column = 'VAR_name', value = pd.Series(independent_var.columns))
    PP_output.insert(loc = 2, column = 'PP_PvalueBanerjee', value = np.interp(PP_output['PP_Statistic'], np.where((len(independent_var.iloc[:,i])-math.trunc(pow(len(independent_var.iloc[:,i])-1,1/3)))<=25,pp_y25,np.where((len(independent_var.iloc[:,i])-math.trunc(pow(len(independent_var.iloc[:,i])-1,1/3)))<=50,pp_y50,pp_y100)),x))
    PP_output['PP'] = np.where( (PP_output['PP_PvalueBanerjee']<=p_value_blue), True, False)

    # Create a dataframe with the final status of stationarity for the independent variables
    stationarity_ind = pd.merge(ADF_output, KPSS_output, how= "inner" , on=['VAR_name','VAR_name'])
    stationarity_ind = pd.merge(stationarity_ind, PP_output, how= "inner" , on=['VAR_name','VAR_name'])
    stationarity_ind['ADF_KPSS_PP'] = np.where( (stationarity_ind['ADF']==True) | (stationarity_ind['KPSS']==True) | (stationarity_ind['PP']==True),
                                            True, False)

    stationarity_ind.to_excel (str(new_path)+"\\"+r'02.Stationarity_ind_var_'+portfolio+'_'+hypo+'.xlsx', index = False, header=True)


    # Calculate the STATIONARITY for DEPENDENT variables
    # Augmented Dickey Fueller Test
    ADF_output = pd.DataFrame(columns = [0, 1, 2, 3, 4, 5, 6])
    for i in range (0,len(dependent_var.columns)):
        df = pd.concat([pd.DataFrame(adfuller(dependent_var.iloc[:,i], regression='ct', autolag = None, maxlag= math.trunc(pow(len(dependent_var.iloc[:,i])-1,1/3))))[0:4], pd.DataFrame(list(adfuller(dependent_var.iloc[:,i], regression='ct', autolag = None, maxlag= math.trunc(pow(len(dependent_var.iloc[:,i])-1,1/3)))[4].items()))[1]], ignore_index=True)
        df = df.T
        ADF_output = pd.concat([ADF_output, df], ignore_index=True)
    ADF_output = ADF_output.set_axis(['ADF_Statistic','ADF_PvalueMacKinnon','ADF_lags_used','ADF_observation_used','ADF_Critical_Value_1%', 'ADF_Critical_Value_5%','ADF_Critical_Value_10%'], axis=1, inplace = False)
    ADF_output.insert(loc = 0, column = 'VAR_name', value = pd.Series(dependent_var.columns))
    ADF_output.insert(loc = 2, column = 'ADF_PvalueBanerjee', value = np.interp(ADF_output['ADF_Statistic'], np.where((len(independent_var.iloc[:,i])-math.trunc(pow(len(independent_var.iloc[:,i])-1,1/3)))<=25,adf_y25,np.where((len(independent_var.iloc[:,i])-math.trunc(pow(len(independent_var.iloc[:,i])-1,1/3)))<=50,adf_y50,adf_y100)),x))
    ADF_output['ADF'] = np.where(ADF_output['ADF_PvalueBanerjee']<=p_value_blue, True, False)

    # Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test - verify the stationarity around a constant
    KPSS_output = pd.DataFrame(columns = [0, 1, 2, 3, 4, 5, 6])
    for i in range (0,len(dependent_var.columns)):
        df = pd.concat([pd.DataFrame(kpss(dependent_var.iloc[:,i], regression='ct', nlags = math.trunc(4*pow(len(dependent_var.iloc[:,i])/100,0.25)))[0:3]), pd.DataFrame(list(kpss(dependent_var.iloc[:,i], regression='ct', nlags = math.trunc(4*pow(len(dependent_var.iloc[:,i])/100,0.25)))[3].items()))[1]], ignore_index=True)
        df = df.T
        KPSS_output = pd.concat([KPSS_output, df], ignore_index=True)
    KPSS_output = KPSS_output.set_axis(['KPSS_Statistic','KPSS_Pvalue','KPSS_lags_used','KPSS_Critical_Value_10%', 'KPSS_Critical_Value_5%','KPSS_Critical_Value_2.5%', 'KPSS_Critical_Value_1%'], axis=1, inplace = False)
    KPSS_output.insert(loc = 0, column = 'VAR_name', value = pd.Series(dependent_var.columns))
    KPSS_output['KPSS'] = np.where( (KPSS_output['KPSS_Pvalue']>=0.05), True, False)


    # Philipp Perion (PP) Test
    PP_output = pd.DataFrame(columns = [0, 1, 2])
    for i in range (0,len(dependent_var.columns)):
        df = pd.concat([pd.DataFrame(data=PhillipsPerron(dependent_var.iloc[:,i],lags=math.trunc(4*pow(len(dependent_var.iloc[:,i])/100,0.25)),trend='ct',test_type='rho').stat,index=["row1"], columns=["PP_Statistic"]),
                        pd.DataFrame(data=PhillipsPerron(dependent_var.iloc[:,i],lags=math.trunc(4*pow(len(dependent_var.iloc[:,i])/100,0.25)),trend='ct',test_type='rho').pvalue,index=["row1"], columns=["PP_Pvalue"]),
                        pd.DataFrame(data=PhillipsPerron(dependent_var.iloc[:,i],lags=math.trunc(4*pow(len(dependent_var.iloc[:,i])/100,0.25)),trend='ct',test_type='rho').lags,index=["row1"], columns=["PP_Lags"])],axis=1, ignore_index=True)
        #df = df.T
        PP_output = pd.concat([PP_output, df], ignore_index=True)
    PP_output = PP_output.set_axis(['PP_Statistic','PP_PvalueMacKinnon','PP_Lags'], axis=1, inplace = False)
    PP_output.insert(loc = 0, column = 'VAR_name', value = pd.Series(dependent_var.columns))
    PP_output.insert(loc = 2, column = 'PP_PvalueBanerjee', value = np.interp(PP_output['PP_Statistic'], np.where((len(independent_var.iloc[:,i])-math.trunc(pow(len(independent_var.iloc[:,i])-1,1/3)))<=25,pp_y25,np.where((len(independent_var.iloc[:,i])-math.trunc(pow(len(independent_var.iloc[:,i])-1,1/3)))<=50,pp_y50,pp_y100)),x))
    PP_output['PP'] = np.where( (PP_output['PP_PvalueBanerjee']<=p_value_blue), True, False)


    # Create a dataframe with the final status of stationarity for the dependent variables
    stationarity_dep = pd.merge(ADF_output, KPSS_output, how= "inner" , on=['VAR_name','VAR_name'])
    stationarity_dep = pd.merge(stationarity_dep, PP_output, how= "inner" , on=['VAR_name','VAR_name'])
    stationarity_dep['ADF_KPSS_PP'] = np.where( ( (Stationarity_flag=="Y") & (stationarity_dep['ADF']==True) | (stationarity_dep['KPSS']==True) | (stationarity_dep['PP']==True)),
        True,
        (np.where( ( (Stationarity_flag=="Y") & (stationarity_dep['ADF']==False) & (stationarity_dep['KPSS']==False) & (stationarity_dep['PP']==False) ),
                False,
                True)))

    stationarity_dep.to_excel (str(new_path)+"\\"+r'03.Stationarity_dep_var_'+portfolio+'_'+hypo+'.xlsx', index = False, header=True)

    # Create a dataframe with the final status of stationarity for the dependent and independent variables
    stationarity = pd.concat([stationarity_dep, stationarity_ind], axis = 0)

    stationarity.to_excel (str(new_path)+"\\"+r'04.Stationarity_'+str(portfolio+'_'+hypo)+'.xlsx', index = False, header=True)



    # Keep only the stationary variables ADF or KPSS or PP test passed in the dataset_train
    dataset_train = pd.DataFrame(dataset_train.loc[:,((stationarity.loc[stationarity['ADF_KPSS_PP']==True,:]).get('VAR_name').tolist())])


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

    #Present updated dataset in accordance with BT expectations: no linear interpolated values to be used in development sample
    dataset = pd.DataFrame(dataset.iloc[Resize_sample+1:])

    # Sort the columns from dataset alphabetically and export to excel
    dataset = dataset[sorted(dataset.columns)]
    dataset.to_excel (str(new_path)+"\\"+r'06.Dataset_'+portfolio+'_'+hypo+'.xlsx', index = False, header=True)

    # Generate ALL Models
    # Create a dataframe with the stationary dependent variables which will be used further in the loop
    df = pd.Series(dataset_train.columns[pd.Series(dataset_train.columns).str.startswith(portfolio)==True])
    df = df[df.str.endswith('lagged')==False]
    df = df.reset_index(drop=True)
    df.count()

    # Create the structure of the dataframe which will be filled in with the results of the OLS model
    model_results = pd.DataFrame(columns=list(df+'_lagged'))
    model_results[list( dataset_train.columns[(dataset_train.columns).str.startswith('x')==True])] = None
    model_results[['Model_number','Dependent','Independent','Number_of_ind_variables','Intercept','*.adjRsq','*.model_pvalue','Intercept_pvalue','logLik','AIC']]=None
    model_results[[ "*.var"+str(m+1)+"_pvalue" for m in range(no_ind_var) ]] = None


    #start = timeit.default_timer()
    model_results_list = list()



    def processInput(k):
        X_train = pd.DataFrame(data_temp.loc[:,(comb[k])]) # define the independent variable
        X_train.insert(loc=0, column='const', value=1)
        Y_train = pd.DataFrame(data_temp.iloc[:,0]) # define the dependent variable
        model = (sm.OLS(Y_train, X_train)).fit() # linear regression output
        
        while (model.rsquared_adj>0.3 and model.f_pvalue<0.25 and max(model.pvalues[1:])<0.25):
            model_coeff = pd.DataFrame(model.params).transpose() # store the model's coefficients, incl. intercept
            model_coeff = model_coeff.rename(columns={"const":"Intercept"}) # rename the first column 
            
            r_sq_adj = pd.DataFrame(pd.Series(model.rsquared_adj,index=['*.adjRsq'])).transpose() # store the R square adjusted
            
            model_pvalue = pd.DataFrame(pd.Series(model.f_pvalue,index=['*.model_pvalue'])).T # store the p value of F-stat test of the model
            
            # Rename the columns with p values for each variable, dependending on the number of variables
            model_var_pvalues = pd.DataFrame(model.pvalues).reset_index(drop=True).transpose()
            model_var_pvalues.columns = [ "*.var"+str(l)+"_pvalue" for l in range(len(model_var_pvalues.columns))]
            model_var_pvalues = model_var_pvalues.rename(columns={"*.var0_pvalue":"Intercept_pvalue"})
            
            model_log_likelyhood = pd.DataFrame(pd.Series(model.llf,index=['logLik'])).transpose() # store the likelihood of the model
            
            model_aic = pd.DataFrame(pd.Series(model.aic, index = ['AIC'])).transpose() # store the AIC value of the model
            
            model_interim_results = pd.concat([model_coeff,r_sq_adj,model_pvalue,model_var_pvalues,
                                model_log_likelyhood,model_aic], axis=1) # concatenate all the results of the model
            model_interim_results['Dependent']=list(Y_train.columns) # add the label for dependent variable Y_train
            
            model_interim_results['Independent'] = str(comb[k]) # store the independent variables included in the model
            
            model_interim_results['Model_number'] = comb.index(comb[k]) # store the number of the model from the list of all combinations
            
            model_interim_results['Number_of_ind_variables']=len(X_train.columns)-1
            
            # model_results = pd.merge(model_results, model_interim_results, on=list(model_interim_results.columns), how='outer')
            return model_interim_results


    for i in range(0,len(df)):
        # Create the data_temp with the necessary variables for running the linear regressions
        data_temp = pd.concat( [pd.DataFrame( dataset_train.loc[:,(df.iloc[i])] ), pd.DataFrame( dataset_train.loc[:,(df.iloc[i]+'_lagged')] ), 
                                pd.DataFrame(dataset_train.loc[:,dataset_train.columns[pd.Series(dataset_train.columns).str.startswith('x')==True]]) 
                                ], axis=1)
        # Count the number of groups of independent variables automatically; to be used in case of necessity, for the moment the parameter no_ind_var will be used
        #no_ind_var_groups = pd.Series(((pd.Series(dataset_train.columns[pd.Series(dataset_train.columns).str.startswith('x')==True])).str[:3]).unique())
        #no_ind_var_groups.count()
        #len(no_ind_var_groups)
        print("Variabila dependenta numarul ",i+1,"din ",len(df))
        # Create a list with all combinations of variables up until parameter no_ind_var; if the automated version will be used, meaning the number of group of main variables, then no_ind_var_groups from above will be used, instead of no_ind_var
        comb = list()
        comb_full = list()
        for j in range(1,no_ind_var+1):
            #comb_temp = list(combinations(range(1,len(data_temp.columns[1:])+1),j))
            comb_full_temp = list(combinations(data_temp.columns[1:],j))
            comb_full.extend(comb_full_temp)        
            # Run all OLS models
        
        for d in range(0,len(comb_full)):
            if ((len(set(w[0:3] for w in comb_full[d]))==len(comb_full[d])) and (check_gdp_prcons not in list(combinations(set(w[0:4] for w in comb_full[d]),2))) and (check_prcons_gdp not in list(combinations(set(w[0:4] for w in comb_full[d]),2))))==True:
                comb_temp = list(comb_full[d])
                comb.append(comb_temp)
        
        inputs = range (0,len(comb))
        try:
            model_results=pd.concat(Parallel(n_jobs=2, prefer="threads", verbose=5)(delayed(processInput)(k) for k in inputs), axis=0, ignore_index=True)
            model_results.to_csv (str(new_path)+"\\"+r'07.Model_results'+'_'+portfolio+'_'+hypo+'_'+str(i)+'.csv', index = False, header=True)
        #model_results_list.append(Parallel(n_jobs=6,verbose=5)(delayed(processInput)(k) for k in inputs))
        except  ValueError: 
            print("No models found")
        
    # setting the path for joining multiple files
    files = os.path.join(new_path, "07.Model_results*.csv")

    # list of merged files returned
    files = glob.glob(files)

    # joining files with concat and read_csv
    model_results = pd.concat(map(pd.read_csv, files), axis=0, ignore_index=True)


    # Rearrange the columns
    header_1 = list(pd.concat( [pd.DataFrame( columns = ['Model_number','Number_of_ind_variables','Dependent',
                                                        'Independent','Intercept']),
                                pd.DataFrame( columns = list(model_results.columns[(model_results.columns).str.startswith(portfolio)==True])),
                                pd.DataFrame( columns = list(model_results.columns[(model_results.columns).str.startswith('x')==True])),
                                pd.DataFrame( columns = list(model_results.columns[(model_results.columns).str.endswith('pvalue')==True])),
                            pd.DataFrame( columns = ['*.adjRsq','logLik','AIC'])
                            ]))
    model_results = model_results[header_1]

    # Export the results of the OLS models in Excel
    model_results.to_csv (str(new_path)+"\\"+r'07.Model_results'+'_'+portfolio+'_'+hypo+'.csv', index = False, header=True)


    # Model selection
    # 1. Select only the models that have R squared Adjusted >= 40% and AIC < 0
    model_list_final_subset = pd.DataFrame(model_results.loc[ ((model_results['*.adjRsq']>=0.4)),: ])

    # 2. Eliminate the linear regressions with one independent variable which is the lagged dependent variable (model_number=0)
    model_list_final_subset = pd.DataFrame(model_list_final_subset.loc[ model_list_final_subset['Model_number'] != 0, : ])

    # # 3. Eliminate the models with p-value of the entire model and for each variables > p_value_model set as parameter at the beginning of the program
    # model_list_final_subset = pd.DataFrame(model_list_final_subset.loc[ model_list_final_subset[pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.endswith('_pvalue')==True)])].max(axis=1) <= p_value_model, :] )

    # 4. Eliminate the models with the sign of coefficients not aligned with the business model
    ### ATTENTION!!! If the model includes other variables than the ones presented here, then they must be added below
    # xGDP% & PD - negative sign
    # xFX_RATES% & PD - positive sign
    # xUR% (Unemployment Rate) & PD - positive sign
    # xCPI% (Inflation Rate) & PD - positive sign
    # xHPI% (House Price Index) & PD - negative sign versus LGD
    # xIR% (Interest Rates) & PD - positive sign

    model_list_final_subset = model_list_final_subset.loc[
        (
            ~(model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xGDP')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].max(axis=1) < 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xGDP')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].min(axis=1) > 0)
            )
        ) |
        (
            (model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xGDP')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].min(axis=1) > 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xGDP')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].max(axis=1) < 0)
            )
        ) |
        ((model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xGDP')==True)]) ]).isnull().sum(axis=1) == len( pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xGDP')==True)]) ) ) 
        , :]

    model_list_final_subset = model_list_final_subset.loc[
        (
            ~(model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xPRCONS')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].max(axis=1) < 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xPRCONS')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].min(axis=1) > 0)
            )
        ) |
        (
            (model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xPRCONS')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].min(axis=1) > 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xPRCONS')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].max(axis=1) < 0)
            )
        ) |
        ((model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xPRCONS')==True)]) ]).isnull().sum(axis=1) == len( pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xPRCONS')==True)]) ) ) 
        , :]

    model_list_final_subset = model_list_final_subset.loc[
        (
            ~(model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xEUR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].min(axis=1) > 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xEUR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].max(axis=1) < 0)
            )
        ) |
        (
            (model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xEUR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].max(axis=1) < 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xEUR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].min(axis=1) > 0)
            )
        ) |
        ((model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xEUR')==True)]) ]).isnull().sum(axis=1) == len( pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xEUR')==True)]) ) ) 
        , :]

    model_list_final_subset = model_list_final_subset.loc[
        (
            ~(model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xCHF')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].min(axis=1) > 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xCHF')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].max(axis=1) < 0)
            )
        ) |
        (
            (model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xCHF')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].max(axis=1) < 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xCHF')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].min(axis=1) > 0)
            )
        ) |
        ((model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xCHF')==True)]) ]).isnull().sum(axis=1) == len( pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xCHF')==True)]) ) ) 
        , :]

    model_list_final_subset = model_list_final_subset.loc[
        (
            ~(model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xUSD')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].min(axis=1) > 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xUSD')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].max(axis=1) < 0) 
            )
        ) |
        (
            (model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xUSD')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].max(axis=1) < 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xUSD')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].min(axis=1) > 0) 
            )
        ) |
        ((model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xUSD')==True)]) ]).isnull().sum(axis=1) == len( pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xUSD')==True)]) ) ) 
        , :]

    model_list_final_subset = model_list_final_subset.loc[
        (
            ~(model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xUR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].min(axis=1) > 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xUR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].max(axis=1) < 0) 
            )
        ) |
        (
            (model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xUR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].max(axis=1) < 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xUR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].min(axis=1) > 0) 
            )
        ) |
        ((model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xUR')==True)]) ]).isnull().sum(axis=1) == len( pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xUR')==True)]) ) ) 
        , :]

    model_list_final_subset = model_list_final_subset.loc[
        (
            ~(model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xCPI')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].min(axis=1) > 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xCPI')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].max(axis=1) < 0)
            )
        ) |
        (
            (model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xCPI')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].max(axis=1) < 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xCPI')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].min(axis=1) > 0)
            )
        ) |
        ((model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xCPI')==True)]) ]).isnull().sum(axis=1) == len( pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xCPI')==True)]) ) ) 
        , :]

    model_list_final_subset = model_list_final_subset.loc[
        (
            ~(model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xHPI')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].max(axis=1) < 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xHPI')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].min(axis=1) > 0)
            )
        ) |
        (
            (model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xHPI')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].min(axis=1) > 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xHPI')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].max(axis=1) < 0)
            )
        ) |
        ((model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xHPI')==True)]) ]).isnull().sum(axis=1) == len( pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xHPI')==True)]) ) ) 
        , :]

    model_list_final_subset = model_list_final_subset.loc[
        (
            ~(model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xIR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].min(axis=1) > 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xIR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].max(axis=1) < 0)
            )
        ) |
        (
            (model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xIR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].max(axis=1) < 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xIR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].min(axis=1) > 0)
            )
        ) |
        ((model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xIR')==True)]) ]).isnull().sum(axis=1) == len( pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xIR')==True)]) ) ) 
        , :]

    model_list_final_subset = model_list_final_subset.loc[
        (
            ~(model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xROBOR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].min(axis=1) > 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xROBOR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].max(axis=1) < 0)
            )
        ) |
        (
            (model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xROBOR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].max(axis=1) < 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xROBOR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].min(axis=1) > 0)
            )
        ) |
        ((model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xROBOR')==True)]) ]).isnull().sum(axis=1) == len( pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xROBOR')==True)]) ) ) 
        , :]

    model_list_final_subset = model_list_final_subset.loc[
        (
            ~(model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xEURIBOR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].min(axis=1) > 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xEURIBOR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].max(axis=1) < 0)
            )
        ) |
        (
            (model_list_final_subset['Dependent'].str.contains("reciprocal")) &
            (
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xEURIBOR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==False)]) ].max(axis=1) < 0) |
                (model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xEURIBOR')==True) & ((model_list_final_subset.columns).str.endswith('reciprocal')==True)]) ].min(axis=1) > 0)
            )
        ) |
        ((model_list_final_subset[ pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xEURIBOR')==True)]) ]).isnull().sum(axis=1) == len( pd.Series(model_list_final_subset.columns[((model_list_final_subset.columns).str.startswith('xEURIBOR')==True)]) ) ) 
        , :]


    model_list_final_subset.to_excel (str(new_path)+"\\"+r'08.Model_list_final_subset'+'_'+portfolio+'_'+hypo+'.xlsx', index = False, header=True)


    # BLUE TESTS
    # create out of sample dataframe for calculating Y_pred
    dataset_test = pd.DataFrame(dataset.loc[dataset.Date>RepDate,:]) 

    # reset the index in order to run the for
    model_list_final_subset = model_list_final_subset.reset_index(drop=True) 

    # Store the forecasted dates
    Date = dataset_test['Date'].reset_index(drop=True)
    New_Date = []
    for date in Date:
        New_Date.append(date.strftime('%Y%m%d'))
    New_Date_prefixed = ['Y_pred_'+char for char in New_Date]

    # Store the actual dates
    Date_prev = (dataset.loc[dataset.Date<=RepDate,'Date']).reset_index(drop=True)
    Old_Date = []
    for date2 in Date_prev:
        Old_Date.append(date2.strftime('%Y%m%d'))
    Old_Date_prefixed = ['Y_act_'+char for char in Old_Date]
    Old_Date_prefixed = Old_Date_prefixed[len(Old_Date_prefixed)-5:]

    # Create a new data frame to store the results of the Blue Tests
    model_list_interim = pd.DataFrame()
    model_list_interim[['Dependent_check','Independent_check','*.intercept_VIF']] = None
    model_list_interim[[ "*.var"+str(n+1)+"_VIF" for n in range(no_ind_var) ]] = None # add columns for VIF
    model_list_interim[[ "*.comb"+str(o+1)+"_Corr_Pearson" for o in range(no_ind_var) ]] = None # add columns for Corr_Pearson
    model_list_interim[['BG1_Lagrange_Multiplier_stat','BG1_Lagrange_Multiplier_pvalue',
                    'BG1_F_test_stat','BG1_F_test_pvalue',
                    'BG2_Lagrange_Multiplier_stat','BG2_Lagrange_Multiplier_pvalue',
                    'BG2_F_test_stat','BG2_F_test_pvalue',
                    'BG3_Lagrange_Multiplier_stat','BG3_Lagrange_Multiplier_pvalue',
                    'BG3_F_test_stat','BG3_F_test_pvalue',
                    'BG4_Lagrange_Multiplier_stat','BG4_Lagrange_Multiplier_pvalue',
                    'BG4_F_test_stat','BG4_F_test_pvalue','DW_stat','SW_stat',
                    'SW_pvalue','JB_stat','JB_pvalue','BP_Lagrange_Multiplier_stat','BP_Lagrange_Multiplier_pvalue',
                    'BP_F_stat','BP_F_pvalue','WH_stat','WH_pvalue','WH_F_stat','WH_F_pvalue',
                    '*.new_model_pvalue','Intercept_new_pvalue']] = None
    model_list_interim[[ "*.var"+str(s+1)+"_new_pvalue" for s in range(no_ind_var) ]] = None
    model_list_interim[[Old_Date_prefixed]] = None
    model_list_interim[[New_Date_prefixed]] = None

    for i in range(0, len(model_list_final_subset)):
        #if i > 1:
        #    break
        temp_list = (((((((model_list_final_subset.iloc[i]['Independent']).replace('(','')).replace(',)','')).replace("'","")).replace(")","")).replace("[","")).replace("]","")).split(", ")
        # X_train = sm.add_constant(pd.DataFrame(dataset_train[temp_list])) # define the independent variable
        X_train = pd.DataFrame(dataset_train[temp_list])
        X_train.insert(loc=0, column='const', value=1) # define the independent variable
        Y_train = pd.DataFrame(dataset_train.iloc[:,dataset_train.columns==model_list_final_subset.iloc[i]['Dependent']]) # define the dependent variable
        model = (sm.OLS(Y_train, X_train)).fit() # linear regression output
        
        # Multicollinearity using VIF
        VIF_ind = pd.DataFrame([variance_inflation_factor(X_train.values, j) for j in range(len(X_train.columns))]).T
        VIF_ind.columns = [ "*.var"+str(n)+"_VIF" for n in range(len(X_train.columns))]
        VIF_ind = VIF_ind.rename(columns={"*.var0_VIF":"*.intercept_VIF"})
        
        ## Correlation with Pearson
        Corr_Pearson =[]
        comb_corr_pearson=list(combinations(pd.DataFrame(dataset_train[temp_list]).columns,2))
        for m in range(len(comb_corr_pearson)):
                        Corr_Pearson_temp = pd.DataFrame(np.corrcoef(X_train.loc[:,comb_corr_pearson[m][0]],X_train.loc[:,comb_corr_pearson[m][1]])).iloc[1,0]
                        Corr_Pearson.append(Corr_Pearson_temp)
        Corr_Pearson = pd.DataFrame(Corr_Pearson).T
        Corr_Pearson.columns = [ "*.comb"+str(n+1)+"_Corr_Pearson" for n in range(len(comb_corr_pearson))]     
            
        if(len(comb_corr_pearson)==0):    
            Corr_Pearson=pd.DataFrame({'*.comb1_Corr_Pearson':[0.0]})
            
        # Serial correlation of residuals / errors using Breusch-Godfrey test
        BG1_output = pd.DataFrame(dg.acorr_breusch_godfrey(model, nlags = 1)).T
        BG1_output = BG1_output.rename(columns={0:"BG1_Lagrange_Multiplier_stat", 1:"BG1_Lagrange_Multiplier_pvalue", 2:"BG1_F_test_stat", 3:"BG1_F_test_pvalue"})

        BG2_output = pd.DataFrame(dg.acorr_breusch_godfrey(model, nlags = 2)).T
        BG2_output = BG2_output.rename(columns={0:"BG2_Lagrange_Multiplier_stat", 1:"BG2_Lagrange_Multiplier_pvalue", 2:"BG2_F_test_stat", 3:"BG2_F_test_pvalue"})

        BG3_output = pd.DataFrame(dg.acorr_breusch_godfrey(model, nlags = 3)).T
        BG3_output = BG3_output.rename(columns={0:"BG3_Lagrange_Multiplier_stat", 1:"BG3_Lagrange_Multiplier_pvalue", 2:"BG3_F_test_stat", 3:"BG3_F_test_pvalue"})
        
        BG4_output = pd.DataFrame(dg.acorr_breusch_godfrey(model, nlags = 4)).T
        BG4_output = BG4_output.rename(columns={0:"BG4_Lagrange_Multiplier_stat", 1:"BG4_Lagrange_Multiplier_pvalue", 2:"BG4_F_test_stat", 3:"BG4_F_test_pvalue"})
        
        # Autocorrelation of residuals / errors using Durbin - Watson for lag = 1
        DW_output = pd.DataFrame(pd.Series(durbin_watson(model.resid)), columns=['DW_stat'])
        
        # Normality of residuals / errors using Shapiro-Wilk
        SW_output = pd.DataFrame(shapiro(model.resid)).T
        SW_output = SW_output.rename(columns={0:"SW_stat", 1:"SW_pvalue"})
        
        # Normality of residuals / errors using Shapiro-Wilk
        JB_output = pd.DataFrame(stats.jarque_bera(model.resid)).T
        JB_output = JB_output.rename(columns={0:"JB_stat", 1:"JB_pvalue"})
        
        # Heteroskedasticity using Breusch-Pagan
        BP_output = pd.DataFrame(sms.het_breuschpagan(model.resid, model.model.exog)).T
        BP_output = BP_output.rename(columns={0:"BP_Lagrange_Multiplier_stat", 1:"BP_Lagrange_Multiplier_pvalue", 2:"BP_F_stat", 3:"BP_F_pvalue"})

        # Heteroskedasticity using White Test
        WH_output = pd.DataFrame(sms.het_white(model.resid, model.model.exog)).T
        WH_output = WH_output.rename(columns={0:"WH_stat", 1:"WH_pvalue", 2:"WH_F_stat", 3:"WH_F_pvalue"})

        # Newey - West approach, for solving heteroskedasticity and autocorrelation in residuals (abbreviated as HAC), if the above test fail (BG, DW, BP)
        # Only the F & t statistics of the model and coeffiecients, including the pvalues are changing. The model's statistics and coefficients are not changing 
            
        new_model = model.get_robustcov_results(cov_type='HAC', maxlags=8)
        new_model_pvalue = pd.DataFrame(pd.Series(new_model.f_pvalue,index=['*.new_model_pvalue'])).T # store the p value of F-stat test of the new_model
        
        new_model_var_pvalues = pd.DataFrame(new_model.pvalues).reset_index(drop=True).transpose()
        new_model_var_pvalues.columns = [ "*.var"+str(l)+"_new_pvalue" for l in range(len(new_model_var_pvalues.columns))]
        new_model_var_pvalues = new_model_var_pvalues.rename(columns={"*.var0_new_pvalue":"Intercept_new_pvalue"})

        # Keep the most recent 5 variables from Y_train
        Y_train_prel = pd.DataFrame(Y_train.iloc[len(Y_train)-5:,:]).T
        Y_train_prel.columns = Old_Date[len(Old_Date)-5:]
        Y_train_prel = Y_train_prel.add_prefix('Y_act_')
        Y_train_prel = Y_train_prel.reset_index(drop=True)
        
        # Obtain the Y predicted, different calculations if X matrix contains the dependent variable lagged
        if (model_list_final_subset.iloc[i]['Dependent'])+'_lagged' in (model_list_final_subset.iloc[i]['Independent']):
            X_test = pd.DataFrame(dataset_test[temp_list])
            X_test.insert(loc=0, column='const', value=1) # define the independent variable
            X_test = X_test.reset_index(drop=True)
        
            Y_pred = pd.DataFrame(model.predict(X_test)).T
            for m in range(1,len(X_test)):
                X_test.iloc[m,1] = Y_pred.iloc[0,m-1]
                Y_pred = pd.DataFrame(model.predict(X_test)).T
            Y_pred.columns = New_Date
            Y_pred = Y_pred.add_prefix('Y_pred_') # define the dependent variable predicted
        else:    
            X_test = pd.DataFrame(dataset_test[temp_list])
            X_test.insert(loc=0, column='const', value=1) # define the independent variable
        
            Y_pred = pd.DataFrame(model.predict(X_test)).T
            Y_pred.columns = New_Date
            Y_pred = Y_pred.add_prefix('Y_pred_') # define the dependent variable predicted

        # Concatenate the results for statistic tests
        model_list_temp = pd.concat([VIF_ind, Corr_Pearson, BG1_output, BG2_output, BG3_output, BG4_output, DW_output, SW_output, JB_output, BP_output, WH_output, new_model_pvalue, new_model_var_pvalues, Y_train_prel, Y_pred], axis=1)
        model_list_temp['Dependent_check']=list(Y_train.columns)
        model_list_temp['Independent_check']=str(list(X_train.columns))
        
        # Append the results to model_list_interim dataframe
        model_list_interim = pd.merge(model_list_interim, model_list_temp, on=list(model_list_temp.columns), how='outer')

    model_list_interim = model_list_interim.drop(['*.intercept_VIF'], axis = 1)
    # Include columns with tests results
    # 1. VIF test only for independent variables, without testing the intercept (>=5 Pass, otherwise Fail)
    model_list_interim['VIF_test'] = np.where( (model_list_interim.loc[:,model_list_interim.columns[pd.Series(model_list_interim.columns).str.endswith('_VIF')]]).max(axis=1) <= 5,
                                            'Pass', 'Fail')

    # 2. Pearson Correlation for independent variables, without testing the intercept (<0.5 Pass, otherwise Fail)
    model_list_interim['Corr_Pearson_test'] = np.where( (abs(model_list_interim.loc[:,model_list_interim.columns[pd.Series(model_list_interim.columns).str.endswith('_Corr_Pearson')]])).max(axis=1) < 0.5,
                                            'Pass', 'Fail')

    # 3. BG_test (Breusch-Godfrey)
    model_list_interim['BG1_test'] = np.where( model_list_interim['BG1_Lagrange_Multiplier_pvalue'] > p_value_blue,
                                            'Pass', 'Fail')
    model_list_interim['BG2_test'] = np.where( model_list_interim['BG2_Lagrange_Multiplier_pvalue'] > p_value_blue,
                                            'Pass', 'Fail')
    model_list_interim['BG3_test'] = np.where( model_list_interim['BG3_Lagrange_Multiplier_pvalue'] > p_value_blue,
                                            'Pass', 'Fail')
    model_list_interim['BG4_test'] = np.where( model_list_interim['BG4_Lagrange_Multiplier_pvalue'] > p_value_blue,
                                            'Pass', 'Fail')

    # 4. DW_test (Durbin-Watson)
    model_list_interim['DW_test'] = np.where( ((model_list_interim['DW_stat'] > 1.5) &
                                            (model_list_interim['DW_stat'] < 2.5)) ,
                                            'Pass', 'Fail')

    # 5. SW_test (Shapiro-Wilk)
    model_list_interim['SW_test'] = np.where( (model_list_interim['SW_pvalue'] > p_value_blue) ,
                                            'Pass', 'Fail')

    # 6. JB_test (Jarque-Bera)
    model_list_interim['JB_test'] = np.where( (model_list_interim['JB_pvalue'] > p_value_blue) ,
                                            'Pass', 'Fail')


    # 7. Error_Normality test (Shapiro - Wilk & Jarque Bera)

    model_list_interim['Error_Normality_tests'] = np.where( (model_list_interim['JB_pvalue'] > p_value_blue)  & (model_list_interim['SW_pvalue'] > p_value_blue) ,
                                            'Pass', 'Fail')


    # 8. BP_test (Breusch-Pagan)
    model_list_interim['BP_test'] = np.where( (model_list_interim['BP_Lagrange_Multiplier_pvalue'] > p_value_blue) ,
                                            'Pass', 'Fail')

    # 9. WH_test (White's test)
    model_list_interim['WH_test'] = np.where( (model_list_interim['WH_pvalue'] > p_value_blue) ,
                                            'Pass', 'Fail')

    # 10. NW test (Newey-West)
    for d in range (1, no_ind_var+1):
        model_list_interim['NW_test_var' + str(d)] = np.where( (model_list_interim['BG1_test']=='Pass') & (model_list_interim['BG2_test']=='Pass') & (model_list_interim['BG3_test']=='Pass') & (model_list_interim['BG4_test']=='Pass') & (model_list_interim['DW_test']=='Pass') & (model_list_interim['BP_test']=='Pass') & (model_list_interim['WH_test']=='Pass'),'NW not required',
                                                            np.where ( (pd.notnull(model_list_interim['*.var'+ str(d) +'_new_pvalue'])) & (model_list_interim['*.var'+ str(d) +'_new_pvalue'] <= p_value_model), 'Significant',
                                                                        np.where ( ((pd.notnull(model_list_interim['*.var'+ str(d) +'_new_pvalue'])) & (model_list_interim['*.var'+ str(d) +'_new_pvalue'] > p_value_model)), 'Not significant', 'Not applicable'
                                                                                )
                                                                        )
                                                            )
        

    # Concatenate the initial results of the selected models with the 2nd round of results from Blue Tests (for the same models)
    model_list_final = pd.concat([model_list_final_subset, model_list_interim], axis=1)

    # Order the columns from the dataset model_list_final
    header_2 = list(pd.concat( [pd.DataFrame( columns = ['Model_number','Number_of_ind_variables','Dependent',
                                        'Dependent_check','Independent','Independent_check','Intercept']),
                            pd.DataFrame( columns = list(model_list_final.columns[(model_list_final.columns).str.startswith(portfolio)==True])),
                            pd.DataFrame( columns = list(model_list_final.columns[(model_list_final.columns).str.startswith('x')==True])),
                            pd.DataFrame( columns = ['*.adjRsq','*.model_pvalue','Intercept_pvalue'] ),
                            pd.DataFrame( columns = [ "*.var"+str(l)+"_pvalue" for l in range(1,no_ind_var+1)]),
                            pd.DataFrame( columns = ['logLik','AIC']),
                            pd.DataFrame( columns = [ "*.var"+str(l)+"_VIF" for l in range(1,no_ind_var+1)]),
                            pd.DataFrame( columns = ['VIF_test']),
                            pd.DataFrame( columns = [ "*.comb"+str(l)+"_Corr_Pearson" for l in range(1,no_ind_var+1)]),
                            pd.DataFrame( columns = ['Corr_Pearson_test','BG1_Lagrange_Multiplier_stat','BG1_Lagrange_Multiplier_pvalue',
                                        'BG1_F_test_stat','BG1_F_test_pvalue','BG1_test',
                                        'BG2_Lagrange_Multiplier_stat','BG2_Lagrange_Multiplier_pvalue',
                                        'BG2_F_test_stat','BG2_F_test_pvalue','BG2_test',
                                        'BG3_Lagrange_Multiplier_stat','BG3_Lagrange_Multiplier_pvalue',
                                        'BG3_F_test_stat','BG3_F_test_pvalue','BG3_test',
                                        'BG4_Lagrange_Multiplier_stat','BG4_Lagrange_Multiplier_pvalue',
                                        'BG4_F_test_stat','BG4_F_test_pvalue','BG4_test',
                                        'DW_stat','DW_test','SW_stat', 'SW_pvalue','SW_test',
                                        'JB_stat', 'JB_pvalue','JB_test', 'Error_Normality_tests',
                                        'BP_Lagrange_Multiplier_stat','BP_Lagrange_Multiplier_pvalue',
                                        'BP_F_stat','BP_F_pvalue','BP_test',
                                        'WH_stat','WH_pvalue',
                                        'WH_F_stat','WH_F_pvalue','WH_test','*.new_model_pvalue','Intercept_new_pvalue']),
                            pd.DataFrame( columns = [ "*.var"+str(l)+"_new_pvalue" for l in range(1,no_ind_var+1)]),
                            pd.DataFrame( columns = [ "NW_test_var"+str(l) for l in range(1,no_ind_var+1)]),
                                                    pd.DataFrame( columns = list(model_list_final.columns[(model_list_final.columns).str.startswith('Y_act')==True])),
                            pd.DataFrame( columns = list(model_list_final.columns[(model_list_final.columns).str.startswith('Y_pred')==True]))
                            ]))
    model_list_final = model_list_final[header_2]

    # Delete all columns that have only missing values
    model_list_final = model_list_final.dropna(axis='columns', how='all')

    # Delete other unnecessary columns
    # model_list_final = model_list_final.drop('*.intercept_VIF', axis=1)


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
        return Y_orig


    # Calculate Y_orig
    Y_orig_df = pd.DataFrame()
    for i in range (0, len(model_list_final)):
        Y = pd.DataFrame( model_list_final.loc[i, list(model_list_final.columns[(model_list_final.columns).str.startswith('Y')==True])] )
        dependent_var_name = model_list_final.loc[i,'Dependent']
        Y_orig = reverse_y_pred(dependent_var_name, Y)  
        Y_orig = pd.DataFrame(Y_orig.iloc[5:,-1]).T
        Y_orig.columns = New_Date
        Y_orig = Y_orig.add_prefix('Y_orig_')
        Y_orig = Y_orig.reset_index(drop=True)
        Y_orig_df = pd.concat([Y_orig_df,Y_orig], axis=0)
    Y_orig_df = Y_orig_df.reset_index(drop=True)

    # Add the Y_orig to the model_list_final dataframe
    model_list_final = pd.concat([model_list_final, Y_orig_df], axis = 1)

    # Calculate the min and max of Y_orig for each model and test if it fits into interval [0,1]
    model_list_final['Min_Y_orig'] = model_list_final[ pd.Series(model_list_final.columns[((model_list_final.columns).str.startswith('Y_orig')==True)]) ].min(axis=1)
    model_list_final['Max_Y_orig'] = model_list_final[ pd.Series(model_list_final.columns[((model_list_final.columns).str.startswith('Y_orig')==True)]) ].max(axis=1)
    try: 
        model_list_final['Avg_Y_orig'] = np.average(model_list_final[ pd.Series(model_list_final.columns[((model_list_final.columns).str.startswith('Y_orig')==True)])[0:4] ],axis=1)
    except ZeroDivisionError: 
        0
    model_list_final['Y_pred_check'] = np.where( ((model_list_final['Min_Y_orig'] >= 0) & (model_list_final['Max_Y_orig'] <= 1)),
                                                'OK', 'NOK')

    model_list_final.to_excel (str(new_path)+"\\"+r'09.Model_list_final'+'_'+portfolio+'_'+hypo+'.xlsx', index = False, header=True)


    ## integration with "R - language" for several BLUE tests
    os.chdir(input_dir)
    r.r.source(os.path.join(os.path.dirname(__file__), 'R_NW_WH.R'))
    os.chdir(str(new_path)) 
    output = r.r['R_BlueTests'](portfolio, hypo, p_value_model, p_value_blue, no_ind_var,os.getcwd())

    # read the results after computing the Blue tests in R
    model_list_final_candidate = pd.read_excel(str(new_path)+"\\"+r'11.Final_Results'+'_'+portfolio+'_'+hypo+'.xlsx')

    ###### Apply criteria for selecting the feasible candidates

    ## VIF Test
    model_list_final_candidate = pd.DataFrame(model_list_final_candidate.loc[ ((model_list_final_candidate['VIF_test']=='Pass')),: ])

    ## Corr_Pearson_test 
    model_list_final_candidate = pd.DataFrame(model_list_final_candidate.loc[ ((model_list_final_candidate['Corr_Pearson_test']=='Pass')),: ])

    ## Error_Normality_tests 
    model_list_final_candidate = pd.DataFrame(model_list_final_candidate.loc[ ((model_list_final_candidate['Error_Normality_tests']=='Pass')),: ])

    ## Predictions should be between 0% and 100%
    model_list_final_candidate = pd.DataFrame(model_list_final_candidate.loc[ ((model_list_final_candidate['Y_pred_check']=='OK')),: ])

    ## Newey-West significance
    #var1
    model_list_final_candidate = pd.DataFrame(model_list_final_candidate.loc[ ((model_list_final_candidate['R_NW_test_var1']!='Not significant')),: ])

    #var2
    model_list_final_candidate = pd.DataFrame(model_list_final_candidate.loc[ ((model_list_final_candidate['R_NW_test_var2']!='Not significant')),: ])

    #var3
    model_list_final_candidate = pd.DataFrame(model_list_final_candidate.loc[ ((model_list_final_candidate['R_NW_test_var3']!='Not significant')),: ])


    # sort by AICc
    model_list_final_candidate = model_list_final_candidate.sort_values("R_AICc")

    #print the new file including feasible candidate models
    model_list_final_candidate.to_excel (str(new_path)+"\\"+r'12.Final_feasible_candidates'+'_'+portfolio+'_'+hypo+'.xlsx', index = False, header=True)

    # Executive Summary
    mod=0
    #f = open(str(('executive_summary_model_'+str(model_list_final.loc[mod,'Model_number'])+'.txt')),'w')

    model_list_final_candidate.reset_index(drop=True, inplace=True)

    f = None

    try:
        f = open(str(('executive_summary_model_%s.txt' %(model_list_final_candidate.loc[mod,'Model_number']))),'w')
    except KeyError:
        print('No models available to generate executive summary')
        exit()
        
    portfolio_print =('FWL model development for portfolio: %s using quarterly data between %s and %s' %(portfolio, str(min_rep_date.strftime('%Y-%m-%d')), str(RepDate.strftime('%Y-%m-%d'))))
    f.write('\n%s' %(portfolio_print))
    f.write('\n')

    nr_models = len(comb_full)*len(df)
    nr_models_total=('The number of models generated is: %d for a number of %d dependent variables, ' %(nr_models, len(df)))
    f.write('\n%s' %(nr_models_total))
    
    original_independent_vars = ('the following independent variables: **%s' %(extParams['modelVars'])) + '**'
    f.write('\n%s' %(original_independent_vars))

    transf_list_dep = []
    for i in range(1,len(dependent_var.columns)):
        if((dependent_var.columns[i].rfind('_')>portfolio.rfind('_')) and (dependent_var.columns[i].find('lag')==-1)):
            transf_list_dep_temp=str('data_' + dependent_var.columns[i][-(len(dependent_var.columns[i].partition(portfolio)[2]))+1:])
            transf_list_dep.append(transf_list_dep_temp)

    transf_list_dep_unique=list(set(transf_list_dep))

    transf_list_indep = []
    for i in range(1,len(independent_var.columns)):
        if((independent_var.columns[i].rfind('_')>-1) and (independent_var.columns[i].find('lag')==-1)):
            transf_list_indep_temp=str('data_' + independent_var.columns[i][-(len(independent_var.columns[i])-independent_var.columns[i].find('_'))+1:])
            transf_list_indep.append(transf_list_indep_temp)

    transf_list_indep_unique=list(set(transf_list_indep))

    transf =('using the following transformations:') 
    transf_dep=('* for dependent variable: data, %s and ' %(transf_list_dep_unique))
    transf_indep =('* for independent variable: data, %s and' %(transf_list_indep_unique))

    if(Stationarity_flag=="Y"): 
            Stationarity=('Stationary series used')
    else:   
            Stationarity=('NON Stationary series used')
    hypo_print = ('hypotheses: Maximum of explanatories in the model: %d, the p-value of model and explanatories: %s, number of lags used: %d, and %s' %(no_ind_var, p_value_model,no_lags,Stationarity))
    f.write('\n%s' %(transf))
    f.write('\n%s' %(transf_dep))
    f.write('\n%s' %(transf_indep))
    f.write('\n%s' %(hypo_print))
    f.write('\n')

    nr_models_non_macro_duplic = len(comb)*len(df)
    eliminate_macro=('out of which %s for a number of %d dependent variables,have: ' %(nr_models_non_macro_duplic,len(df)))
    criteria_macro1=('* no explanatories derived from a single macroeconomic variable')
    criteria_macro2=('* no combination of transformations for GDP and Private consumption within the same model')
    f.write('\n%s' %(eliminate_macro))
    f.write('\n%s' %(criteria_macro1))
    f.write('\n%s' %(criteria_macro2))

    f.write('\n')
    nr_candidates = ('out of which %d have: ' %(len(model_results)))
    criteria_stats =('* Adjusted R squared greater than 0.4;\n' + '* Model p-value is lower than 0.25;\n' + '* explanatories p-values are lower than 0.25')
    f.write('\n%s' %(nr_candidates))
    f.write('\n%s' %(criteria_stats))

    f.write('\n')
    nr_candidates_final = ('out of which %d have: ' %(len(model_list_final_candidate)))
    criteria_stats_final=('* Adjusted R square greater or equal to 0.4; \n* Model p-value is lower or equal to %s \n* explanatories p-values are lower or equal to %s;' %(p_value_model, p_value_model))
    criteria_bus_final=('* Macroeconomic meaning is identified (signs of macroeconomic variables)')
    f.write('\n%s' %(nr_candidates_final))
    f.write('\n%s' %(criteria_stats_final))
    f.write('\n%s' %(criteria_bus_final))
    f.write('\n')

    nr_candidates_final_print=('Number of final candidate models is: %d' %(len(model_list_final_candidate)))
    f.write('\n%s' %(nr_candidates_final_print))
    f.write('\n')


    selection_process1=('Furthermore, model selection is based on: ')
    selection_process2=('* ordering the candidate models ascending by AIC indicator')
    selection_process3=('* ordering the candidate models descending by Adjusted R-squared')
    selection_process4=('* existing of positive estimated PDs')
    selection_process5=('* existing of estimated PDs lower than 100%')
    selection_process6=('* passing VIF test')
    selection_process7=('* passing Correlation with Pearson test')
    selection_process8=('* passing Normality using SW and JB tests')
    selection_process9=('* having statistical significance after Newey-West approach, if applicable')
    f.write('\n%s' %(selection_process1))
    f.write('\n%s' %(selection_process2))
    f.write('\n%s' %(selection_process3))
    f.write('\n%s' %(selection_process4))
    f.write('\n%s' %(selection_process5))
    f.write('\n%s' %(selection_process6))
    f.write('\n%s' %(selection_process7))
    f.write('\n%s' %(selection_process8))
    f.write('\n%s' %(selection_process9))
    f.write('\n')


    final_candidate_nr =('Final model is no: %s, ' %(model_list_final_candidate.loc[mod,'Model_number']))
    variables_included =('having %s as dependent variable and explanatories as follows: %s' %(model_list_final_candidate.loc[mod,'Dependent'],model_list_final_candidate.loc[mod,'Independent']))
    f.write('\n%s' %(final_candidate_nr))
    f.write('\n%s' %(variables_included))
    f.write('\n')

    model_def=('The definition of the model is presented below:')
    table3_caption = 'Table 3 Model Definition Summary'
    header_3 = list(pd.concat( [pd.DataFrame( columns = ['Model_number','Number_of_ind_variables','Dependent',
                                                        'Independent','Intercept']),
                            pd.DataFrame( columns = list(model_list_final_candidate.columns[(model_list_final_candidate.columns).str.startswith(portfolio)==True])),
                            pd.DataFrame( columns = list(model_list_final_candidate.columns[(model_list_final_candidate.columns).str.startswith('x')==True])),
                            pd.DataFrame( columns = ['*.model_pvalue','Intercept_pvalue'] ),
                            pd.DataFrame( columns = [ "*.var"+str(l)+"_pvalue" for l in range(1,no_ind_var+1)]),
                            pd.DataFrame( columns = ['*.adjRsq','R_AICc'])
                            ]))
    model_list_final_selected = model_list_final_candidate[header_3]
    model_details = pd.DataFrame(model_list_final_selected.loc[mod,:])
    model_details.columns= ['Info']
    model_details_not_null=model_details.dropna(axis=0, how='all')
    f.write('\n%s' %(str(model_def)))
    f.write('\n%s' %(str(table3_caption)))
    f.write('\n%s' %(str(model_details_not_null)))
    f.write('\n')

    blue_intro=('\nBLUE tests applied on this model reflect the following: ')

    table4_caption = 'Table 4 BLUE Tests Summary'
    header_4 = list(pd.concat( [pd.DataFrame( columns = ['Model_number','Number_of_ind_variables','Dependent',
                                                        'Independent','Intercept']),
                            pd.DataFrame( columns = ['VIF_test',
                                                    'Corr_Pearson_test',
                                                    'BG1_test',
                                                    'BG2_test',
                                                    'BG3_test',
                                                    'BG4_test',
                                                    'R_DW_test',
                                                    'SW_test',
                                                    'JB_test',
                                                    'Error_Normality_tests',
                                                    'BP_test',
                                                    'R_WH_test',
                                                    '*.new_model_pvalue','Intercept_new_pvalue']),
                            pd.DataFrame( columns = [ "R_NW_test_var"+str(l) for l in range(1,no_ind_var+1)]),
                            ]))
    model_list_final_blue = model_list_final_candidate[header_4]
    model_details_blue = pd.DataFrame(model_list_final_blue.loc[mod,:])
    model_details_blue.columns= ['Info']
    model_details_not_null_blue=model_details_blue.dropna(axis=0, how='all')

    model_details_blue_renamed=model_details_not_null_blue.rename(index={'VIF_test':'Multicolinnearity using VIF',
                                                                        'Corr_Pearson':'Independent variables correlations with Pearson',
                                                                        'BG1_test':'Serial correlation using Breusch-Godfrey test 1 lag','BG2_test':'Serial correlation using Breusch-Godfrey test 2 lag',
                                                                        'BG3_test':'Serial correlation using Breusch-Godfrey test 3 lag','BG4_test':'Serial correlation using Breusch-Godfrey test 4 lag',
                                                                        'R_DW_test':'Autocorrelation using Durbin - Watson performed in R', 
                                                                        'SW_test':'Normality of errors using Shapiro-Wilk','JB_test':'Normality of errors using with Jarque - Bera', 'Error_Normality_tests':'Normality of errors using SW and JB',
                                                                        'BP_test':'Heteroscedasticity using Breusch-Pagan','R_WH_test':'Heteroscedasticity using White test performed in R',
                                                                        '*.new_model_pvalue':'p-value of the model after Newey-West',
                                                                        'Intercept_new_pvalue':'p-value of the Intercept after Newey-West','R_NW_test_var1':'Significance after Newey-West for var1 performed in R',
                                                                        'R_NW_test_var2':'Significance after Newey-West for var2 performed in R','R_NW_test_var3':'Significance after Newey-West for var3 performed in R'})
    f.write('\n%s' %(blue_intro))
    f.write('\n%s' %(str(table4_caption)))
    f.write('\n%s' %(str(model_details_blue_renamed[0:17])))
    f.write('\n')

    if model_list_final_blue.loc[mod,'R_NW_test_var1']!="NW not required":
        NW_results=('To correct the p-values of the estimators, we applied Newey - West approach. The significance of explanatories are as follows:')
        table5_caption = 'Table 5 Significance Summary'
        f.write('\n%s' %(str(table5_caption)))
        f.write('\n%s' %(str(NW_results)))
        f.write('\n%s' %(str(model_details_blue_renamed.iloc[np.r_[0:4,17:len(model_details_blue_renamed)]])))
        f.write('\n')
    else: 
        NW_results=('Newey - West approach is not required and existing estimations are correct.')
        f.write('\n\n%s\n' %(str(NW_results)))

    f.close()

    stop = timeit.default_timer()

except Exception as e:
    print("There was a problem processing the portfolio:\n")
    print(str(e))