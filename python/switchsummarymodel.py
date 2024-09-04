import sys
import glob
import numpy as np
import pandas as pd
import os
import re

pd.set_option('display.max_colwidth', None)

os.chdir(sys.argv[1])
model_list_final_candidate = pd.read_excel(glob.glob('12.*')[0])
mod = int(sys.argv[2])

portfolio = re.search(r'Dataset_train_(.+?)\.xlsx', glob.glob('05.Dataset_train_*')[0]).group(1)

sumfn = glob.glob('executive_summary_*')[0]
origsum = ''
with open(sumfn) as f:
    origsum = f.read()
no_ind_var = int(re.search(r'Maximum of explanatories in the model: (\d+)', origsum).group(1))

print(origsum.split('\nFinal model is ')[0])

final_candidate_nr =('Final model is no: %s, ' %(model_list_final_candidate.loc[mod, 'Model_number']))
variables_included =('having %s as dependent variable and explanatories as follows: %s' %(model_list_final_candidate.loc[mod,'Dependent'],model_list_final_candidate.loc[mod,'Independent']))
print('\n%s' %(final_candidate_nr))
print('\n%s' %(variables_included))
print('\n')

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
print('\n%s' %(str(model_def)))
print('\n%s' %(str(table3_caption)))
print('\n%s' %(str(model_details_not_null)))
print('\n')

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
                                                                     'Corr_Pearson_test':'Independent variables correlations with Pearson',
                                                                     'BG1_test':'Serial correlation using Breusch-Godfrey test 1 lag','BG2_test':'Serial correlation using Breusch-Godfrey test 2 lag',
                                                                     'BG3_test':'Serial correlation using Breusch-Godfrey test 3 lag','BG4_test':'Serial correlation using Breusch-Godfrey test 4 lag',
                                                                     'R_DW_test':'Autocorrelation using Durbin - Watson performed in R', 
                                                                     'SW_test':'Normality of errors using Shapiro-Wilk','JB_test':'Normality of errors using with Jarque - Bera','Error_Normality_tests':'Normality of errors using SW and JB',
                                                                     'BP_test':'Heteroscedasticity using Breusch-Pagan','R_WH_test':'Heteroscedasticity using White test performed in R',
                                                                     '*.new_model_pvalue':'p-value of the model after Newey-West',
                                                                     'Intercept_new_pvalue':'p-value of the Intercept after Newey-West','R_NW_test_var1':'Significance after Newey-West for var1 performed in R',
                                                                     'R_NW_test_var2':'Significance after Newey-West for var2 performed in R','R_NW_test_var3':'Significance after Newey-West for var3 performed in R'})
print('\n%s' %(blue_intro))
print('\n%s' %(str(table4_caption)))
print('\n%s' %(str(model_details_blue_renamed[0:17])))
print('\n')


if model_list_final_blue.loc[mod,'R_NW_test_var1']!="NW not required":
    NW_results=('To correct the p-values of the estimators, we applied Newey - West approach. The significance of explanatories are as follows:')
    table5_caption = 'Table 5 Significance Summary'
    print('\n%s' %(str(table5_caption)))
    print('\n%s' %(str(NW_results)))
    print('\n%s' %(str(model_details_blue_renamed.iloc[np.r_[0:4,17:len(model_details_blue_renamed)]])))
    print('\n')
else: 
    NW_results=('Newey - West approach is not required and existing estimations are correct.')
    print('\n\n%s\n' %(str(NW_results)))
