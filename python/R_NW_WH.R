R_BlueTests <- function  (portfolio, hypo, p_value_model, p_value_blue, no_ind_var,location)
{
  ## load required libraries
  library('tseries')
  library('fpp');
  library('urca')
  library('ggcorrplot')
  library('ggplot2')
  library(normtest)
  library('Matrix')
  library('car')
  library('MASS')
  library('nlme')
  library('MuMIn')
  library(readxl)
  library(tidyr)
  library(broom)
  library(Hmisc)
  library(zoo)
  library(imputeTS)
  library(forecast)
  library(lmtest)
  library(sandwich)
  library(data.table)
  library(plyr)
  library(dplyr)
  library(TTR)
  library(ecm)
  library(stringr)
  library(skedastic)
  library(qdapRegex)
  library(openxlsx)
  
  
  durbinH <- function(model, ylag1var){
    d <- car::durbinWatsonTest(model)
    n <- length(model$fitted.values) + 1
    v <- summary(model)$coef[which(row.names(summary(model)$coef)==ylag1var),2]^2
    durbinH <- (1 - 0.5 * d$dw) * sqrt(abs(n / (1 - n*v)))
    return(durbinH)
  }
  
  ###set wd and import data
  #rm(list = ls())
  setwd(location)
  options(scipen=999) # remove scientific notation in printing the data
  p_value_model<-as.numeric(p_value_model)
  p_value_blue<-as.numeric(p_value_blue)
  
  ## correctly read the data
  nms<-names(read_excel(paste("09.Model_list_final_",portfolio,"_",hypo,".xlsx",sep="")))
  ct<-ifelse((grepl("_lagged", nms) | grepl("x", nms)), "numeric", "guess")
  
  Model_list_final<- read_excel(paste("09.Model_list_final_",portfolio,"_",hypo,".xlsx",sep=""),col_types = ct)
  Dataset<- read_excel(paste("05.Dataset_train_",portfolio,".xlsx",sep=""))
  
  
  for (i in 1:nrow(Model_list_final))
  {
    temp_list_indep <-c(qdapRegex::ex_between(Model_list_final$Independent[i], "'", "'"))
    temp_list_dep <- c(Model_list_final$Dependent[i])
    data_temp <- Dataset %>% select(temp_list_dep,temp_list_indep)
    model = lm(data_temp)
    
    ##BLUE TESTS
    Model_list_final$R_AICc[i] <-AICc(model)
    Model_list_final$R_DW_stat[i] <- ifelse(grepl("_lagged", temp_list_indep)[1],durbinH(model,names(coef(model))[2]),as.numeric(dwtest(model)$stat))
    Model_list_final$R_DW_pvalue[i] <- ifelse(grepl("_lagged", temp_list_indep)[1],"Not applicable",dwtest(model)$p)
    Model_list_final$R_DW_test[i] <- ifelse(grepl("_lagged", temp_list_indep[1]),ifelse(Model_list_final$R_DW_stat[i]<=1.96,"Pass","Fail"),ifelse((Model_list_final$R_DW_stat[i]>1.5 & Model_list_final$R_DW_stat[i]<2.5),"Pass","Fail"))
    Model_list_final$R_WH_stat[i] <-white(model)$statistic
    Model_list_final$R_WH_pvalue[i] <-white(model)$p.value
    Model_list_final$R_WH_test[i] <- ifelse(Model_list_final$R_WH_pvalue[i]>p_value_blue,"Pass","Fail")
    temp.summ <- summary(model)
    temp.summ$coefficients <- unclass(coeftest(model, vcov. = NeweyWest(model),df=Inf))
    Model_list_final$R_Intercept_new_pvalue[i]<-temp.summ$coefficients[,"Pr(>|z|)"][1]
    Model_list_final$R_var1_new_pvalue[i]<-temp.summ$coefficients[,"Pr(>|z|)"][2]
    Model_list_final$R_var2_new_pvalue[i]<-temp.summ$coefficients[,"Pr(>|z|)"][3]
    Model_list_final$R_var3_new_pvalue[i]<-temp.summ$coefficients[,"Pr(>|z|)"][4]
    if(Model_list_final$BG1_test[i]=="Pass"& Model_list_final$BG2_test[i]=="Pass"& Model_list_final$BG3_test[i]=="Pass"&Model_list_final$BG4_test[i]=="Pass"&
       Model_list_final$BP_test[i]=="Pass"& Model_list_final$R_DW_test[i]=="Pass" &Model_list_final$R_WH_test[i]=="Pass")
    {
      Model_list_final$R_NW_test_var1[i]<-"NW not required"
      Model_list_final$R_NW_test_var2[i]<-"NW not required"
      Model_list_final$R_NW_test_var3[i]<-"NW not required"
    } else 
    {
      if(is.na(Model_list_final$R_var1_new_pvalue[i]))
      {
        Model_list_final$R_NW_test_var1[i]<-"Not applicable"
      } else ifelse(Model_list_final$R_var1_new_pvalue[i]<=p_value_blue, Model_list_final$R_NW_test_var1[i]<-"Significant", Model_list_final$R_NW_test_var1[i]<-"Not significant")
      
      if(is.na(Model_list_final$R_var2_new_pvalue[i]))
      {
        Model_list_final$R_NW_test_var2[i]<-"Not applicable"
      } else ifelse(Model_list_final$R_var2_new_pvalue[i]<=p_value_blue, Model_list_final$R_NW_test_var2[i]<-"Significant", Model_list_final$R_NW_test_var2[i]<-"Not significant")
      if(is.na(Model_list_final$R_var3_new_pvalue[i]))
      {
        Model_list_final$R_NW_test_var3[i]<-"Not applicable"
      } else ifelse(Model_list_final$R_var3_new_pvalue[i]<=p_value_blue, Model_list_final$R_NW_test_var3[i]<-"Significant", Model_list_final$R_NW_test_var3[i]<-"Not significant")
    }
    
  } 
  
  
  # Model_list_final_after_pvalue <- Model_list_final %>%
  #                         filter(Model_list_final$R_NW_var1_test=="NW not required" & 
  #                               Model_list_final$`*.model_pvalue`<p_value & 
  #                               Model_list_final$Intercept_pvalue<p_value & 
  #                               Model_list_final$`*.var1_pvalue`<p_value &
  #                               (Model_list_final$`*.var2_pvalue` <p_value |is.na(Model_list_final$`*.var2_pvalue`))&
  #                               (Model_list_final$`*.var3_pvalue` <p_value | is.na(Model_list_final$`*.var3_pvalue`)) |
  #                               Model_list_final$R_NW_var1_test!="NW not required") 
  
  if (no_ind_var==3)
  {
  Model_list_final_after_pvalue <- Model_list_final %>%
    filter(Model_list_final$R_NW_test_var1 =="NW not required" & 
             pmax(Model_list_final$`*.model_pvalue`, 
                  Model_list_final$Intercept_pvalue, 
                  Model_list_final$`*.var1_pvalue`,
                  Model_list_final$`*.var2_pvalue`,
                  Model_list_final$`*.var3_pvalue`, 
                  na.rm = TRUE) <= p_value_model |
             Model_list_final$R_NW_test_var1!="NW not required") 
  } else 
  {
    Model_list_final_after_pvalue <- Model_list_final %>%
      filter(Model_list_final$R_NW_test_var1 =="NW not required" & 
               pmax(Model_list_final$`*.model_pvalue`, 
                    Model_list_final$Intercept_pvalue, 
                    Model_list_final$`*.var1_pvalue`,
                    Model_list_final$`*.var2_pvalue`,
                    #Model_list_final$`*.var3_pvalue`, 
                    na.rm = TRUE) <= p_value_model |
               Model_list_final$R_NW_test_var1!="NW not required") 
  }
  
  openxlsx::write.xlsx(Model_list_final, file=paste("10.Results_Blue_Tests_",portfolio,"_",hypo,".xlsx",sep=""))
  openxlsx::write.xlsx(Model_list_final_after_pvalue, file=paste("11.Final_Results_",portfolio,"_",hypo,".xlsx",sep=""))
  openxlsx::write.xlsx(Model_list_final_after_pvalue, file=paste("11.Final_Results_",portfolio,"_",hypo,"_MACRO.xlsx",sep=""))
  
}




# portfolio <-'Hipotecario'
# hypo <-'H0'
# p_value_model <- 0.1
# p_value_blue <- 0.1
# no_ind_var <- 3
# location <- 'C:\\FWL Models\\External POC\\EY_Spain\\LDP_H0'
