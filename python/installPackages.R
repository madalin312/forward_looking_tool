list.of.packages <- c(
    "tseries", 
    "fpp", 
    "urca", 
    "ggcorrplot", 
    "ggplot2",
    "Matrix", 
    "car", 
    "MASS", 
    "nlme", 
    "MuMIn", 
    "readxl", 
    "tidyr", 
    "broom", 
    "Hmisc", 
    "zoo", 
    "imputeTS", 
    "forecast", 
    "lmtest", 
    "sandwich", 
    "data.table", 
    "plyr", 
    "dplyr", 
    "TTR", 
    "ecm", 
    "stringr", 
    "skedastic", 
    "qdapRegex", 
    "openxlsx",
    "remotes"
)
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
options(repos=c(CRAN="https://cran.wu.ac.at/"))
if(length(new.packages)) install.packages(new.packages)

if (system.file(package='normtest') == '') {
    library(remotes)

    install_version("normtest", "1.1")
}
