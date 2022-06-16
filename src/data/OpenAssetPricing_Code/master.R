# master.R 

# You need to set the project path below before executing other scripts

# scripts below do the following
#   sets up paths (see 00_SettingsAndTools.R)
#   downloads crsp data that can't be easily shared and creates related predictors
#   creates all portfolios and saves csvs to disk
#   creates all exhibits for the paper

# the scripts require
#   user entry of pathProject in 00_SettingsAndTools.R
#   signal-firm-month csvs created by the signals code

# Most people only need to run up to 20_PredictorPorts.R and can skip any exhibits
# Exhibits are run immediately after the data required is created so
# you don't need to run every damn thing to update exhibits.

# exhibits can break if quickrun == T, and also you probably don't want
# incomplete exhibits anyway.

# I think it takes about 12 hours to run everything, and about 45 min to run up
# to 20_PredictorPorts.R

# ENVIRONMENT ####
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rm(list = ls())
# ENTER PROJECT PATH HERE (i.e. this should be the path to your local repo folder & location of SignalDoc.csv)
# if using Rstudio, pathProject = paste0(getwd(), '/') should work
pathProject = paste0(getwd(), '/src/data/OpenAssetPricing_Code')
pathThesisData = paste0(getwd(), '/data/')

quickrun =  T # use T if you want to run quickly for testing
#quickrunlist = c('Accruals','AM') # list of signals to use for quickrun
skipdaily = T # use T to skip daily CRSP which is very slow
feed.verbose = T # use T if you want lots of feedback


# setwd to folder with all R scripts for convenience
setwd(pathProject)

source('00_SettingsAndTools.R', echo=T)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# PREPARE INTERMEDIATE DATA ####

print('master: 10_DownloadCRSP.R')
tryCatch({
    source('10_DownloadCRSP.R', echo=T)
})

print('master: 11_ProcessCrsp.R')
tryCatch({
    source('11_ProcessCRSP.R', echo=T)
})

print('master: 12_CreateCRSPPredictors.R')
tryCatch({
    source('12_CreateCRSPPredictors.R', echo=T)
})

print('Downloading signal data...')
tryCatch({
  source('Download_Signals.R', echo=T)
})
