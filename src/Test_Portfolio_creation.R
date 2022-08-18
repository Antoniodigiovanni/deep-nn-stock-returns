library(tidyverse)
rm(list = ls())
file1 = "~/thesis/saved/results/GuGridSearchResults/predicted_returns/20220731-02_11:08:473504 - predicted_returns.csv"
df <- read.csv(file1)
df$...1 <- NULL


FF <- "~/thesis/data/external/F-F_Research_Data_5_Factors_2x3.csv"
headers <- read.csv(file = FF, skip = 2,nrows = 1, as.is=T, header=F)
headers[1,1] <- 'yyyymm'
FF <- read.csv(FF, skip = 4, nrows = 706, header=F)
colnames(FF) = headers

MOM_path <-'~/thesis/data/external/F-F_Momentum_Factor.CSV'
headers <- read.csv(file = MOM_path, skip = 12,nrows = 1, as.is=T, header=F)
headers[1,1] <- 'yyyymm'
MOM <- read.csv(MOM_path, skip=12, nrows=1145)#, skip = 4, nrows = 706, header=F)
colnames(MOM) <- headers

rm(list = c('headers', 'file1', MOM_path)) 

FF <- FF %>% left_join(MOM, by='yyyymm')
FF$RF <- NULL

df = df %>% 
  group_by(yyyymm) %>% 
  mutate(quantile = ntile(predicted_ret, 10))

crspmret <- read.csv('~/thesis/data/external/crspmret.csv')

df <- left_join(df, crspmret, by = c('permno','yyyymm'))
df$date <- NULL

#df2= df2 %>% 
#  group_by(yyyymm) %>% 
#  mutate(quantile_real = ntile(ret, 10))

df <- df %>% group_by(yyyymm, quantile) %>% 
  mutate(
    pweight = melag/sum(melag)
  )

port_wide <- df %>% group_by(yyyymm, quantile) %>% 
  mutate(preturn = ret * pweight) %>% 
  summarise(preturn = sum(preturn)) %>% 
  spread(quantile, preturn)


port_ret_python <- read.csv('~/thesis/saved/results/GuGridSearchResults/portfolio_returns/20220731-02_11:08:473504 - portfolio_returns.csv')

port_wide[-1] <-port_wide[-1]/100 
port_ret_python$X <- NULL

colnames(port_ret_python) <- colnames(port_wide)
colnames(port_ret_python)[12] <- '10-1'

port_ret_python$`10-1`<- NULL

round(port_wide$`1`, digits = 3)

test <- port_ret_python - port_wide
test <- round(test, digits=3)

# Calculating Long-short and running regression
port_wide$'10-1' <- port_wide$`10` - port_wide$`1`

portfolio_ret <- port_wide %>% select(yyyymm, '10')

portfolio_ret <- left_join(portfolio_ret, FF, by = 'yyyymm')
portfolio_ret$`10` <- portfolio_ret$`10`*100

model <- lm(`10` ~ `Mkt-RF` + SMB + HML + RMW + CMA + `Mom   `, data = portfolio_ret)
summary(model)


FF_portfolio <- 