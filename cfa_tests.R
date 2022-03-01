library(data.table); library(dplyr); library(ggplot2); library(caret) 
library(lavaan); library(semPlot)

# READ DATA AND REPLACE MISSING VALUES WITH A K-NEAREST NEIGHBOUR ESTIMATE

df <- fread("/Users/julia.ericson/HMM/data/test_data_max_long_25min.csv")  
# df <- df_all[continued == 1]
# df[, c('continued','Instructions25', 'Instructions30', 'Maths25', 'Maths30', 'OddOneOut25', 'OddOneOut30'):=NULL]

preProcValues <- preProcess(df %>% 
                              dplyr::select(Instructions1, Instructions2, Instructions3, Instructions4, Instructions5, Instructions6,
                                            Maths1, Maths2, Maths3, Maths4, Maths5, Maths6, OddOneOut1, OddOneOut2, OddOneOut3,
                                            OddOneOut4, OddOneOut5, OddOneOut6),
                            method = c("knnImpute"),
                            k = 50,
                            knnSummary = mean)

imputed_df <- predict(preProcValues, df, na.action = na.pass)

columns = names(preProcValues$mean)
for(i in columns){
  imputed_df[[i]] <-imputed_df[[i]]*preProcValues$std[i]+preProcValues$mean[i] 
}

imputed_df[, c('Instructions1', 'Instructions2', 'Instructions3', 'Instructions4', 'Instructions5', 'Instructions6')] <- (imputed_df[, c('Instructions1', 'Instructions2', 'Instructions3', 'Instructions4', 'Instructions5', 'Instructions6')] - preProcValues$mean['Instructions2']) / preProcValues$std['Instructions2']
imputed_df[, c('Maths1', 'Maths2', 'Maths3', 'Maths4', 'Maths5', 'Maths6')] <- (imputed_df[, c('Maths1', 'Maths2', 'Maths3', 'Maths4', 'Maths5', 'Maths6')] - preProcValues$mean['Maths2']) / preProcValues$std['Maths2']
imputed_df[, c('OddOneOut1', 'OddOneOut2', 'OddOneOut3', 'OddOneOut4', 'OddOneOut5', 'OddOneOut6')] <- (imputed_df[, c('OddOneOut1', 'OddOneOut2', 'OddOneOut3', 'OddOneOut4', 'OddOneOut5', 'OddOneOut6')] - preProcValues$mean['OddOneOut2']) / preProcValues$std['OddOneOut2']

#================ INVARIANCE TESTING ===========================================

config_model <- '
  # Define latent
    factor2 =~ NA*Instructions2 + lambda1*Instructions2 + Maths2 + OddOneOut2
    factor3 =~ NA*Instructions3 + lambda1*Instructions3 + Maths3 + OddOneOut3
    factor4 =~ NA*Instructions4 + lambda1*Instructions4 + Maths4 + OddOneOut4
    factor5 =~ NA*Instructions5 + lambda1*Instructions5 + Maths5 + OddOneOut5
    factor6 =~ NA*Instructions6 + lambda1*Instructions6 + Maths6 + OddOneOut6
    
  # Intercepts
    Instructions2 ~ i1*1
    Maths2 ~ 1
    OddOneOut2 ~ 1
    Instructions3 ~ i1*1
    Maths3 ~ 1
    OddOneOut3 ~ 1 
    Instructions4 ~ i1*1
    Maths4 ~ 1
    OddOneOut4 ~ 1 
    Instructions5 ~ i1*1
    Maths5 ~ 1
    OddOneOut5 ~ 1 
    Instructions6 ~ i1*1
    Maths6 ~ 1
    OddOneOut6 ~ 1 
  
  # Variances
    Instructions2 ~~ Instructions2
    Maths2 ~~ Maths2
    OddOneOut2 ~~ OddOneOut2
    Instructions3 ~~ Instructions3
    Maths3 ~~ Maths3
    OddOneOut3 ~~ OddOneOut3
    Instructions4 ~~ Instructions4
    Maths4 ~~ Maths4
    OddOneOut4 ~~ OddOneOut4
    Instructions5 ~~ Instructions5
    Maths5 ~~ Maths5
    OddOneOut5 ~~ OddOneOut5
    Instructions6 ~~ Instructions6
    Maths6 ~~ Maths6
    OddOneOut6 ~~ OddOneOut6
    
  # Correlations observations40
    Instructions2 ~~ Instructions3 + Instructions4 + Instructions5 + Instructions6
    Instructions3 ~~ Instructions4 + Instructions5 + Instructions6
    Instructions4 ~~ Instructions5 + Instructions6
    Instructions5 ~~ Instructions6
    Maths2 ~~ Maths3 + Maths4 + Maths5 + Maths6
    Maths3 ~~ Maths4 + Maths5 + Maths6
    Maths4 ~~ Maths5 + Maths6
    Maths5 ~~ Maths6
    OddOneOut2 ~~ OddOneOut3 + OddOneOut4 + OddOneOut5 + OddOneOut6
    OddOneOut3 ~~ OddOneOut4 + OddOneOut5 + OddOneOut6
    OddOneOut4 ~~ OddOneOut5 + OddOneOut6
    OddOneOut5 ~~ OddOneOut6
    
  # Latent Variable Means
    factor2 ~ 0*1
    factor3 ~ 1
    factor4 ~ 1
    factor5 ~ 1
    factor6 ~ 1
    
  # Latent variables variances and covariances
    factor2 ~~ 1*factor2
    factor3 ~~ factor3
    factor4 ~~ factor4
    factor5 ~~ factor5
    factor6 ~~ factor6
    factor2 ~~ factor3 + factor4 + factor5 + factor6
    factor3 ~~ factor4 + factor5 + factor6
    factor4 ~~ factor5 + factor6
    factor5 ~~ factor6
'

fit_config <- cfa(config_model, data = imputed_df, estimator = "MLM")
summary(fit_config, fit.measures = TRUE)

weak_model <- '
  # Define latent
    factor2 =~ NA*Instructions2 + lambda1*Instructions2 + lambda2*Maths2 + lambda3*OddOneOut2
    factor3 =~ NA*Instructions3 + lambda1*Instructions3 + lambda2*Maths3 + lambda3*OddOneOut3
    factor4 =~ NA*Instructions4 + lambda1*Instructions4 + lambda2*Maths4 + lambda3*OddOneOut4
    factor5 =~ NA*Instructions5 + lambda1*Instructions5 + lambda2*Maths5 + lambda3*OddOneOut5
    factor6 =~ NA*Instructions6 + lambda1*Instructions6 + lambda2*Maths6 + lambda3*OddOneOut6
    
  # Intercepts
    Instructions2 ~ i1*1
    Maths2 ~ 1
    OddOneOut2 ~ 1
    Instructions3 ~ i1*1
    Maths3 ~ 1
    OddOneOut3 ~ 1 
    Instructions4 ~ i1*1
    Maths4 ~ 1
    OddOneOut4 ~ 1 
    Instructions5 ~ i1*1
    Maths5 ~ 1
    OddOneOut5 ~ 1 
    Instructions6 ~ i1*1
    Maths6 ~ 1
    OddOneOut6 ~ 1 
  
  # Variances
    Instructions2 ~~ Instructions2
    Maths2 ~~ Maths2
    OddOneOut2 ~~ OddOneOut2
    Instructions3 ~~ Instructions3
    Maths3 ~~ Maths3
    OddOneOut3 ~~ OddOneOut3
    Instructions4 ~~ Instructions4
    Maths4 ~~ Maths4
    OddOneOut4 ~~ OddOneOut4
    Instructions5 ~~ Instructions5
    Maths5 ~~ Maths5
    OddOneOut5 ~~ OddOneOut5
    Instructions6 ~~ Instructions6
    Maths6 ~~ Maths6
    OddOneOut6 ~~ OddOneOut6
    
  # Correlations observations40
    Instructions2 ~~ Instructions3 + Instructions4 + Instructions5 + Instructions6
    Instructions3 ~~ Instructions4 + Instructions5 + Instructions6
    Instructions4 ~~ Instructions5 + Instructions6
    Instructions5 ~~ Instructions6
    Maths2 ~~ Maths3 + Maths4 + Maths5 + Maths6
    Maths3 ~~ Maths4 + Maths5 + Maths6
    Maths4 ~~ Maths5 + Maths6
    Maths5 ~~ Maths6
    OddOneOut2 ~~ OddOneOut3 + OddOneOut4 + OddOneOut5 + OddOneOut6
    OddOneOut3 ~~ OddOneOut4 + OddOneOut5 + OddOneOut6
    OddOneOut4 ~~ OddOneOut5 + OddOneOut6
    OddOneOut5 ~~ OddOneOut6
    
  # Latent Variable Means
    factor2 ~ 0*1
    factor3 ~ 1
    factor4 ~ 1
    factor5 ~ 1
    factor6 ~ 1
    
  # Latent variables variances and covariances
    factor2 ~~ 1*factor2
    factor3 ~~ factor3
    factor4 ~~ factor4
    factor5 ~~ factor5
    factor6 ~~ factor6
    factor2 ~~ factor3 + factor4 + factor5 + factor6
    factor3 ~~ factor4 + factor5 + factor6
    factor4 ~~ factor5 + factor6
    factor5 ~~ factor6
'

fit_weak <- cfa(weak_model, data = imputed_df, estimator = "MLM")
summary(fit_weak, fit.measures = TRUE)

strong_model <- '
  # Define latent
    factor2 =~ NA*Instructions2 + lambda1*Instructions2 + lambda2*Maths2 + lambda3*OddOneOut2
    factor3 =~ NA*Instructions3 + lambda1*Instructions3 + lambda2*Maths3 + lambda3*OddOneOut3
    factor4 =~ NA*Instructions4 + lambda1*Instructions4 + lambda2*Maths4 + lambda3*OddOneOut4
    factor5 =~ NA*Instructions5 + lambda1*Instructions5 + lambda2*Maths5 + lambda3*OddOneOut5
    factor6 =~ NA*Instructions6 + lambda1*Instructions6 + lambda2*Maths6 + lambda3*OddOneOut6
    
  # Intercepts
    Instructions2 ~ i1*1
    Maths2 ~ i2*1
    OddOneOut2 ~ i3*1
    Instructions3 ~ i1*1
    Maths3 ~ i2*1
    OddOneOut3 ~ i3*1 
    Instructions4 ~ i1*1
    Maths4 ~ i2*1
    OddOneOut4 ~ i3*1 
    Instructions5 ~ i1*1
    Maths5 ~ i2*1
    OddOneOut5 ~ i3*1 
    Instructions6 ~ i1*1
    Maths6 ~ i2*1
    OddOneOut6 ~ i3*1 
  
  # Variances
    Instructions2 ~~ Instructions2
    Maths2 ~~ Maths2
    OddOneOut2 ~~ OddOneOut2
    Instructions3 ~~ Instructions3
    Maths3 ~~ Maths3
    OddOneOut3 ~~ OddOneOut3
    Instructions4 ~~ Instructions4
    Maths4 ~~ Maths4
    OddOneOut4 ~~ OddOneOut4
    Instructions5 ~~ Instructions5
    Maths5 ~~ Maths5
    OddOneOut5 ~~ OddOneOut5
    Instructions6 ~~ Instructions6
    Maths6 ~~ Maths6
    OddOneOut6 ~~ OddOneOut6
    
  # Correlations observations40
    Instructions2 ~~ Instructions3 + Instructions4 + Instructions5 + Instructions6
    Instructions3 ~~ Instructions4 + Instructions5 + Instructions6
    Instructions4 ~~ Instructions5 + Instructions6
    Instructions5 ~~ Instructions6
    Maths2 ~~ Maths3 + Maths4 + Maths5 + Maths6
    Maths3 ~~ Maths4 + Maths5 + Maths6
    Maths4 ~~ Maths5 + Maths6
    Maths5 ~~ Maths6
    OddOneOut2 ~~ OddOneOut3 + OddOneOut4 + OddOneOut5 + OddOneOut6
    OddOneOut3 ~~ OddOneOut4 + OddOneOut5 + OddOneOut6
    OddOneOut4 ~~ OddOneOut5 + OddOneOut6
    OddOneOut5 ~~ OddOneOut6
    
  # Latent Variable Means
    factor2 ~ 0*1
    factor3 ~ 1
    factor4 ~ 1
    factor5 ~ 1
    factor6 ~ 1
    
  # Latent variables variances and covariances
    factor2 ~~ 1*factor2
    factor3 ~~ factor3
    factor4 ~~ factor4
    factor5 ~~ factor5
    factor6 ~~ factor6
    factor2 ~~ factor3 + factor4 + factor5 + factor6
    factor3 ~~ factor4 + factor5 + factor6
    factor4 ~~ factor5 + factor6
    factor5 ~~ factor6
'

fit_strong <- cfa(strong_model, data = imputed_df, estimator = "MLM")
summary(fit_strong, fit.measures = TRUE)
semPaths(fit_strong, what="est", sizeLat = 7, sizeMan = 7, edge.label.cex = .75)
