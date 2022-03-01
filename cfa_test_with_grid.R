library(data.table); library(dplyr); library(ggplot2); library(caret) 
library(lavaan); library(semPlot)

# READ DATA AND REPLACE MISSING VALUES WITH A K-NEAREST NEIGHBOUR ESTIMATE

df <- fread("/Users/julia.ericson/HMM/data/test_data_max_long_grid_25min.csv")  

preProcValues <- preProcess(df %>% 
                              dplyr::select(Instructions1, Instructions2, Instructions3, Instructions4, Instructions5, Instructions6,
                                            Maths1, Maths2, Maths3, Maths4, Maths5, Maths6, OddOneOut1, OddOneOut2, OddOneOut3,
                                            OddOneOut4, OddOneOut5, OddOneOut6, grid2, grid3, grid4, grid5, grid6),
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
imputed_df[, c('grid2', 'grid3', 'grid4', 'grid5', 'grid6')] <- (imputed_df[, c('grid2', 'grid3', 'grid4', 'grid5', 'grid6')] - preProcValues$mean['grid2']) / preProcValues$std['grid2']

#================ INVARIANCE TESTING ===========================================

config_model <- '
  # Define latent
    factor2 =~ NA*Instructions2 + lambda1*Instructions2 + lambda2*Maths2 + lambda3*OddOneOut2 + grid2
    factor3 =~ NA*Instructions3 + lambda1*Instructions3 + lambda2*Maths3 + lambda3*OddOneOut3 + grid3
    factor4 =~ NA*Instructions4 + lambda1*Instructions4 + lambda2*Maths4 + lambda3*OddOneOut4 + grid4
    factor5 =~ NA*Instructions5 + lambda1*Instructions5 + lambda2*Maths5 + lambda3*OddOneOut5 + grid5
    factor6 =~ NA*Instructions6 + lambda1*Instructions6 + lambda2*Maths6 + lambda3*OddOneOut6 + grid6
       
  # Intercepts
    Instructions2 ~ i1*1
    Maths2 ~ i2*1
    OddOneOut2 ~ i3*1
    grid2 ~ 1
    Instructions3 ~ i1*1
    Maths3 ~ i2*1
    OddOneOut3 ~ i3*1 
    grid3 ~ 1
    Instructions4 ~ i1*1
    Maths4 ~ i2*1
    OddOneOut4 ~ i3*1 
    grid4 ~ 1
    Instructions5 ~ i1*1
    Maths5 ~ i2*1
    OddOneOut5 ~ i3*1 
    grid5 ~ 1
    Instructions6 ~ i1*1
    Maths6 ~ i2*1
    OddOneOut6 ~ i3*1 
    grid6 ~ 1
  
  # Variances
    Instructions2 ~~ Instructions2
    Maths2 ~~ Maths2
    OddOneOut2 ~~ OddOneOut2
    grid2 ~~grid2
    Instructions3 ~~ Instructions3
    Maths3 ~~ Maths3
    OddOneOut3 ~~ OddOneOut3
    grid3 ~~grid3
    Instructions4 ~~ Instructions4
    Maths4 ~~ Maths4
    OddOneOut4 ~~ OddOneOut4
    grid4 ~~grid4
    Instructions5 ~~ Instructions5
    Maths5 ~~ Maths5
    OddOneOut5 ~~ OddOneOut5
    grid5 ~~grid5
    Instructions6 ~~ Instructions6
    Maths6 ~~ Maths6
    OddOneOut6 ~~ OddOneOut6
    grid6 ~~grid6
    
  # Correlations observations
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
    grid2 ~~ grid3 + grid4 + grid5 + grid6
    grid3 ~~ grid4 + grid5 + grid6
    grid4 ~~ grid5 + grid6
    grid5 ~~ grid6
    
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
    factor2 =~ NA*Instructions2 + lambda1*Instructions2 + lambda2*Maths2 + lambda3*OddOneOut2 + lambda4*grid2
    factor3 =~ NA*Instructions3 + lambda1*Instructions3 + lambda2*Maths3 + lambda3*OddOneOut3 + lambda4*grid3
    factor4 =~ NA*Instructions4 + lambda1*Instructions4 + lambda2*Maths4 + lambda3*OddOneOut4 + lambda4*grid4
    factor5 =~ NA*Instructions5 + lambda1*Instructions5 + lambda2*Maths5 + lambda3*OddOneOut5 + lambda4*grid5
    factor6 =~ NA*Instructions6 + lambda1*Instructions6 + lambda2*Maths6 + lambda3*OddOneOut6 + lambda4*grid6
     
  # Intercepts
    Instructions2 ~ i1*1
    Maths2 ~ i2*1
    OddOneOut2 ~ i3*1
    grid2 ~ 1
    Instructions3 ~ i1*1
    Maths3 ~ i2*1
    OddOneOut3 ~ i3*1 
    grid3 ~ 1
    Instructions4 ~ i1*1
    Maths4 ~ i2*1
    OddOneOut4 ~ i3*1 
    grid4 ~ 1
    Instructions5 ~ i1*1
    Maths5 ~ i2*1
    OddOneOut5 ~ i3*1 
    grid5 ~ 1
    Instructions6 ~ i1*1
    Maths6 ~ i2*1
    OddOneOut6 ~ i3*1 
    grid6 ~ 1
  
  # Variances
    Instructions2 ~~ Instructions2
    Maths2 ~~ Maths2
    OddOneOut2 ~~ OddOneOut2
    grid2 ~~grid2
    Instructions3 ~~ Instructions3
    Maths3 ~~ Maths3
    OddOneOut3 ~~ OddOneOut3
    grid3 ~~grid3
    Instructions4 ~~ Instructions4
    Maths4 ~~ Maths4
    OddOneOut4 ~~ OddOneOut4
    grid4 ~~grid4
    Instructions5 ~~ Instructions5
    Maths5 ~~ Maths5
    OddOneOut5 ~~ OddOneOut5
    grid5 ~~grid5
    Instructions6 ~~ Instructions6
    Maths6 ~~ Maths6
    OddOneOut6 ~~ OddOneOut6
    grid6 ~~grid6
    
  # Correlations observations
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
    grid2 ~~ grid3 + grid4 + grid5 + grid6
    grid3 ~~ grid4 + grid5 + grid6
    grid4 ~~ grid5 + grid6
    grid5 ~~ grid6
    
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

# Note: Almost strong model, intercept for grid2 is different to later ones.  
strong_model <- '
  # Define latent
    factor2 =~ NA*Instructions2 + lambda1*Instructions2 + lambda2*Maths2 + lambda3*OddOneOut2 + lambda4*grid2
    factor3 =~ NA*Instructions3 + lambda1*Instructions3 + lambda2*Maths3 + lambda3*OddOneOut3 + lambda4*grid3
    factor4 =~ NA*Instructions4 + lambda1*Instructions4 + lambda2*Maths4 + lambda3*OddOneOut4 + lambda4*grid4
    factor5 =~ NA*Instructions5 + lambda1*Instructions5 + lambda2*Maths5 + lambda3*OddOneOut5 + lambda4*grid5
    factor6 =~ NA*Instructions6 + lambda1*Instructions6 + lambda2*Maths6 + lambda3*OddOneOut6 + lambda4*grid6
     
  # Intercepts
    Instructions2 ~ i1*1
    Maths2 ~ i2*1
    OddOneOut2 ~ i3*1
    grid2 ~ 0*1
    Instructions3 ~ i1*1
    Maths3 ~ i2*1
    OddOneOut3 ~ i3*1 
    grid3 ~ i4*1
    Instructions4 ~ i1*1
    Maths4 ~ i2*1
    OddOneOut4 ~ i3*1 
    grid4 ~ i4*1
    Instructions5 ~ i1*1
    Maths5 ~ i2*1
    OddOneOut5 ~ i3*1 
    grid5 ~ i4*1
    Instructions6 ~ i1*1
    Maths6 ~ i2*1
    OddOneOut6 ~ i3*1 
    grid6 ~ i4*1
  
  # Variances
    Instructions2 ~~ Instructions2
    Maths2 ~~ Maths2
    OddOneOut2 ~~ OddOneOut2
    grid2 ~~grid2
    Instructions3 ~~ Instructions3
    Maths3 ~~ Maths3
    OddOneOut3 ~~ OddOneOut3
    grid3 ~~grid3
    Instructions4 ~~ Instructions4
    Maths4 ~~ Maths4
    OddOneOut4 ~~ OddOneOut4
    grid4 ~~grid4
    Instructions5 ~~ Instructions5
    Maths5 ~~ Maths5
    OddOneOut5 ~~ OddOneOut5
    grid5 ~~grid5
    Instructions6 ~~ Instructions6
    Maths6 ~~ Maths6
    OddOneOut6 ~~ OddOneOut6
    grid6 ~~grid6
    
  # Correlations observations
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
    grid2 ~~ grid3 + grid4 + grid5 + grid6
    grid3 ~~ grid4 + grid5 + grid6
    grid4 ~~ grid5 + grid6
    grid5 ~~ grid6
    
  # Latent Variable Means
    factor2 ~ 0*1
    factor3 ~ 1
    factor4 ~ 1
    factor5 ~ f*1
    factor6 ~ f*1
    
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
lavInspect(fit_strong, "cov.lv")

#===============================================================================
# FINAL MODEL WITH GRID INTERCEPT CHANGED TO FACTOR
# Grid intercept have to be a factor to easily extrtact it. 
#===============================================================================

strong_model_final <- '
  # Define latent
    factor2 =~ NA*Instructions2 + lambda1*Instructions2 + lambda2*Maths2 + lambda3*OddOneOut2 + lambda4*grid2
    factor3 =~ NA*Instructions3 + lambda1*Instructions3 + lambda2*Maths3 + lambda3*OddOneOut3 + lambda4*grid3
    factor4 =~ NA*Instructions4 + lambda1*Instructions4 + lambda2*Maths4 + lambda3*OddOneOut4 + lambda4*grid4
    factor5 =~ NA*Instructions5 + lambda1*Instructions5 + lambda2*Maths5 + lambda3*OddOneOut5 + lambda4*grid5
    factor6 =~ NA*Instructions6 + lambda1*Instructions6 + lambda2*Maths6 + lambda3*OddOneOut6 + lambda4*grid6
    
    factor22 =~ 1*grid2
    factor32 =~ 1*grid3
    factor42 =~ 1*grid4
    factor52 =~ 1*grid5
    factor62 =~ 1*grid6
    
  # Intercepts
    Instructions2 ~ i1*1
    Maths2 ~ i2*1
    OddOneOut2 ~ i3*1
    grid2 ~ 0*1
    Instructions3 ~ i1*1
    Maths3 ~ i2*1
    OddOneOut3 ~ i3*1 
    grid3 ~ 0*1
    Instructions4 ~ i1*1
    Maths4 ~ i2*1
    OddOneOut4 ~ i3*1 
    grid4 ~ 0*1
    Instructions5 ~ i1*1
    Maths5 ~ i2*1
    OddOneOut5 ~ i3*1
    grid5 ~ 0*1
    Instructions6 ~ i1*1
    Maths6 ~ i2*1
    OddOneOut6 ~ i3*1 
    grid6 ~ 0*1
  
  # Variances
    Instructions2 ~~ Instructions2
    Maths2 ~~ Maths2
    OddOneOut2 ~~ OddOneOut2
    grid2 ~~ 0*grid2
    Instructions3 ~~ Instructions3
    Maths3 ~~ Maths3
    OddOneOut3 ~~ OddOneOut3
    grid3 ~~ 0*grid3
    Instructions4 ~~ Instructions4
    Maths4 ~~ Maths4
    OddOneOut4 ~~ OddOneOut4
    grid4 ~~ 0*grid4
    Instructions5 ~~ Instructions5
    Maths5 ~~ Maths5
    OddOneOut5 ~~ OddOneOut5
    grid5 ~~ 0*grid5
    Instructions6 ~~ Instructions6
    Maths6 ~~ Maths6
    OddOneOut6 ~~ OddOneOut6
    grid6 ~~ 0*grid6
    
  # Correlations observations
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
    grid2 ~~ 0*grid3 + 0*grid4 + 0*grid5 + 0*grid6
    grid3 ~~ 0*grid4 + 0*grid5 + 0*grid6
    grid4 ~~ 0*grid5 + 0*grid6
    grid5 ~~ 0*grid6
    
  # Latent Variable Means
    factor2 ~ 0*1
    factor3 ~ 1
    factor4 ~ 1
    factor5 ~ 1
    factor6 ~ 1
    
    factor22 ~ 0*1
    factor32 ~ mu*1
    factor42 ~ mu*1
    factor52 ~ mu*1
    factor62 ~ mu*1
    
  # Latent variables variances and covariances
    factor2 ~~ 1*factor2
    factor3 ~~ factor3
    factor4 ~~ factor4
    factor5 ~~ factor5
    factor6 ~~ factor6
    
    factor22 ~~ factor22  # 0
    factor32 ~~ factor32
    factor42 ~~ factor42
    factor52 ~~ factor52
    factor62 ~~ factor62
    
    factor2 ~~ factor3 + factor4 + factor5 + factor6 + 0*factor22 + 0*factor32 + 0*factor42 + 0*factor52 + 0*factor62
    factor3 ~~ factor4 + factor5 + factor6 + 0*factor32 + 0*factor22 + 0*factor42 + 0*factor52 + 0*factor62
    factor4 ~~ factor5 + factor6 + 0*factor22 +  0*factor32 + 0*factor42 + 0*factor52 + 0*factor62
    factor5 ~~ factor6 + 0*factor22 +  0*factor32 + 0*factor42 + 0*factor52 + 0*factor62
    factor6 ~~ 0*factor32 + 0*factor22 + 0*factor42 + 0*factor52 + 0*factor62
    
    factor22 ~~ factor32 + factor42 + factor52 + factor62
    factor32 ~~ factor42 + factor52 + factor62
    factor42 ~~ factor52 + factor62
    factor52 ~~ factor62
'

fit_strong_final <- lavaan(strong_model_final, data = imputed_df, estimator = "MLM")
summary(fit_strong_final, fit.measures = TRUE)

fscores_strat <- lavPredict(fit_strong_final)
