library(data.table); library(dplyr); library(ggplot2); library(caret) 
library(lavaan); library(semPlot)

df <- fread("/Users/julia.ericson/HMM/data/cogmed_test_and_train_25min_9to11.csv")

preProcValues <- preProcess(df %>% 
                              dplyr::select(Instructions1, Instructions2, Instructions3, Instructions4, Instructions5, Instructions6,
                                            Maths1, Maths2, Maths3, Maths4, Maths5, Maths6, OddOneOut1, OddOneOut2, OddOneOut3,
                                            OddOneOut4, OddOneOut5, OddOneOut6, grid2, grid3, grid4, grid5, grid6),
                            method = c("knnImpute"),
                            k = 35,
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

#================ 1. INVARIANCE TESTING WITHOUT GRID ===========================

config_model <- '
  # Define latent
    factor2 =~ NA*Instructions2 + lambda1*Instructions2 + lambda2*Maths2 + lambda3*OddOneOut2
    factor3 =~ NA*Instructions3 + lambda4*Instructions3 + lambda5*Maths3 + lambda6*OddOneOut3
    factor4 =~ NA*Instructions4 + lambda7*Instructions4 + lambda8*Maths4 + lambda9*OddOneOut4
    factor5 =~ NA*Instructions5 + lambda10*Instructions5 + lambda11*Maths5 + lambda12*OddOneOut5
    factor6 =~ NA*Instructions6 + lambda13*Instructions6 + lambda14*Maths6 + lambda15*OddOneOut6
    
    lambda1 + lambda2 + lambda3 == 3
    lambda4 + lambda5 + lambda6 == 3
    lambda7 + lambda8 + lambda9 == 3
    lambda10 + lambda11 + lambda12 == 3
    lambda13 + lambda14 + lambda15 == 3
    
  # Intercepts
    Instructions2 ~ i1*1
    Maths2 ~ i2*1
    OddOneOut2 ~ i3*1
    Instructions3 ~ i4*1
    Maths3 ~ i5*1
    OddOneOut3 ~ i6*1 
    Instructions4 ~ i7*1
    Maths4 ~ i8*1
    OddOneOut4 ~ i9*1 
    Instructions5 ~ i10*1
    Maths5 ~ i11*1
    OddOneOut5 ~ i12*1 
    Instructions6 ~ i13*1
    Maths6 ~ i14*1
    OddOneOut6 ~ i15*1 

    i1 + i2 + i3 == 0
    i4 + i5 + i6 == 0
    i7 + i8 + i9 == 0
    i10 + i11 + i12 == 0
    i13 + i14 + i15 == 0
  
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
    factor2 ~ 1
    factor3 ~ 1
    factor4 ~ 1
    factor5 ~ 1
    factor6 ~ 1
    
  # Latent variables variances and covariances
    factor2 ~~ factor2
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
lavInspect(fit_config, "cov.lv")
options(max.print=2000)
standardizedSolution(fit_config)

weak_model <- '
  # Define latent
    factor2 =~ NA*Instructions2 + lambda1*Instructions2 + lambda2*Maths2 + lambda3*OddOneOut2
    factor3 =~ NA*Instructions3 + lambda1*Instructions3 + lambda2*Maths3 + lambda3*OddOneOut3
    factor4 =~ NA*Instructions4 + lambda1*Instructions4 + lambda2*Maths4 + lambda3*OddOneOut4
    factor5 =~ NA*Instructions5 + lambda1*Instructions5 + lambda2*Maths5 + lambda3*OddOneOut5
    factor6 =~ NA*Instructions6 + lambda1*Instructions6 + lambda2*Maths6 + lambda3*OddOneOut6
    
    lambda1 + lambda2 + lambda3 == 3
    
  # Intercepts
    Instructions2 ~ i1*1
    Maths2 ~ i2*1
    OddOneOut2 ~ i3*1
    Instructions3 ~ i4*1
    Maths3 ~ i5*1
    OddOneOut3 ~ i6*1 
    Instructions4 ~ i7*1
    Maths4 ~ i8*1
    OddOneOut4 ~ i9*1 
    Instructions5 ~ i10*1
    Maths5 ~ i11*1
    OddOneOut5 ~ i12*1 
    Instructions6 ~ i13*1
    Maths6 ~ i14*1
    OddOneOut6 ~ i15*1 
    
    i1 + i2 + i3 == 0
    i4 + i5 + i6 == 0
    i7 + i8 + i9 == 0
    i10 + i11 + i12 == 0
    i13 + i14 + i15 == 0
  
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
    factor2 ~ 1
    factor3 ~ 1
    factor4 ~ 1
    factor5 ~ 1
    factor6 ~ 1
    
  # Latent variables variances and covariances
    factor2 ~~ factor2
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
anova(fit_config,fit_weak)

strong_model <- '
  # Define latent
    factor2 =~ NA*Instructions2 + lambda1*Instructions2 + lambda2*Maths2 + lambda3*OddOneOut2
    factor3 =~ NA*Instructions3 + lambda1*Instructions3 + lambda2*Maths3 + lambda3*OddOneOut3
    factor4 =~ NA*Instructions4 + lambda1*Instructions4 + lambda2*Maths4 + lambda3*OddOneOut4
    factor5 =~ NA*Instructions5 + lambda1*Instructions5 + lambda2*Maths5 + lambda3*OddOneOut5
    factor6 =~ NA*Instructions6 + lambda1*Instructions6 + lambda2*Maths6 + lambda3*OddOneOut6
    
    lambda1 + lambda2 + lambda3 == 3
    
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
    
    i1 + i2 + i3 == 0
  
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
    factor2 ~ 1
    factor3 ~ 1
    factor4 ~ 1
    factor5 ~ 1
    factor6 ~ 1
    
  # Latent variables variances and covariances
    factor2 ~~ factor2
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
standardizedSolution(fit_strong)
anova(fit_config,fit_weak,fit_strong)
RDR_config_weak<-sqrt((17.063-8)/(8*1166))
RDR_config_weak  # 0.03117035
RDR_weak_strong<-sqrt((24.252 -8)/(8*1166))
RDR_weak_strong  # 0.04174064

strict_model <- '
  # Define latent
    factor2 =~ NA*Instructions2 + lambda1*Instructions2 + lambda2*Maths2 + lambda3*OddOneOut2
    factor3 =~ NA*Instructions3 + lambda1*Instructions3 + lambda2*Maths3 + lambda3*OddOneOut3
    factor4 =~ NA*Instructions4 + lambda1*Instructions4 + lambda2*Maths4 + lambda3*OddOneOut4
    factor5 =~ NA*Instructions5 + lambda1*Instructions5 + lambda2*Maths5 + lambda3*OddOneOut5
    factor6 =~ NA*Instructions6 + lambda1*Instructions6 + lambda2*Maths6 + lambda3*OddOneOut6
    
    lambda1 + lambda2 + lambda3 == 3
    
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
    
    i1 + i2 + i3 == 0
  
  # Variances
    Instructions2 ~~ res1*Instructions2
    Maths2 ~~ res2*Maths2
    OddOneOut2 ~~ res3*OddOneOut2
    Instructions3 ~~ res1*Instructions3
    Maths3 ~~ res2*Maths3
    OddOneOut3 ~~ res3*OddOneOut3
    Instructions4 ~~ res1*Instructions4
    Maths4 ~~ res2*Maths4
    OddOneOut4 ~~ res3*OddOneOut4
    Instructions5 ~~ res1*Instructions5
    Maths5 ~~ res2*Maths5
    OddOneOut5 ~~ res3*OddOneOut5
    Instructions6 ~~ res1*Instructions6
    Maths6 ~~ res2*Maths6
    OddOneOut6 ~~ res3*OddOneOut6
    
  # Correlations observations40
    Instructions2 ~~ Instructions3 + Instructions4 + Instructions5 + Instructions6
    Instructions3 ~~ Instructions4 + Instructions5 + Instructions6
    Instructions4 ~~ Instructions5 + Instructions6
    Instructions5 ~~ Instructions6
    Maths2 ~~ corr1*Maths3 + corr2*Maths4 + corr3*Maths5 + Maths6
    Maths3 ~~ corr1*Maths4 + corr2*Maths5 + corr3*Maths6
    Maths4 ~~ corr1*Maths5 + corr2*Maths6
    Maths5 ~~ corr1*Maths6
    OddOneOut2 ~~ OddOneOut3 + OddOneOut4 + OddOneOut5 + OddOneOut6
    OddOneOut3 ~~ OddOneOut4 + OddOneOut5 + OddOneOut6
    OddOneOut4 ~~ OddOneOut5 + OddOneOut6
    OddOneOut5 ~~ OddOneOut6
    
  # Latent Variable Means
    factor2 ~ 1
    factor3 ~ 1
    factor4 ~ 1
    factor5 ~ 1
    factor6 ~ 1
    
  # Latent variables variances and covariances
    factor2 ~~ factor2
    factor3 ~~ factor3
    factor4 ~~ factor4
    factor5 ~~ factor5
    factor6 ~~ factor6
    factor2 ~~ factor3 + factor4 + factor5 + factor6
    factor3 ~~ factor4 + factor5 + factor6
    factor4 ~~ factor5 + factor6
    factor5 ~~ factor6
'

fit_strict <- cfa(strict_model, data = imputed_df, estimator = "MLM")
summary(fit_strict, fit.measures = TRUE)
standardizedSolution(fit_strong)
anova(fit_config,fit_weak,fit_strong)
RDR_config_weak<-sqrt((17.063-8)/(8*1166))
RDR_config_weak  # 0.03117035
RDR_weak_strong<-sqrt((24.252 -8)/(8*1166))
RDR_weak_strong  # 0.04174064

#================ 2. INVARIANCE TESTING WITH GRID ==============================

config_model <- '
  # Define latent
    factor2 =~ NA*Instructions2 + lambda1*Instructions2 + lambda2*Maths2 + lambda3*OddOneOut2 + lambda4*grid2
    factor3 =~ NA*Instructions3 + lambda5*Instructions3 + lambda6*Maths3 + lambda7*OddOneOut3 + lambda8*grid3
    factor4 =~ NA*Instructions4 + lambda9*Instructions4 + lambda10*Maths4 + lambda11*OddOneOut4 + lambda12*grid4
    factor5 =~ NA*Instructions5 + lambda13*Instructions5 + lambda14*Maths5 + lambda15*OddOneOut5 + lambda16*grid5
    factor6 =~ NA*Instructions6 + lambda17*Instructions6 + lambda18*Maths6 + lambda19*OddOneOut6 + lambda20*grid6
    
    lambda1 + lambda2 + lambda3 +  lambda4 == 4
    lambda1 + lambda6 + lambda7 +  lambda8 == 4
    lambda9 + lambda10 + lambda11 +  lambda12 == 4
    lambda13 + lambda14 + lambda15 +  lambda16 == 4
    lambda17 + lambda18 + lambda19 +  lambda20 == 4
    
  # Intercepts
    Instructions2 ~ i1*1
    Maths2 ~ i2*1
    OddOneOut2 ~ i3*1
    grid2 ~ i4*1
    Instructions3 ~ i5*1
    Maths3 ~ i6*1
    OddOneOut3 ~ i7*1 
    grid3 ~ i8*1
    Instructions4 ~ i9*1
    Maths4 ~ i10*1
    OddOneOut4 ~ i11*1 
    grid4 ~ i12*1
    Instructions5 ~ i13*1
    Maths5 ~ i14*1
    OddOneOut5 ~ i15*1 
    grid5 ~ i16*1
    Instructions6 ~ i17*1
    Maths6 ~ i18*1
    OddOneOut6 ~ i19*1 
    grid6 ~ i20*1

    i1 + i2 + i3 == 0
    i5 + i6 + i7 == 0
    i9 + i10 + i11 == 0
    i13 + i14 + i15 == 0
    i17 + i18 + i19 == 0
  
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
    grid2 ~~ grid3 + grid4 + grid5 + grid6
    grid3 ~~ grid4 + grid5 + grid6
    grid4 ~~ grid5 + grid6
    grid5 ~~ grid6
    
  # Latent Variable Means
    factor2 ~ 1
    factor3 ~ 1
    factor4 ~ 1
    factor5 ~ 1
    factor6 ~ 1
    
  # Latent variables variances and covariances
    factor2 ~~ factor2
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
lavInspect(fit_config, "cov.lv")
options(max.print=2000)
standardizedSolution(fit_config)

weak_model <- '
  # Define latent
    factor2 =~ NA*Instructions2 + lambda1*Instructions2 + lambda2*Maths2 + lambda3*OddOneOut2 + lambda4*grid2
    factor3 =~ NA*Instructions3 + lambda1*Instructions3 + lambda2*Maths3 + lambda3*OddOneOut3 + lambda4*grid3
    factor4 =~ NA*Instructions4 + lambda1*Instructions4 + lambda2*Maths4 + lambda3*OddOneOut4 + lambda4*grid4
    factor5 =~ NA*Instructions5 + lambda1*Instructions5 + lambda2*Maths5 + lambda3*OddOneOut5 + lambda4*grid5
    factor6 =~ NA*Instructions6 + lambda1*Instructions6 + lambda2*Maths6 + lambda3*OddOneOut6 + lambda4*grid6
    
    lambda1 + lambda2 + lambda3 + lambda4 == 4
     
  # Intercepts
    Instructions2 ~ i1*1
    Maths2 ~ i2*1
    OddOneOut2 ~ i3*1
    grid2 ~ i4*1
    Instructions3 ~ i5*1
    Maths3 ~ i6*1
    OddOneOut3 ~ i7*1 
    grid3 ~ i8*1
    Instructions4 ~ i9*1
    Maths4 ~ i10*1
    OddOneOut4 ~ i11*1 
    grid4 ~ i12*1
    Instructions5 ~ i13*1
    Maths5 ~ i14*1
    OddOneOut5 ~ i15*1 
    grid5 ~ i16*1
    Instructions6 ~ i17*1
    Maths6 ~ i18*1
    OddOneOut6 ~ i19*1 
    grid6 ~ i20*1

    i1 + i2 + i3 + i4 == 0
    i5 + i6 + i7 + i8 == 0
    i9 + i10 + i11 + i12 == 0
    i13 + i14 + i15 + i16 == 0
    i17 + i18 + i19 + i20 == 0
  
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
    factor2 ~ 1
    factor3 ~ 1
    factor4 ~ 1
    factor5 ~ 1
    factor6 ~ 1
    
  # Latent variables variances and covariances
    factor2 ~~ factor2
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
anova(fit_config,fit_weak)

strong_model <- '
  # Define latent
    factor2 =~ NA*Instructions2 + lambda1*Instructions2 + lambda2*Maths2 + lambda3*OddOneOut2 + lambda4*grid2
    factor3 =~ NA*Instructions3 + lambda1*Instructions3 + lambda2*Maths3 + lambda3*OddOneOut3 + lambda4*grid3
    factor4 =~ NA*Instructions4 + lambda1*Instructions4 + lambda2*Maths4 + lambda3*OddOneOut4 + lambda4*grid4
    factor5 =~ NA*Instructions5 + lambda1*Instructions5 + lambda2*Maths5 + lambda3*OddOneOut5 + lambda4*grid5
    factor6 =~ NA*Instructions6 + lambda1*Instructions6 + lambda2*Maths6 + lambda3*OddOneOut6 + lambda4*grid6
    
    lambda1 + lambda2 + lambda3 + lambda4 == 4
     
  # Intercepts
    Instructions2 ~ i1*1
    Maths2 ~ i2*1
    OddOneOut2 ~ i3*1
    grid2 ~ i4*1
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
  
    i1 + i2 + i3 + i4 == 0
  
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
    factor2 ~ 1
    factor3 ~ 1
    factor4 ~ 1
    factor5 ~ 1
    factor6 ~ 1
    
  # Latent variables variances and covariances
    factor2 ~~ factor2
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

anova(fit_config,fit_weak,fit_strong)
RDR_config_weak<-sqrt((52.03 - 12)/(12*1166))
RDR_config_weak  # 0.05348757
RDR_weak_strong<-sqrt((428.73 - 12)/(12*1166))
RDR_weak_strong  # 0.1725788

#================ 3. FREE INTERCEPTS GRID ======================================

strong_model_very_relaxed <- '
  # Define latent
    factor2 =~ NA*Instructions2 + lambda1*Instructions2 + lambda2*Maths2 + lambda3*OddOneOut2 + lambda4*grid2
    factor3 =~ NA*Instructions3 + lambda1*Instructions3 + lambda2*Maths3 + lambda3*OddOneOut3 + lambda4*grid3
    factor4 =~ NA*Instructions4 + lambda1*Instructions4 + lambda2*Maths4 + lambda3*OddOneOut4 + lambda4*grid4
    factor5 =~ NA*Instructions5 + lambda1*Instructions5 + lambda2*Maths5 + lambda3*OddOneOut5 + lambda4*grid5
    factor6 =~ NA*Instructions6 + lambda1*Instructions6 + lambda2*Maths6 + lambda3*OddOneOut6 + lambda4*grid6
    
    lambda1 + lambda2 + lambda3 + lambda4 == 4
     
 # Intercepts
    Instructions2 ~ i1*1
    Maths2 ~ i2*1
    OddOneOut2 ~ i3*1
    grid2 ~ i4*1
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
    
    i1 + i2 + i3 == 0
  
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
    OddOneOut2 ~~ OddOneOut3 + OddOneOut4 + OddOneOut5 + OddOneOut6
    OddOneOut3 ~~ OddOneOut4 + OddOneOut5 + OddOneOut6
    OddOneOut4 ~~ OddOneOut5 + OddOneOut6
    OddOneOut5 ~~ OddOneOut6
    Maths2 ~~ Maths3 + Maths4 + Maths5 + Maths6
    Maths3 ~~ Maths4 + Maths5 + Maths6
    Maths4 ~~ Maths5 + Maths6
    Maths5 ~~ Maths6
    grid2 ~~ grid3 + grid4 + grid5 + grid6
    grid3 ~~ grid4 + grid5 + grid6
    grid4 ~~ grid5 + grid6
    grid5 ~~ grid6
    
  # Latent Variable Means
    factor2 ~ 1
    factor3 ~ 1
    factor4 ~ 1
    factor5 ~ 1
    factor6 ~ 1
    
  # Latent variables variances and covariances
    factor2 ~~ factor2
    factor3 ~~ factor3
    factor4 ~~ factor4
    factor5 ~~ factor5
    factor6 ~~ factor6
    factor2 ~~ factor3 + factor4 + factor5 + factor6
    factor3 ~~ factor4 + factor5 + factor6
    factor4 ~~ factor5 + factor6
    factor5 ~~ factor6
'

fit_strong_very_relaxed <- cfa(strong_model_very_relaxed, data = imputed_df, estimator = "MLM")
summary(fit_strong_very_relaxed, fit.measures = TRUE) 

anova(fit_weak,fit_strong_very_relaxed)
RDR_config_weak<-sqrt((26.099-8)/(8*1166))
RDR_config_weak  # 0.04404869

strong_model_relaxed <- '
  # Define latent
    factor2 =~ NA*Instructions2 + lambda1*Instructions2 + lambda2*Maths2 + lambda3*OddOneOut2 + lambda4*grid2
    factor3 =~ NA*Instructions3 + lambda1*Instructions3 + lambda2*Maths3 + lambda3*OddOneOut3 + lambda4*grid3
    factor4 =~ NA*Instructions4 + lambda1*Instructions4 + lambda2*Maths4 + lambda3*OddOneOut4 + lambda4*grid4
    factor5 =~ NA*Instructions5 + lambda1*Instructions5 + lambda2*Maths5 + lambda3*OddOneOut5 + lambda4*grid5
    factor6 =~ NA*Instructions6 + lambda1*Instructions6 + lambda2*Maths6 + lambda3*OddOneOut6 + lambda4*grid6
    
    lambda1 + lambda2 + lambda3 + lambda4 == 4
     
  # Intercepts
    Instructions2 ~ i1*1
    Maths2 ~ i2*1
    OddOneOut2 ~ i3*1
    grid2 ~ i4*1
    Instructions3 ~ i1*1
    Maths3 ~ i2*1
    OddOneOut3 ~ i3*1 
    grid3 ~ i5*1
    Instructions4 ~ i1*1
    Maths4 ~ i2*1
    OddOneOut4 ~ i3*1 
    grid4 ~ i5*1
    Instructions5 ~ i1*1
    Maths5 ~ i2*1
    OddOneOut5 ~ i3*1 
    grid5 ~ i5*1
    Instructions6 ~ i1*1
    Maths6 ~ i2*1
    OddOneOut6 ~ i3*1 
    grid6 ~ i5*1
    
    i1 + i2 + i3 == 0
  
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
    OddOneOut2 ~~ OddOneOut3 + OddOneOut4 + OddOneOut5 + OddOneOut6
    OddOneOut3 ~~ OddOneOut4 + OddOneOut5 + OddOneOut6
    OddOneOut4 ~~ OddOneOut5 + OddOneOut6
    OddOneOut5 ~~ OddOneOut6
    Maths2 ~~ Maths3 + Maths4 + Maths5 + Maths6
    Maths3 ~~ Maths4 + Maths5 + Maths6
    Maths4 ~~ Maths5 + Maths6
    Maths5 ~~ Maths6
    grid2 ~~ grid3 + grid4 + grid5 + grid6
    grid3 ~~ grid4 + grid5 + grid6
    grid4 ~~ grid5 + grid6
    grid5 ~~ grid6
    
  # Latent Variable Means
    factor2 ~ 1
    factor3 ~ 1
    factor4 ~ 1
    factor5 ~ 1
    factor6 ~ 1
    
  # Latent variables variances and covariances
    factor2 ~~ factor2
    factor3 ~~ factor3
    factor4 ~~ factor4
    factor5 ~~ factor5
    factor6 ~~ factor6
    factor2 ~~ factor3 + factor4 + factor5 + factor6
    factor3 ~~ factor4 + factor5 + factor6
    factor4 ~~ factor5 + factor6
    factor5 ~~ factor6
'

fit_strong_relaxed <- cfa(strong_model_relaxed, data = imputed_df, estimator = "MLM")
summary(fit_strong_relaxed, fit.measures = TRUE) 

anova(fit_strong_very_relaxed,fit_strong_relaxed)
RDR_weak_strong<-sqrt((49.799 -3)/(3*1300))
RDR_weak_strong  # 0.1238775

#================ 4. MODEL TO EXTRACT TASK INTERCEPT ===========================

strong_model_final <- '
  # Define latent
    factor2 =~ NA*Instructions2 + lambda1*Instructions2 + lambda2*Maths2 + lambda3*OddOneOut2 + lambda4*grid2
    factor3 =~ NA*Instructions3 + lambda1*Instructions3 + lambda2*Maths3 + lambda3*OddOneOut3 + lambda4*grid3
    factor4 =~ NA*Instructions4 + lambda1*Instructions4 + lambda2*Maths4 + lambda3*OddOneOut4 + lambda4*grid4
    factor5 =~ NA*Instructions5 + lambda1*Instructions5 + lambda2*Maths5 + lambda3*OddOneOut5 + lambda4*grid5
    factor6 =~ NA*Instructions6 + lambda1*Instructions6 + lambda2*Maths6 + lambda3*OddOneOut6 + lambda4*grid6
    
    lambda1 + lambda2 + lambda3 + lambda4 == 4
    
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
    factor2 ~ 1
    factor3 ~ 1
    factor4 ~ 1
    factor5 ~ 1
    factor6 ~ 1
    
    factor22 ~ i0*1
    factor32 ~ step*1
    factor42 ~ step*1
    factor52 ~ step*1
    factor62 ~ step*1
    
    i0 + i1 + i2 + i3 == 0
    
  # Latent variables variances and covariances
    factor2 ~~ factor2
    factor3 ~~ factor3
    factor4 ~~ factor4
    factor5 ~~ factor5
    factor6 ~~ factor6
    
    factor22 ~~ factor22  
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

fscores <- lavPredict(fit_strong_final)
write.table(fscores, file = "HMM/scores_9to11.csv", sep = ",", row.names = T)

#================ 5. LATENT GROWTH MODEL =======================================

separate_LGMs <- '

    i_inst =~ 1*Instructions2 + 1*Instructions3 + 1*Instructions4 + 1*Instructions5 + 1*Instructions6
    s_inst =~ 0*Instructions2 + 1*Instructions3 + 2*Instructions4 + 3*Instructions5 + 4*Instructions6

    i_grid =~ 1*grid2 + 1*grid3 + 1*grid4 + 1*grid5 + 1*grid6
    s_grid =~ 0*grid2 + 1*grid3 + 2*grid4 + 3*grid5 + 4*grid6
    step_grid =~ 0*grid2 + 1*grid3 + 1*grid4 + 1*grid5 + 1*grid6
    
    i_math =~ 1*Maths2 + 1*Maths3 + 1*Maths4 + 1*Maths5 + 1*Maths6
    s_math =~ 0*Maths2 + 1*Maths3 + 2*Maths4 + 3*Maths5 + 4*Maths6
    
    i_odd =~ 1*OddOneOut2 + 1*OddOneOut3 + 1*OddOneOut4 + 1*OddOneOut5 + 1*OddOneOut6
    s_odd =~ 0*OddOneOut2 + 1*OddOneOut3 + 2*OddOneOut4 + 3*OddOneOut5 + 4*OddOneOut6

  # Intercepts
    Instructions2 ~ 0
    grid2 ~ 0
    Maths2 ~ 0 
    OddOneOut2 ~ 0
    Instructions3 ~ 0
    grid3 ~ 0
    Maths3 ~ 0 
    OddOneOut3 ~ 0
    Instructions4 ~ 0
    grid4 ~ 0
    Maths4 ~ 0 
    OddOneOut4 ~ 0
    Instructions5 ~ 0
    grid5 ~ 0
    Maths5 ~ 0 
    OddOneOut5 ~ 0 
    Instructions6 ~ 0
    grid6 ~ 0
    Maths6 ~ 0 
    OddOneOut6 ~ 0 
  
  # Variances
    Instructions2 ~~ Instructions2
    grid2 ~~ res*grid2
    Maths2  ~~ Maths2
    OddOneOut2 ~~ OddOneOut2
    Instructions3 ~~ Instructions3
    grid3 ~~ res*grid3
    Maths3  ~~ Maths3
    OddOneOut3 ~~ OddOneOut3
    Instructions4 ~~ Instructions4
    grid4 ~~ res*grid4
    Maths4  ~~ Maths4
    OddOneOut4 ~~ OddOneOut4
    Instructions5 ~~ Instructions5
    grid5 ~~ res*grid5
    Maths5  ~~ Maths5
    OddOneOut5 ~~ OddOneOut5
    Instructions6 ~~ Instructions6
    grid6 ~~ res*grid6
    Maths6  ~~ Maths6
    OddOneOut6 ~~ OddOneOut6
    
  # Latent Variable Means
    i_inst ~ 1
    i_grid ~ 1
    i_math ~ 1
    i_odd  ~ 1
    s_inst ~ 1
    s_grid ~ 1
    s_math ~ 1
    s_odd  ~ 1
    step_grid ~ 1
    
  # Latent variables variances and covariances
    i_inst ~~ i_inst
    i_grid ~~ i_grid
    i_math ~~ i_math
    i_odd ~~ i_odd
    s_inst ~~ s_inst
    s_grid ~~ s_grid
    s_math ~~ s_math
    s_odd ~~ s_odd
    step_grid ~~ step_grid
    
    i_inst ~~ i_grid + i_math + i_odd + s_inst + s_grid + s_math + s_odd + step_grid
    i_grid ~~ i_odd + i_math + s_inst + s_grid + s_math + s_odd + step_grid
    i_odd ~~ i_math + s_inst + s_grid+ s_math + s_odd + step_grid
    i_math ~~ s_inst + s_grid + s_math + s_odd + step_grid
    s_inst ~~ s_grid+ s_math + s_odd + step_grid
    s_grid ~~ s_math + s_odd + step_grid
    s_math ~~ s_odd + step_grid
    s_odd ~~ step_grid
'

fit_separate_LGMs <- cfa(separate_LGMs, data = imputed_df, estimator = "MLM")
summary(fit_separate_LGMs, fit.measures = TRUE)
standardizedSolution(fit_separate_LGMs)
