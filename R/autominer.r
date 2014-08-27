#' CARET Autominer
#'
#' This process aims to streamline the process of using CARET for data mining by providing automatic
#' training test splits as well implementation of models in parallel for cross method comparisons
#' @param formula Either a plain text formula or an item of the class formula
#' @param model CARET model for use, can accept a list through the lapply or sapply wrapper
#' @param tactic A string representing the assessment criteria for model, Regression supports 'RMSE' or 'R2' and Classification features 'Kappa' or 'Accuracy'
#' @param data A dataframe featuring all the variables used in the formula
#' @param seed A random number seed
#' @keywords caret, data mining, classification, regression
#' @export
#' @examples
#' Regression    
#' automine(Sepal.Width~., 'earth', 'RMSE', iris, 11)
#' 
#' Classification   
#' automine(Species~., 'rf', 'Kappa', iris, 11)
#' 
#' Multiple Models   
#' modlist<-list('glm','rf','earth')
#' reg.grid<-sapply(modlist, function(modlist) automine(Sepal.Width~., modlist, 'RMSE', iris, 11)


automine<-function(formula,model,tactic,data,seed){
  require(doMC)
  require(pROC)
  require(caret)
  set.seed(seed)
  count<-0
  dv<-all.vars(update(formula, . ~ 1))
  registerDoMC(cores=detectCores()-2)
  splitter<-unlist(data[dv][,1])
  trainIndex <- createDataPartition(splitter, p = .7,
                                    list = FALSE,
                                    times=1)
  train <- data[ trainIndex,]
  test  <- data[-trainIndex,]
  
  if (is.factor(splitter)==TRUE){
    fitControl<- trainControl(
      method = "repeatedcv",
      number = 10,
      repeats = 2,
      classProbs = TRUE,
      allowParallel=TRUE)
    m1<-train(formula, data=train, method=model, metric=tactic, trControl = fitControl)
    m2<-predict(m1, newdata=test)
    m3<-predict(m1, newdata=test, type='prob')
    m4<-confusionMatrix(m2, test[dv][,1])
    m5<-roc(as.numeric(test[dv][,1]), as.numeric(m3[,1]))
    m6<-ggplot(m1)
    bundle<-list(m1,m2,m3,m4,m5,m6)
    return(bundle)
  }
  if (is.factor(splitter)==FALSE){
    regfit<- trainControl(
      method = "repeatedcv",
      number = 10,
      repeats = 2,
      allowParallel=TRUE)
    re1<-train(formula, data = train, method=model, trControl =regfit, metric=tactic)
    re2<-predict(re1, newdata=test)
    re3<-test[dv][,1]-re2
    re4<-postResample(re2, unlist(test[dv][,1]))
    rundle<-list(re1,re2,re3,re4)  
    return(rundle)
  }
}
