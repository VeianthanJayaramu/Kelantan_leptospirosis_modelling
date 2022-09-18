#=========================================== MODEL DEVELOPMENT ===============================================
# set working directory

# call libraries
library(randomForest)
library(caret)
library(doParallel)

# import and view structure of data
mydata <- read.csv("M1_finalInput.csv", header = TRUE, stringsAsFactors = TRUE)
str(mydata)

# set seed
seed <- 1

# resample for data splitting
set.seed(seed)
inTrain <- createDataPartition(y = mydata$Cases, p = 0.8, list = FALSE)

# split into training and testing sets - view the distribution
training <- mydata[inTrain, ]
testing <- mydata[-inTrain, ]

# Develop a model using bagging ensemble technique - Random Forest ---------------------------------------------

# customize random forest
#****************************************************************************************
customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree", "nodesize"),
                                  class = rep("numeric", 3),
                                  label = c("mtry", "ntree", "nodesize"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs) {
  randomForest(x, y, mtry = param$mtry, ntree = param$ntree, nodesize = param$nodesize)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]), ]
#*****************************************************************************************

# control model training
set.seed(seed)
ind <- createMultiFolds(training$Cases, k = 10, times = 3)
control <- trainControl(method = "repeatedcv",
                        returnResamp = "all",
                        savePredictions = "all",
                        classProbs = TRUE,
                        index = ind,
                        allowParallel = FALSE)

# create different combinations of tuning parameters
tunegrid <- expand.grid(.mtry = seq(from = 1, to = 10, by = 1), # number of predictors
                        .ntree = seq(from = 200, to = 700, by = 25), # number of trees
                        .nodesize = seq(from = 1, to = 10, by = 1)) # number of terminal node

# activate parallel processing
ncores <- detectCores()
cl <- makePSOCKcluster(ncores)
registerDoParallel(cl)

# record training start time
start.time <- proc.time()

# train and tune model
set.seed(seed)
model.rf <- train(Cases~.,
                  data = training,
                  method = customRF,
                  metric = "Accuracy",
                  tuneGrid = tunegrid,
                  trControl = control)

# record training stop time and calculate model run time
stop.time <- proc.time()
run.time <- stop.time - start.time; print(run.time)

# deactivate parallel processing
stopCluster(cl)

# measure training performance
pred <- model.rf$finalModel$predicted
obs <- training$Cases
confusionMatrix(pred, obs, positive = 'High')

# predict testing set output using developed 
set.seed(seed)
prediction <- predict(model.rf, testing)

# calculate testing accuracy
confusionMatrix(prediction, testing$Cases, positive = 'High')
