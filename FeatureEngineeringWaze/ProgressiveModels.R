########################################################################################
########### Data infromation
########################################################################################
# datetime - hourly date + timestamp  
# season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
# holiday - whether the day is considered a holiday
# workingday - whether the day is neither a weekend nor holiday
# weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy 
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
# temp - temperature in Celsius
# atemp - "feels like" temperature in Celsius
# humidity - relative humidity
# windspeed - wind speed
# casual - number of non-registered user rentals initiated
# registered - number of registered user rentals initiated
# count - number of total rentals

########################################################################################
########### Evaluation
########################################################################################
#Submissions are evaluated one the Root Mean Squared Logarithmic Error (RMSLE). The RMSLE is calculated as

#Where:  
#n is the number of hours in the test set
#pi is your predicted count
#ai is the actual count
#log(x) is the natural logarithm

########################################################################################
########### reading the data
########################################################################################

train.file <- gzfile("./CountPrediction.train.csv.gz")
train.df <- read.csv(train.file, stringsAsFactors = FALSE)
test.file <- gzfile("./CountPrediction.test.csv.gz")
test.df <- read.csv(test.file, stringsAsFactors = FALSE)
both.df <- rbind(train.df, test.df) 
both.df$y <- log10(1+both.df$cnt)
both.df$dteday.date <- as.Date(both.df$dteday)
both.df$dteday.int <- as.integer(both.df$dteday.date)
both.df$time <- 24 * both.df$dteday.int + both.df$hr

both.df$weather.1.hour.before <- both.df$weathersit[match(both.df$time-1, both.df$time)]
both.df$weather.2.hour.before <- both.df$weathersit[match(both.df$time-2, both.df$time)]
both.df$weather.3.hour.before <- both.df$weathersit[match(both.df$time-3, both.df$time)]
both.df$weather.max.3.hours <- apply(both.df[,c('weathersit', 'weather.1.hour.before', 'weather.2.hour.before', 'weather.3.hour.before')],1,max, na.rm=TRUE)

set.seed(123)
train.indexes <- sample(1:nrow(both.df))[1:(round(0.5*nrow(both.df)))]
train.df <- both.df[train.indexes,]
test.df <- both.df[-train.indexes,]

########################################################################################
########### helper functions
########################################################################################
library(gbm)

r2.vec <- function(y.hat, y) {
  return (mean((y.hat - y)^2)/mean((y-mean(y))^2))
}

r2 <- function(y.hat, y) {
  return (colMeans((y.hat - y)^2)/mean((y-mean(y))^2))
}

########################################################################################
########### weather baseline
########################################################################################
set.seed(1234)
n.trees = round(seq(50, 200, length.out = 20))
library(gbm)
gbm.model <- gbm(formula = y~temp+atemp+hum+windspeed,
    distribution = "gaussian",                       
    data=train.df,
    n.trees = max(n.trees),
    interaction.depth = 2,
    n.minobsinnode = 50,
    shrinkage = 0.1,
    bag.fraction = 0.5,
    train.fraction = 1.0,
    cv.folds=0)
#summary(gbm.model)

train.r2 <- r2(predict(gbm.model, train.df, n.trees), train.df$y)
test.r2 <- r2(predict(gbm.model, test.df, n.trees), test.df$y)
plot(n.trees, train.r2, col = 'black', ylim = c(min(c(train.r2,test.r2)), max(c(train.r2,test.r2))))
points(n.trees, test.r2, col = 'red')
min(test.r2)#0.7281588

########################################################################################
########### day + hr as "is"
########################################################################################
set.seed(1234)
n.trees = round(seq(50, 800, length.out = 20))
gbm.model <- gbm(formula = y~temp+atemp+hum+windspeed+weekday+hr,
                 distribution = "gaussian",                       
                 data=train.df,
                 n.trees = max(n.trees),
                 interaction.depth = 3,
                 n.minobsinnode = 50,
                 shrinkage = 0.1,
                 bag.fraction = 0.5,
                 train.fraction = 1.0,
                 cv.folds=0)
#summary(gbm.model)
train.r2 <- r2(predict(gbm.model, train.df, n.trees), train.df$y)
test.r2 <- r2(predict(gbm.model, test.df, n.trees), test.df$y)
plot(n.trees, train.r2, col = 'black', ylim = c(min(c(train.r2,test.r2)), max(c(train.r2,test.r2))))
points(n.trees, test.r2, col = 'red')
min(test.r2)#0.1016788
which.min(test.r2)

########################################################################################
########### model per day
########################################################################################

set.seed(1234)
max.n.trees <- 600
train.df$y.hat <- NA
day.values <- sort(unique(train.df$weekday))
for (i in day.values) { 
  hold.in.indexes <- which(i==train.df$weekday)
  hold.in <- train.df[hold.in.indexes,]
  gbm.model.i <- gbm(formula = y~temp+atemp+hum+windspeed+hr,
                 distribution = "gaussian",                       
                 data=hold.in,
                 n.trees = max.n.trees,
                 interaction.depth = 3,
                 n.minobsinnode = 50,
                 shrinkage = 0.1,
                 bag.fraction = 0.5,
                 train.fraction = 1.0,
                 cv.folds=0)
  hold.in$y.hat <- predict(gbm.model.i, hold.in, max.n.trees)
  train.df$y.hat[hold.in.indexes] <- hold.in$y.hat
  test.df$y.hat[i==test.df$weekday] <- predict(gbm.model.i, test.df[i==test.df$weekday,], max.n.trees)
}
#summary(gbm.model)
train.r2 <- r2.vec(train.df$y.hat, train.df$y)
test.r2 <- r2.vec(test.df$y.hat, test.df$y)
#0.1063321

########################################################################################
########### Last hour
########################################################################################
train.df$y.1.hour.before <- train.df$y[match(train.df$time-1, train.df$time)]
train.df$y.2.hour.before <- train.df$y[match(train.df$time-2, train.df$time)]
train.df$y.3.hour.before <- train.df$y[match(train.df$time-3, train.df$time)]
test.df$y.1.hour.before  <- train.df$y[match(test.df$time-1, train.df$time)]
test.df$y.2.hour.before  <- train.df$y[match(test.df$time-2, train.df$time)]
test.df$y.3.hour.before  <- train.df$y[match(test.df$time-3, train.df$time)]

set.seed(1234)
n.trees = round(seq(50, 900, length.out = 20))
gbm.model <- gbm(formula = y~temp+atemp+hum+windspeed+weekday+hr+y.1.hour.before+y.2.hour.before+y.3.hour.before,
                 distribution = "gaussian",                       
                 data=train.df,
                 n.trees = max(n.trees),
                 interaction.depth = 3,
                 n.minobsinnode = 50,
                 shrinkage = 0.1,
                 bag.fraction = 0.5,
                 train.fraction = 1.0,
                 cv.folds=0)
train.r2 <- r2(predict(gbm.model, train.df, n.trees), train.df$y)
test.r2 <- r2(predict(gbm.model, test.df, n.trees), test.df$y)
plot(n.trees, train.r2, col = 'black', ylim = c(min(c(train.r2,test.r2)), max(c(train.r2,test.r2))))
points(n.trees, test.r2, col = 'red')
min(test.r2)#0.08008987
which.min(test.r2)

########################################################################################
########### weather history
########################################################################################
set.seed(1234)
n.trees = round(seq(50, 900, length.out = 20))
gbm.model <- gbm(formula = y~temp+atemp+hum+windspeed+weekday+hr+y.1.hour.before+
                   y.2.hour.before+y.3.hour.before+weather.max.3.hours,
                 distribution = "gaussian",                       
                 data=train.df,
                 n.trees = max(n.trees),
                 interaction.depth = 3,
                 n.minobsinnode = 50,
                 shrinkage = 0.1,
                 bag.fraction = 0.5,
                 train.fraction = 1.0,
                 cv.folds=0)
train.r2 <- r2(predict(gbm.model, train.df, n.trees), train.df$y)
test.r2 <- r2(predict(gbm.model, test.df, n.trees), test.df$y)
plot(n.trees, train.r2, col = 'black', ylim = c(min(c(train.r2,test.r2)), max(c(train.r2,test.r2))))
points(n.trees, test.r2, col = 'red')
min(test.r2)#0.07798527
which.min(test.r2)

########################################################################################
########### adding the date as integer
########################################################################################

n.trees = round(seq(50, 2000, length.out = 20))
gbm.model <- gbm(formula = y~temp+atemp+hum+windspeed+weekday+hr+y.1.hour.before+
                   y.2.hour.before+y.3.hour.before+weather.max.3.hours+dteday.int,
                 distribution = "gaussian",                       
                 data=train.df,
                 n.trees = max(n.trees),
                 interaction.depth = 3,
                 n.minobsinnode = 50,
                 shrinkage = 0.1,
                 bag.fraction = 0.5,
                 train.fraction = 1.0,
                 cv.folds=0)
#summary(gbm.model)
train.r2 <- r2(predict(gbm.model, train.df, n.trees), train.df$y)
test.r2 <- r2(predict(gbm.model, test.df, n.trees), test.df$y)
plot(n.trees, train.r2, col = 'black', ylim = c(min(c(train.r2,test.r2)), max(c(train.r2,test.r2))))
points(n.trees, test.r2, col = 'red')
min(test.r2)#0.0558564
which.min(test.r2)

