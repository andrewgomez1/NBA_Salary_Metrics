library(ggplot2)
library(data.table)
library(corrplot)
library(GGally)
library(tidyverse)
library(plotly)
library(caret)
install.packages("ggm")
library(ggm)
install.packages("glmnet")
library(glmnet)


# Import Data
players_stats <- read.csv("/Users/andygomez/Desktop/Courses/
                          Spring 2023/MAP 4102/R Project/
                          NBA Player Stats(1950 - 2022).csv")
salaries_data <- read.csv("/Users/andygomez/Desktop/Courses/
                          Spring 2023/MAP 4102/R Project/
                          NBA Salaries(1990-2023).csv")
head(salaries_data)
head(players_stats)

stats16 <- players_stats[(players_stats$Season == 2016),]
sal16 <- salaries_data[(salaries_data$seasonStartYear == 2015),]

stalaries <- merge(stats16, sal16, by.x = "Player",
                   by.y = "playerName")

head(stalaries)
tail(stalaries)
tail(salaries_data)
nrow(sal16)
nrow(stats16)
names(players_stats)

#maybe I could have done percent of salary cap,  but I decided to do inflationAdj

# Data Cleaning
stalariesNew$PPG <- stalariesNew$PTS/stalariesNew$G
stalariesNew$RPG <- (stalariesNew$ORB + stalariesNew$DRB)/stalariesNew$G
stalariesNew$APG <- stalariesNew$AST/stalariesNew$G
stalariesNew$BPG <- stalariesNew$BLK/stalariesNew$G
stalariesNew$TOPG <- stalariesNew$TOV/stalariesNew$G
stalariesNew$PFPG <- stalariesNew$PF/stalariesNew$G
stalariesNew$MPG <- stalariesNew$MP/stalariesNew$G
stalariesNew$FPM <- stalariesNew$PF/stalariesNew$MP #fouls per minute played, since fouls might show an increase in salary since theyre playing more, i made it fouls per minute 
stalariesNew$TOPM <- stalariesNew$TOV/stalariesNew$MP #turnovers per minute played, samd as above

points_by_age <- lm(PPG ~ Age, data = players_stats)
summary(points_by_age)
new_ages <- data.frame(Age=c(21,24,25))

predict(points_by_age, newdata = new_ages)

names(stalaries)
stalaries[is.na(stalaries)] <- 0 #turned NA's into zeros

sapply(stalaries, class) #checking which columns are numeric
as.numeric(stalaries$FG.) #turning these columns numeric
as.numeric(stalaries$X3P.)
as.numeric(stalaries$X2P.)
as.numeric(stalaries$FT.)
stalaries$FG. <- stalaries$FG. *100
stalaries$X3P. <- stalaries$X3P. *100 
stalaries$X2P. <- stalaries$X2P. *100 
stalaries$FT. <- stalaries$FT. *100 

stalaries$inflationAdjSalary = as.factor(gsub(",", "",
                                stalaries$inflationAdjSalary))
stalaries$inflationAdjSalary = as.numeric(gsub("\\$", "",
                                stalaries$inflationAdjSalary)) #changed the dollar amounts to numeric

stalaries$G < 20
stalaries_new <- subset(stalaries, G > 20) #removed all rows where the player did not play at least 20 games

corrplot(cor(stalaries_new %>% select(Age,G:MPG, inflationAdjSalary), use = "everything"),
         method = "square", type = "upper") #created correlation plot to visualize which factors influence salary the most

corrplot(cor(stalaries_new %>% select(inflationAdjSalary, PPG, MPG,
                                     FG, FT, TOPG, GS, RPG, PFPG, APG),
             use = "everything"), method = "number", type = "upper")      #created correlation plot to visualize which factors influence salary the most

names(stalaries)

# Split Data
install.packages("caTools")
library(caTools)

set.seed(123)
split = sample.split(stalaries_new$inflationAdjSalary, SplitRatio = 0.7)
train = subset(stalaries_new, split == TRUE)
test = subset(stalaries_new, split == FALSE)

#turned data frame into only numerics and no constant cols
numeric_cols <- unlist(lapply(stalaries_new, is.numeric))
constant_cols <- apply(stalaries_new, 2, function(x) length(unique(x)) <= 1)
stalaries3 <- stalaries_new[ , !(constant_cols) & numeric_cols]

# Multiple Regression Model
model <- lm(inflationAdjSalary ~ PPG + FG + 
              TOPG + RPG + APG, data = train_df)

summary(model) #shows that PPG, FG, TOPG, RPG, APG are the only significant variables

predictions <- predict(model, newdata = test_df)
summary(predictions)


#mean square error
MSE <- mean((test_df$inflationAdjSalary - predictions)^2)
#root mean square error
RMSE <- sqrt(MSE) #RMSE = 0.8435


# Exploratory Data Analysis
hist(stalaries$PPG, xlab = "Points per Game", 
     ylab = "Number of Players", main = "Players PPG")
hist(stalaries$inflationAdjSalary, xlab = "Salary", 
     ylab = "Number of Players", main = "Inflation Adjusted Salaries")
summary(stalaries$inflationAdjSalary) #mean greater than median (right skewed)
boxplot(stalaries$inflationAdjSalary, horizontal = TRUE)
boxplot(stalaries$PPG, horizontal = TRUE)
hist(stalaries$FT.[!stalaries$FT.==0]) #got rid of zeros since they're just the NA's
hist(stalaries$FG.[!stalaries$FG.==0])
hist(stalaries$X3P.[!stalaries$X3P.==0])

pairs(stalaries3[ , c(37,29,7,33,30,31)])
pairs(stalaries3[ , c(37,5,6,17,34)])


points_mins <- lm(inflationAdjSalary ~ PPG + MPG + GS, data = stalaries_new)
summary(points_mins)

#made df into just the variables I need
stalaries3 <- subset(stalaries3[ , c(7,29:31,33,37)])

#PCA
pca_result <- prcomp(scaled_data)
summary(pca_result)
loadings(pca_result)
scores <- as.data.frame(pca_result$x)

#test for multicollinearity
install.packages("car")
library(car)
vif(model)

scaled_data <- scale(stalaries3)
df <- data.frame(scaled_data) #had to turn scaled_data into a data frame

# RIDGE REGRESSION

# Split the data into training and testing sets (70% and 30% respectively)
set.seed(12345) # for reproducibility
train_index <- sample(nrow(df), 0.7 * nrow(df))
train_df <- df[train_index,]
test_df <- df[-train_index,]

# Fit the ridge regression model using the training data
x_train <- as.matrix(train_df[,1:5]) # extract independent variables from training data
y_train <- train_df$inflationAdjSalary # extract dependent variable from training data
ridge_model <- glmnet(x_train, y_train, alpha = 0, lambda = 0.1)
# Make predictions using the testing data
x_test <- as.matrix(test_df[,1:5]) # extract independent variables from testing data
y_test <- test_df$inflationAdjSalary # extract dependent variable from testing data
pred_y <- predict(ridge_model, newx = x_test)
# Evaluate the model using a performance metric, such as mean squared error
mse_ridge <- mean((pred_y - y_test)^2) #0.7356
rmse_ridge <- sqrt(mse_ridge) #0.8577
  #not a good rmse, so try to find better lambda

# Finding the optimal value of lambda
cv.fit <- cv.glmnet(x_train, y_train, alpha = 0,
                    lambda = seq(from = 0.01, to = 1, by = 0.01),
                    nfolds = 10)
plot(cv.fit)

lambda_min <- cv.fit$lambda.min
final_model <- glmnet(x_train, y_train, alpha = 0, lambda = lambda_min)

final_pred <- predict(final_model, newx = x_test)
final_mse <- mean((y_test - final_pred)^2) #0.7145
final_rmse <- sqrt(final_mse) #0.8453



