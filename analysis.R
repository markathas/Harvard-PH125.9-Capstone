# Mark Athas
# Harvardx PH125.9x
# October, 2022
# Copywrite: Mark Athas, 2022
# Licensed under GPLv3.0

# NOTE: RUN TIME is 9 - 12 Minutes on MacBook Pro 2017

########################################################################
## NOTE    NOTE    NOTE
## Minor changes may be required to run this analysis on your computer.
## 1. Change setwd() to your local directory where the project files were
##    downloaded.
## 2. Change loans <- read.csv() to your local directory where the
##    lar_chunk-balanced.csv file was downloaded
########################################################################
setwd("/Users/MyDocuments/HarvardX/DataScienceWithR/09 Capstone/loans")

## lar_chunk-balanced.csv.zip can be found at: 
#  https://drive.google.com/file/d/1oc1d69Ke35iBGgzdLUbS5Gt4J6N3brpo/view?usp=sharing

loans <- read.csv("/Users/MyDocuments/data/lar_chunk-balanced.csv")

if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret")
if (!require(scales)) install.packages("scales")
if (!require(randomForest)) install.packages("randomForest")
if (!require(corrplot)) install.packages("corrplot")
if (!require(knitr)) install.packages("knitr")
if (!require(DataExplorer)) install.packages("DataExplorer")
if (!require(class)) install.packages("class")
if (!require(ROCR)) install.packages("ROCR")
if (!require(GGally)) install.packages("GGally")
if (!require(party)) install.packages("party")
if (!require(e1071)) install.packages("e1071")
if (!require(xgboost)) install.packages("xgboost")

####################
# DATA LOAD / ASSESS
####################
start <- Sys.time()
options(digits = 6)

# Set count of original input data
loans_original_total <- nrow(loans)

# Create function for confusion matrix F1 value
cmf1 <- function(m) {
  f1 <- 2 * ((m$byClass[2] * m$byClass[1]) / (m$byClass[2] + m$byClass[1]))
  return(f1)
}

# Create a standard 2-by for the confusion matrix
ggplotConfusionMatrix <- function(m){
  f1 <- cmf1(m)
  mytitle <- paste("Accuracy", percent(m$overall[1], accuracy = 0.1),
                   "  F1", percent(f1, accuracy = 0.1))
  p <-
    ggplot(data = as.data.frame(m$table) ,
           aes(x = Reference, y = Prediction)) +
    geom_tile(aes(fill = log(Freq)), colour = "white") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(x = Reference, y = Prediction, label = Freq)) +
    theme(legend.position = "none") +
    xlab("Test") +
    ggtitle(mytitle) +
    theme(plot.title = element_text(size = 10))
  return(p)
}


#####################
# DATA WRANGLING (DW)
#####################
# From the previous graphs, many variables as possible features are empty, invalid
# or not needed for this analysis.  Here further limit the data to the following:
#   * Single family units, built on-site
#   * Purposed for a home purchase
#   * Owner occupied
#   * Known Race and Sex
#   * Some level of income

loans <- loans %>%
  mutate(granted = as.numeric(recode(as.character(action_taken), "1" = "1", .default = "0")))

# Create a tidy loans data set
loans_tidy <- loans %>%
  filter(derived_dwelling_category == "Single Family (1-4 Units):Site-Built" & 
         loan_purpose == 1 & total_units == 1 & occupancy_type == 1 &
         applicant_race_1 != 6 & income > 0)  %>%
  mutate(granted = as.numeric(recode(as.character(action_taken), "1" = "1", .default = "0")),
         loanId = seq(1:n()), 
         msa = derived_msa_md,
         val_ratio = as.numeric(combined_loan_to_value_ratio) / 100,
         term = as.numeric(loan_term),  
         value = as.numeric(property_value),
         lv_ratio = as.numeric(combined_loan_to_value_ratio),
         rate = as.numeric(interest_rate), 
         sex = as.numeric(recode(as.character(applicant_sex),
               "1" = "1", "2" = "2", "6" = "3", .default = "4")),
         amount = as.numeric(loan_amount),
         inc_ratio = income / amount, 
         msa_minc = ffiec_msa_md_median_family_income, 
         msa_incper = tract_to_msa_income_percentage,
         race = as.numeric(recode(as.character(applicant_race_1),
                "1" = "1", "3" = "2", "21" = "2", "5" = "3", .default = "4")),
         age = as.numeric(recode(applicant_age,
               "<25" = "20", "25-34" = "30", "35-44" = "40", "45-54" = "50",
               "55-64" = "60", "65-74" = "70", ">74" = "75", "8888" = "-1")),
         debt_ratio = as.numeric(recode(debt_to_income_ratio,
                      "<20%" = "1", "20%-<30%" = "2", "30%-<36%" = "3",
                      "50%-60%" = "4", ">60%" = "5"))) %>%
  select(msa, msa_minc, msa_incper, term, value, lv_ratio,
         rate, income, sex, race, debt_ratio, age, amount, granted)

# Compute median values by metropolitan statistical area
# for features with missing data
loans_tidy$age <- ifelse(loans_tidy$age == "-1", NA, loans_tidy$age)
msa_values <- loans_tidy %>%
  group_by(msa) %>%
  summarize (term_med = median(term, na.rm=TRUE), 
            val_med = median(value, na.rm=TRUE), 
            rate_med = median(rate, na.rm=TRUE),
            inc_med = median(income,na.rm=TRUE), 
            lv_med = median(lv_ratio,na.rm=TRUE), 
            debt_med = median(debt_ratio, na.rm=TRUE),
            age_med = round(median(age, na.rm = TRUE), 0))
loans_tidy <- loans_tidy %>%
  left_join(msa_values, by = "msa")

# Replace NA values with msa median values
loans_tidy$term <- ifelse(is.na(loans_tidy$term), loans_tidy$term_med,  loans_tidy$term)
loans_tidy$value <- ifelse(is.na(loans_tidy$value), loans_tidy$val_med, loans_tidy$value)
loans_tidy$rate <- ifelse(is.na(loans_tidy$rate),loans_tidy$rate_med, loans_tidy$rate)
loans_tidy$income <- ifelse(is.na(loans_tidy$income), loans_tidy$inc_med, loans_tidy$income)
loans_tidy$income <- loans_tidy$income * 1000 # bring to same scale as amount and value
loans_tidy$lv_ratio <- ifelse(is.na(loans_tidy$lv_ratio),loans_tidy$lv_med, loans_tidy$lv_ratio)
loans_tidy$race <- ifelse(is.na(loans_tidy$race), 4, loans_tidy$race)
loans_tidy$debt_ratio <- ifelse(is.na(loans_tidy$debt_ratio), loans_tidy$debt_med,
                                loans_tidy$debt_ratio)
loans_tidy$age <- ifelse(is.na(loans_tidy$age), loans_tidy$age_med,
                                loans_tidy$age)

# Remove msa medians from loans_tidy
loans_tidy <- loans_tidy %>%
  select(-term_med, -val_med, -rate_med, -msa, -inc_med, -debt_med, -age_med, -lv_med)

# Set count of loans cleaned to the analysis criteria

loans_tidy_total <- nrow(loans_tidy)

######################
# EXPLORATORY ANALYSIS
######################

# Create feature description tibble
Feature <- c("amount", "age", "debt_ratio", "income", "lv_ratio",
             "msa_incper", "msa_minc", "race",
             "rate", "sex", "term", "value")
Description <- c("principle amount of the loan",
                 "age of the primary applicant",
                 "debt to income of the applicant",
                 "applicant income",
                 "loan to value ratio",
                 "ratio of income of the tract to the contained-in MSA",
                 "median income of the MSA",
                 "1:Indigenous, 2:African American, 3:White, 4:Other",
                 "interest rate of the mortgage",
                 "1:Male, 2:Female, 3:Other, 4:Not Provided",
                 "term of the loan in months", 
                 "value of the property mortgaged")
feature_desc <- tibble(Feature, Description)
rm(Feature, Description)

# Save percent loans granted & No of columns in the dataset
loans_tidy_granted <- mean(loans_tidy$granted)

# Create histogram plots of the tidy data  
plot_tidy_hist <- plot_histogram(loans_tidy)
# Create  boxplotss of the candidate features
plot_tidy_box <- plot_boxplot(loans_tidy, by =  "granted", 
             geom_boxplot_args = list("outlier.color" = "red"))

#####################
# DATA ANALYSIS: (DA)
#####################

# Create train, test, and verification data sets
set.seed(234, sample.kind="Rounding")
test_index <- createDataPartition(loans_tidy$granted, times = 1, p = 0.7, list = FALSE)
train_set <- loans_tidy[test_index,]

# intermediate set to be split into test and verification data
intermediate <- loans_tidy[-test_index,]
test_index <- createDataPartition(y = intermediate$granted, p = 0.5, list = F)
test_set <- intermediate[test_index,]
test_ver <- intermediate[-test_index,]
rm(intermediate)

# save number of columns in train set
train_cols <- ncol(train_set)

# PREDICTION MODELS
# Each model will follow the following execution pattern, with only some having
# a slight variation that will be noted within the specific model
#
# Common Process Steps for all models
# * Set a seed
# * Fit the model with the training data
# * Predict using the model from the test data
# * Collect the results into a tibble
# * Create a confusion matrix plot of the model
#
#   Note, xgBoos will follow the above pattern after 
#   additional data preparation steps

# PREDICTION MODEL: Random guessing
set.seed(234, sample.kind="Rounding")
# No training nor prediction needed for this model
pred_rand <- sample(0:1, size = nrow(test_set), prob = c(0.5, 0.5), replace = TRUE)
cm_rand <- confusionMatrix(factor(pred_rand), factor(test_set$granted),
                           positive = "1")
f1_rand <- cmf1(cm_rand)
results <- tibble(Method = "random guess", Accuracy = cm_rand$overall["Accuracy"],
                  F1 = f1_rand,
                  Improvement = 0.0)
cm_plot_rand_f1 <- ggplotConfusionMatrix(cm_rand)

# PREDICTION MODEL: Naive Bayes
set.seed(234, sample.kind="Rounding")
fit_nb <- naiveBayes(granted ~ ., data = train_set)
y <- predict(fit_nb, test_set)
pred_nb <- as.numeric(as.character(y))
cm_nb <- confusionMatrix(factor(pred_nb), factor(test_set$granted),
                         positive = "1")
results <- bind_rows(results,
                  tibble(Method = "naive bayes", Accuracy = cm_nb$overall["Accuracy"],
                         F1 = cmf1(cm_nb),
                         Improvement = (cm_nb$overall["Accuracy"] - cm_rand$overall["Accuracy"]) /
                                        cm_rand$overall["Accuracy"]))
cm_plot_nb_f1 <- ggplotConfusionMatrix(cm_nb)

# To show the non-independence of the features, create a regression plot
plot_nb_reg <- test_set %>%
  ggplot (aes(x = amount/1000,  y = value/1000)) +
  geom_point(color = "grey") +
  geom_smooth(method = "lm", formula = y ~ x) +
  xlab("Loan Amount (1000's)") + ylab("Property Value (1000's")
plot_nb_reg 

# PREDICTION MODEL: k nearest neighbors
set.seed(234, sample.kind="Rounding")
# Scale data as knn operates based on neighbor distance
train_scale <- scale(train_set[-train_cols])
test_scale <- scale(test_set[-train_cols])
pred_knn <- knn(train_scale, test_scale, 
               cl = factor(train_set$granted),
               k = 5)
cm_knn <- confusionMatrix(pred_knn, factor(test_set$granted),
                    positive = "1")
results <- bind_rows(results,
                     tibble(Method = "knn", Accuracy = cm_knn$overall["Accuracy"],
                            F1 = cmf1(cm_knn),
                            Improvement = (cm_knn$overall["Accuracy"] - cm_rand$overall["Accuracy"]) /
                              cm_rand$overall["Accuracy"]))
cm_plot_knn_f1 <- ggplotConfusionMatrix(cm_knn)

rm (train_scale, test_scale)

# PREDICTION MODEL: decision tree
set.seed(234, sample.kind="Rounding")
fit_rpart <- train(factor(granted) ~ ., data = train_set,
                    method = "rpart",
                    parms = list(split = "information"))
y <- predict(fit_rpart, test_set)
cm_rpart <- confusionMatrix(y, factor(test_set$granted),
                          positive = "1")
results <- bind_rows(results,
                     tibble(Method = "rpart", Accuracy = cm_rpart$overall["Accuracy"],
                            F1 = cmf1(cm_rpart),
                            Improvement = (cm_rpart$overall["Accuracy"] - cm_rand$overall["Accuracy"]) /
                              cm_rand$overall["Accuracy"]))
cm_plot_rpart_f1 <- ggplotConfusionMatrix(cm_rpart)
cm_plot_rpart_f1

# PREDICTION MODEL: random forest
set.seed(234, sample.kind="Rounding")
fit_rf <- randomForest(granted ~ ., data = train_set, ntree=300)
pred_rf <- round(predict(fit_rf, test_set))
cm_rf <- confusionMatrix(factor(pred_rf), factor(test_set$granted),
                           positive = "1")
results <- bind_rows(results,
                     tibble(Method = "rf", Accuracy = cm_rf$overall["Accuracy"],
                            F1 = cmf1(cm_rf),
                            Improvement = (cm_rf$overall["Accuracy"] - cm_rand$overall["Accuracy"]) /
                            cm_rand$overall["Accuracy"]))
cm_plot_rf_f1 <- ggplotConfusionMatrix(cm_rf)


# PREDICTION MODEL: xgBoost
set.seed(234, sample.kind="Rounding")
# create training matrix for xgbboost from train_set
train_xgb <- xgb.DMatrix(as.matrix(train_set[-train_cols]), 
  label = as.numeric(train_set$granted))
# create test matrix for xgboost from test_set
test_xgb <- xgb.DMatrix(as.matrix(test_set[-train_cols]), 
  label = as.numeric(test_set$granted))
# create verification data set for xgboost from verification set
ver_xgb <- xgb.DMatrix(as.matrix(test_ver[-train_cols]),
  label = as.numeric(as.character(test_ver$granted)))
# Train the XGBoost Model
p_xgb <- list(objective = "binary:logistic", 
         eta = 0.3, max.depth = 6, nthread = 3, 
         eval_metric = "rmse", gamma = 1.1)

# Here the previously mentioned processing model begins for xgBoost
fit_xgb <- xgb.train(data = train_xgb, params = p_xgb, 
           watchlist = list(test = test_xgb, cv = ver_xgb), 
           nrounds = 200, early_stopping_rounds = 50, print_every_n = 20)
y <- predict(fit_xgb, newdata = as.matrix(test_set[-train_cols]), 
             ntreelimit = fit_xgb$bestInd)
y_hat <- round(y)
cm_xgb <- confusionMatrix(factor(y_hat), factor(test_set$granted),
           positive = "1")
results <- bind_rows(results,
           tibble(Method = "xgboost", Accuracy = cm_xgb$overall["Accuracy"],
                  F1 = cmf1(cm_xgb),
                  Improvement = (cm_xgb$overall["Accuracy"] - cm_rand$overall["Accuracy"]) /
                    cm_rand$overall["Accuracy"]))
cm_plot_xgb_f1 <- ggplotConfusionMatrix(cm_xgb)

# Create a variable importance plot from the xgBoost model 
imp_xgb <- xgb.importance(colnames(train_set[-train_cols]),
           model = fit_xgb)
plot_xgb_data <- xgb.plot.importance(imp_xgb, rel_to_first = TRUE, plot = FALSE)
plot_xgb_imp <- plot_xgb_data %>%
  ggplot(aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity") +
  expand_limits(y = c(0.0, 0.6)) +
  geom_label(aes(label = round(Gain, 3)), nudge_y = +0.06, size = 2) + 
  coord_flip() +
  xlab("Feature")

##############
# VERIFICATION
##############
# Random Forest verification
set.seed(234, sample.kind="Rounding")
y_hat <- round(predict(fit_rf, test_ver))
cm_verrf <- confusionMatrix(factor(y_hat),  factor(test_ver$granted),
                             positive = "1")
vresults <- tibble(Method = "Ver rf",
                            Accuracy = cm_verrf$overall["Accuracy"],
                            F1 = cmf1(cm_verrf))
cm_plot_vrf_f1 <- ggplotConfusionMatrix(cm_xgb)

# xgBoost verification
set.seed(234, sample.kind="Rounding")
y_hat <- round(predict(fit_xgb, newdata = as.matrix(test_ver[-train_cols]), 
             ntreelimit = fit_xgb$bestInd))
cm_verxgb <- confusionMatrix(factor(y_hat), 
              factor(test_ver$granted),
              positive = "1")
vresults <- bind_rows(vresults,
                     tibble(Method = "Ver xgb",
                            Accuracy = cm_verxgb$overall["Accuracy"],
                            F1 = cmf1(cm_verxgb)))
cm_plot_vxgb_f1 <- ggplotConfusionMatrix(cm_verxgb)


# ######################
# DATA SAVE AND CLEAN-UP
# ######################
# Remove large data from the environment
rm(loans)
# Save the Global Image so the Knit'ed markdown can use it
# to create the report
save.image(file = 'loans.RData')
# Compute and print the total time for header comments
dur <- Sys.time() - start
print(dur)