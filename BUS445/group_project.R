# Load the required libraries
library(dplyr)
library(ggplot2)
library(leaps)
library(HH)
library(MASS)
library(ROSE)
library(randomForest)
library(cowplot)
library(nnet)
library(pROC)
library(car)
library(caret)
library(corrplot)
library(cowplot)
library("rpart") 
library("rpart.plot")
source("BCA_functions_source_file.R")

bank_data <-  read.csv("bank-additional-full.csv", sep = ";")

# Convert to factors
bank_data$job <- as.factor(bank_data$job)
bank_data$marital <- as.factor(bank_data$marital)
bank_data$education <- as.factor(bank_data$education)
bank_data$default <- as.factor(bank_data$default)
bank_data$housing <- as.factor(bank_data$housing)
bank_data$loan <- as.factor(bank_data$loan)
bank_data$contact <- as.factor(bank_data$contact)
bank_data$month <- as.factor(bank_data$month)
bank_data$day_of_week <- as.factor(bank_data$day_of_week)
bank_data$poutcome <- as.factor(bank_data$poutcome)
bank_data$y <- as.factor(bank_data$y)


# Distribution for target variable
ggplot(bank_data, aes(x = y, fill = as.factor(y))) +
  geom_bar() +
  labs(title = "Distribution of Target Variable (y)", x = "Subscription", y = "Count") +
  scale_fill_manual(values = c("skyblue", "orange"))


#--------------------------------------------------------------

# Balanced Sampling using ROSE

set.seed(445)
balanced_data <- ovun.sample(y ~ ., data = bank_data, method = "over")$data

# Improved distribution for target variable
ggplot(balanced_data, aes(x = y, fill = as.factor(y))) +
  geom_bar() +
  labs(title = "Distribution of Target Variable (y)", x = "Subscription", y = "Count") +
  scale_fill_manual(values = c("skyblue", "orange"))


split <- createDataPartition(balanced_data$y, p = 0.7, list = FALSE)
train_balanced_data <- balanced_data[split, ]
test_balanced_data <- balanced_data[-split, ]

#--------------------------------------------------------------

# Correlation plot for numeric variables
cor_matrix <- cor(train_balanced_data %>% select_if(is.numeric), use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8, 
         number.cex = 0.7, addCoef.col = "black")


# Perform PCA
predictors <- train_balanced_data[, c("euribor3m", "nr.employed", "emp.var.rate")]
pca <- prcomp(predictors, center = TRUE, scale. = TRUE)

# Use the first principal component (PC1)
train_balanced_data$PC1 <- pca$x[, 1]

# Refit the model with PC1
pca_model <- glm(y ~ PC1 + cons.price.idx, 
                 family = "binomial", data = train_balanced_data)
vif(pca_model)

# Update the test data with PC1
test_predictors <- test_balanced_data[, c("euribor3m", "nr.employed", "emp.var.rate"), drop = FALSE]
test_balanced_data$PC1 <- predict(pca, newdata = test_predictors)[, 1]

#--------------------------------------------------------------

# Logistic Regression

logit_model <- glm(y ~ . - euribor3m - nr.employed - emp.var.rate - duration , data = train_balanced_data, family = "binomial")
summary(logit_model)

# Predict and evaluate Logistic Regression
logit_preds <- predict(logit_model, test_balanced_data, type = "response")
logit_preds_class <- ifelse(logit_preds > 0.5, "yes", "no")

# Confusion Matrix
confusion_matrix <- table(test_balanced_data$y, logit_preds_class, dnn = c("Actual", "Predicted"))
confusion_matrix

# ROC-AUC
log_roc_curve <- roc(test_balanced_data$y, logit_preds, levels = rev(levels(test_balanced_data$y)))
auc(log_roc_curve)

#### AUC: 0.7914

#--------------------------------------------------------------

# Random Forest

class_weights <- prop.table(table(train_balanced_data$y))

rf_model <- randomForest(
  y ~ . - euribor3m - nr.employed - emp.var.rate - duration, 
  data = train_balanced_data, 
  ntree = 500, 
  mtry = sqrt(ncol(train_balanced_data) - 1), 
  classwt = as.vector(1 / class_weights)
)
rf_model

predictions <- predict(rf_model, test_balanced_data, type = "response")
prob_predictions <- predict(rf_model, test_balanced_data, type = "prob")[, "yes"]

# Confusion Matrix
confusionMatrix(predictions, test_balanced_data$y, positive = "yes")


# ROC-AUC
rf_roc_curve <- roc(test_balanced_data$y, prob_predictions, levels = rev(levels(test_balanced_data$y)))
auc(rf_roc_curve)

#### AUC: 0.9938

#--------------------------------------------------------------

# Stepwise selection

step_model <- stepAIC(logit_model, direction = "both", trace = FALSE)
summary(step_model)

# Predict and evaluate Logistic Regression
step_preds <- predict(logit_model, test_balanced_data, type = "response")
step_preds_class <- ifelse(logit_preds > 0.5, "yes", "no")

# Confusion Matrix
confusion_matrix <- table(test_balanced_data$y, step_preds_class, dnn = c("Actual", "Predicted"))
confusion_matrix

# ROC-AUC
step_roc_curve <- roc(test_balanced_data$y, step_preds, levels = rev(levels(test_balanced_data$y)))
auc(step_roc_curve)

#### AUC: 0.7914

#--------------------------------------------------------------

# Neural Network

set.seed(445)
nn_model <- nnet(
  y ~ . - euribor3m - nr.employed - emp.var.rate - duration, 
  data = train_balanced_data, 
  size = 4, decay = 0.10, maxit = 1000)

nn_model$value
summary(nn_model)

# Predict and evaluate Neural Network
nn_preds <- predict(nn_model, test_balanced_data, type = "class")
nn_preds_prob <- predict(nn_model, test_balanced_data, type = "raw")

# Confusion Matrix
confusionMatrix(as.factor(nn_preds), test_balanced_data$y, positive = "yes")

# ROC-AUC
nn_roc_curve <- roc(test_balanced_data$y, nn_preds_prob, levels = rev(levels(test_balanced_data$y)))
auc(nn_roc_curve)

#### AUC: 0.8105

#--------------------------------------------------------------

# Trees

# Fit the model
tree_model <- rpart(y ~ . - euribor3m - nr.employed - emp.var.rate - duration, data = train_balanced_data, method = "class")

# Plot the tree
rpart.plot(tree_model, extra = 101, type = 5, fallen.leaves = TRUE, under = TRUE, faclen = 0)

# Predict and evaluate the model
tree_preds <- predict(tree_model, test_balanced_data, type = "class")
tree_preds_prob <- predict(tree_model, test_balanced_data, type = "prob")

# Confusion Matrix
confusionMatrix(as.factor(tree_preds), test_balanced_data$y, positive = "yes")

# ROC-AUC
tree_roc_curve <- roc(test_balanced_data$y, tree_preds_prob[, "yes"], levels = rev(levels(test_balanced_data$y)))
auc(tree_roc_curve)

#### AUC: 0.7445


#--------------------------------------------------------------

# Lift Chart

# Validation data: test_balanced_data
# Estimation data: train_balanced_data

lift.chart(modelList = c("logit_model", "rf_model", "step_model", "nn_model", "tree_model"),
           data = train_balanced_data,
           targLevel = "yes",
           trueResp = 0.11,
           type = "cumulative",
           sub = "Estimation")

lift.chart(modelList = c("nn_model", "rf_model", "logit_model", "step_model", "tree_model"),
           data = test_balanced_data, 
           targLevel = "yes",
           trueResp = 0.11,
           type = "cumulative",
           sub = "Validation")

lift.chart(modelList = c("logit_model", "rf_model", "step_model", "nn_model", "tree_model"),
           data = train_balanced_data,
           targLevel = "yes",
           trueResp = 0.11,
           type = "incremental",
           sub = "Estimation")

lift.chart(modelList = c("nn_model", "rf_model", "logit_model", "step_model", "tree_model"),
           data = test_balanced_data,
           targLevel = "yes",
           trueResp = 0.01,
           type = "incremental",
           sub = "Validation")


#--------------------------------------------------------------

# Plot ROC curves
plot(log_roc_curve, col = "blue", main = "ROC Curve Comparison")
plot(rf_roc_curve, add = TRUE, col = "green")
plot(step_roc_curve, add = TRUE, col = "red")
plot(nn_roc_curve, add = TRUE, col = "orange")
plot(tree_roc_curve, add = TRUE, col = "purple")
legend("bottomright", legend = c("Logistic Regression", "Random Forest","Stepwise Selection", "Neural Network", "Trees"), col = c("blue", "green", "red", "orange","purple"), lty = 1)

#--------------------------------------------------------------


# Feature Importance
importance <- importance(rf_model)
varImpPlot(rf_model, main = "Feature Importance")


#--------------------------------------------------------------

# Visuals to support the recommendation

# Age

# Create age groups (e.g., 10-year intervals)
bank_data <- bank_data %>%
  mutate(age_group = cut(age, breaks = seq(10, 100, by = 10), right = FALSE))

# Calculate percentage of 'yes' subscriptions per age group
age_subscription_stats <- bank_data %>%
  group_by(age_group) %>%
  summarise(
    total = n(),
    subscribed_yes = sum(y == "yes"),
    percentage_yes = (subscribed_yes / total) * 100
  ) %>%
  arrange(desc(percentage_yes))


plot1 <- ggplot(age_subscription_stats, aes(x = age_group, y = percentage_yes)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(
    x = "Age Group",
    y = "Percentage of Subscriptions (%)"
  ) +
  theme_minimal()

# Job
job_subscription_stats <- bank_data %>%
  group_by(job) %>%
  summarise(
    total = n(),
    subscribed_yes = sum(y == "yes"),
    percentage_yes = (subscribed_yes / total) * 100
  ) %>%
  arrange(desc(percentage_yes))


plot2 <- ggplot(job_subscription_stats, aes(x = reorder(job, percentage_yes), y = percentage_yes)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(
    x = "Job",
    y = "Percentage of Subscriptions (%)"
  ) +
  theme_minimal()

# Education

# Calculate percentages for each category
plot_data <- bank_data %>%
  group_by(.data[["education"]]) %>%
  summarise(
    total_count = n(),
    yes_count = sum(y == "yes"),
    percentage_yes = (yes_count / total_count) * 100
  ) %>%
  arrange(desc(percentage_yes))  # Optional: Sort categories by percentage of "yes"

# Plot
plot3 <- ggplot(plot_data, aes(x = reorder(.data[["education"]], -percentage_yes), y = percentage_yes, fill = .data[["education"]])) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  labs(
    x = "Education Level",
    y = "Percentage of Subscriptions (%)"
  ) +
  scale_y_continuous(labels = scales::percent_format(scale = 1)) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# Combine the plots into a grid
combined_plot <- plot_grid(
  plot1, plot2, plot3, 
  ncol = 3, 
  label_size = 14
)

# Add subtitles below each plot
subtitles <- plot_grid(
  textGrob("a. Percentage of Subscriptions by Age Group", gp = gpar(fontsize = 12)), 
  textGrob("b. Percentage of Subscriptions by Job Category", gp = gpar(fontsize = 12)), 
  textGrob("c. Percentage of Subscriptions by Education Level", gp = gpar(fontsize = 12)), 
  ncol = 3
)

# Combine plots and subtitles
plots_with_subtitles <- plot_grid(
  combined_plot,
  subtitles,
  ncol = 1,
  rel_heights = c(0.85, 0.15) # Adjust relative heights
)

# Add a main title at the very top
final_plot <- plot_grid(
  ggdraw() + draw_label("Figure 1: Demographic Distributions", size = 16, fontface = "bold", hjust = 0.5),
  plots_with_subtitles,
  ncol = 1,
  rel_heights = c(0.1, 0.9) # Title takes less space
)

final_plot

#--------------------------------------------------------------

# Macro-economic Trends + Campaign Data

# List of categorical variables
categorical_vars <- c("month")

for (var in categorical_vars) {
  # Calculate percentages for each category
  plot_data <- bank_data %>%
    group_by(.data[[var]]) %>%
    summarise(
      total_count = n(),
      yes_count = sum(y == "yes"),
      percentage_yes = (yes_count / total_count) * 100
    ) %>%
    arrange(desc(percentage_yes))  # Optional: Sort categories by percentage of "yes"
  
  # Plot
  plot4 <- ggplot(plot_data, aes(x = reorder(.data[[var]], -percentage_yes), y = percentage_yes, fill = .data[[var]])) +
    geom_bar(stat = "identity", show.legend = FALSE) +
    labs(
      x = toupper(var),
      y = "Percentage of Subscriptions (%)"
    ) +
    scale_y_continuous(labels = scales::percent_format(scale = 1)) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
}

create_plot <- function(var, bin_width) {
  # Generate bin edges with extended coverage
  num <-  length(var)
  breaks <- seq(floor(min(bank_data[[var]]) - bin_width), ceiling(max(bank_data[[var]]) + bin_width), by = bin_width)
  
  # Create bins and calculate percentage of 'yes' responses
  plot_data <- bank_data %>%
    mutate(binned_var = cut(.data[[var]], breaks = breaks, include.lowest = TRUE)) %>%
    group_by(binned_var) %>%
    summarise(
      total_count = n(),
      yes_count = sum(y == "yes"),
      percentage_yes = (yes_count / total_count) * 100
    ) %>%
    filter(!is.na(binned_var))  # Remove any remaining NA bins
  
   ggplot(plot_data, aes(x = binned_var, y = percentage_yes)) +
    geom_bar(stat = "identity", fill = "skyblue") +
    labs(
      x = toupper(var),
      y = "Percentage of Subscriptions (%)"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
}

plot5 <- create_plot("euribor3m", 0.5)
plot6 <- create_plot("emp.var.rate", 0.5)
plot7 <- create_plot("cons.conf.idx", 5)
plot8 <- create_plot("campaign", 1)
# plot9 <- create_plot("nr.employed", 50)
# plot10 <- create_plot("cons.price.idx", 0.5)
# plot11 <- create_plot("pdays", 10)
# plot12 <- create_plot("previous", 1)


# Combine the plots into a grid
combined_plot <- plot_grid(
  plot4, plot8,
  ncol = 2, 
  label_size = 14
)

# Add subtitles below each plot
subtitles <- plot_grid(
  textGrob("a. Percentage of Subscriptions by Last Contact Month", gp = gpar(fontsize = 12)), 
  textGrob("b. Percentage of Subscriptions by Total Campaign Contacts", gp = gpar(fontsize = 12)), 
  ncol = 2
)

# Combine plots and subtitles
plots_with_subtitles <- plot_grid(
  combined_plot,
  subtitles,
  ncol = 1,
  rel_heights = c(0.85, 0.15) # Adjust relative heights
)

# Add a main title at the very top
final_plot2 <- plot_grid(
  ggdraw() + draw_label("Figure 3: Campaign Data", size = 16, fontface = "bold", hjust = 0.5),
  plots_with_subtitles,
  ncol = 1,
  rel_heights = c(0.1, 0.9) # Title takes less space
)

final_plot2


# Combine the plots into a grid
combined_plot <- plot_grid(
  plot5, plot6, plot7,
  ncol = 3, 
  label_size = 14
)

# Add subtitles below each plot
subtitles <- plot_grid(
  textGrob("a. Percentage of Subscriptions by Interbank Lending Rate ", gp = gpar(fontsize = 12)), 
  textGrob("b. Percentage of Subscriptions by Employment Variation Rate", gp = gpar(fontsize = 12)),
  textGrob("c. Percentage of Subscriptions by Consumer Confidence Index", gp = gpar(fontsize = 12)), 
  ncol = 3
)

# Combine plots and subtitles
plots_with_subtitles <- plot_grid(
  combined_plot,
  subtitles,
  ncol = 1,
  rel_heights = c(0.85, 0.15) # Adjust relative heights
)

# Add a main title at the very top
final_plot3 <- plot_grid(
  ggdraw() + draw_label("Figure 2: Macro-economic Trends", size = 16, fontface = "bold", hjust = 0.5),
  plots_with_subtitles,
  ncol = 1,
  rel_heights = c(0.1, 0.9) # Title takes less space
)

final_plot3
