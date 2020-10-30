# Import Packages
using Pkg
using DataFrames
using CSV
using Plots
using GLM
using StatsBase
using Lathe
using MLBase
using ClassImbalance
using ROCAnalysis

# Enable printing of 1000 columns
ENV["COLUMNS"] = 1000

# Read the file using CSV.File and convert it to DataFrame
df = DataFrame(CSV.File("data.csv"))
first(df,5)

# Summary of dataframe
println(size(df))
describe(df)
# Output: (400, 14)

# Check column names
names(df)
# Output: 4-element Array{Symbol,1}:
#  :admit
#  :gre
#  :gpa
#  :rank

# Now let’s count the number of target classes or Y variable in the data set.
countmap(df.admit)
# Output: Dict{Int64,Int64} with 2 entries:
#   0 => 273
#   1 => 127

# Train test split
using Lathe.preprocess: TrainTestSplit
train, test = TrainTestSplit(df,.75);

# Train logistic regression model
# In order to build a logistic regression, family needs to choosen as `Binomial().
fm = @formula(admit ~ gre + gpa + rank)
logit = glm(fm, train, Binomial(), ProbitLink())

# Predict the target variable on test data 
prediction = predict(logit,test)

# Convert probability score to class
prediction_class = [if x < 0.5 0 else 1 end for x in prediction];

prediction_df = DataFrame(y_actual = test.admit, y_predicted = prediction_class, prob_predicted = prediction);
prediction_df.correctly_classified = prediction_df.y_actual .== prediction_df.y_predicted

# Accuracy Score
accuracy = mean(prediction_df.correctly_classified)
print("Accuracy of the model is : ",accuracy)
#> 0.711340206185567

# confusion_matrix = confusmat(2,prediction_df.y_actual, prediction_df.y_predicted)
confusion_matrix = MLBase.roc(prediction_df.y_actual, prediction_df.y_predicted)

# Import sklearn.metrics to julia
using PyCall
sklearn = pyimport("sklearn.metrics")

# Compute false positive rate, true positive rate and thresholds
fpr, tpr, thresholds = sklearn.roc_curve(prediction_df.y_actual, prediction_df.prob_predicted)

# Plot ROC curve
plot(fpr, tpr)
title!("ROC curve")

total = 0

for i in 1:10000
  using Lathe.preprocess: TrainTestSplit
  train, test = TrainTestSplit(df,.75);
  fm = @formula(admit ~ gre + gpa + rank)
  logit = glm(fm, train, Binomial(), ProbitLink())
  prediction = predict(logit,test)
  prediction_class = [if x < 0.5 0 else 1 end for x in prediction];

  prediction_df = DataFrame(y_actual = test.admit, y_predicted = prediction_class, prob_predicted = prediction);
  prediction_df.correctly_classified = prediction_df.y_actual .== prediction_df.y_predicted

  # Accuracy Score
  accu = mean(prediction_df.correctly_classified)
  total += accu
end 

# Avg Accuracy Score
avgAccuracy = total/10000
print("Average Accuracy of the model is : ", avgAccuracy)