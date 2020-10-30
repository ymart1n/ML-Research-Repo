
using DataFrames
using DecisionTree

# Agreement and correlation conditions
const ks_cutoff  = 0.09
const cvm_cutoff = 0.002

function the_roc(p_labels::Array{Bool, 1}, 
                 p_data::Array{Float64, 1}, 
                 p_weights::Array{Float64, 1})
    local score_index_ordered = sortperm(p_data, rev=true)
    local labels_bool = p_labels[score_index_ordered]
    local data_ordered = p_data[score_index_ordered]
    local weights_ordered = p_weights[score_index_ordered]

    # Find threshold
    local distinct_value_indices = find(convert(Array{Bool, 1}, abs(diff(data_ordered)) .> 1.0e-08))
    local threshold_idxs = [distinct_value_indices; length(labels_bool)]
    # Accumulate true positives
    local tps = copy(weights_ordered)
    tps[find(!labels_bool)] = 0.0
    tps = cumsum(tps)[threshold_idxs]
    local fps = cumsum(weights_ordered)[threshold_idxs] .- tps
    thresholds = data_ordered[threshold_idxs]

    if length(tps) == 0 || fps[1] == 0.0
        tps = [0.0; tps]
        fps = [0.0; fps]
        thresholds = [thresholds[1] + 1.0; thresholds]
    end

    if fps[end] <= 0.0
        error("No negative samples in y_true, false positive value should be meaningless")
    else
        fpr = fps ./ fps[end]
    end

    if tps[end] <= 0.0
        error("No positive samples in y_true, true positive value should be meaningless")
    else
        tpr = tps ./ tps[end]
    end

    return fpr, tpr, thresholds
end

# From evaluation.py
function _roc_curve_splitted(data_zero::Array{Float64,1},
                             data_one::Array{Float64,1}, 
                             sample_weights_zero::Array{Float64,1},
                             sample_weights_one::Array{Float64,1})
#=                             
 Compute roc curve

 data_zero: 0-labeled data
 data_one:  1-labeled data
 sample_weights_zero: weights for 0-labeled data
 sample_weights_one:  weights for 1-labeled data
 return: roc curve
=#
    local labels = convert(Array{Bool, 1}, [zeros(data_zero); ones(data_one)])
    local data_all = [data_zero; data_one]
    local weights = [sample_weights_zero; sample_weights_one]

    fpr, tpr = the_roc(labels, data_all, weights)

    return fpr, tpr
end

function compute_ks(p_data_prediction::Array{Float64,1},
                    p_mc_prediction::Array{Float64,1},
                    p_weights_data::Array{Float64,1},
                    p_weights_mc::Array{Float64,1})
#=                    
 Compute Kolmogorov-Smirnov (ks) distance between real data predictions cdf and Monte Carlo one.

 data_prediction: array-like, real data predictions
 mc_prediction: array-like, Monte Carlo data predictions
 weights_data: array-like, real data weights
 weights_mc: array-like, Monte Carlo weights
 returns: ks value
=#
    @assert length(p_data_prediction) == length(p_weights_data)
    @assert length(p_mc_prediction) == length(p_weights_mc)
    @assert reduce(&, p_data_prediction[:,1] .>= 0.0) && reduce(&, p_data_prediction[:,1] .<= 1.0)
    @assert reduce(&, p_mc_prediction[:,1] .>= 0.0) && reduce(&, p_mc_prediction[:,1] .<= 1.0)

    p_weights_data = p_weights_data ./ sum(p_weights_data)
    p_weights_mc = p_weights_mc ./ sum(p_weights_mc)

    fpr, tpr = _roc_curve_splitted(p_data_prediction, p_mc_prediction, p_weights_data, p_weights_mc)

    Dnm = maximum(abs(fpr .- tpr))

    return Dnm
end

function _rolling_window(p_data::Array{Int64, 1}, 
                         p_window_size::Int64)
#=
 Rolling window: take window with definite size through the array

 data: array-like
 window_size: size
 return: the sequence of windows
 Example: data = array(1, 2, 3, 4, 5, 6), window_size = 4
     Then this function return array(array(1, 2, 3, 4), array(2, 3, 4, 5), array(3, 4, 5, 6))
=#
    return [ p_data[i:i+p_window_size-1] for i in 1:length(p_data)-p_window_size ]
end

function _bincount(p_data::Array{Int64, 1}; p_weights=0, p_minlength=0)
    sz = maximum(p_data) + 1
    sz = ifelse(p_minlength > sz, p_minlength, sz)
    ret = zeros(Int64, sz)
    for i in p_data
        idx = i + 1
        ret[idx] = ret[idx] + 1
    end
    return ret
end

function _cvm(subindices, 
              total_events::Int64)
#=              
 Compute Cramer-von Mises metric.
 Compared two distributions, where first is subset of second one.
 Assuming that second is ordered by ascending

 subindices: indices of events which will be associated with the first distribution
 total_events: count of events in the second distribution
 return: cvm metric
=#
    local target_distribution = collect(1:total_events) ./ total_events
    local subarray_distribution = cumsum(_bincount(subindices, p_minlength=total_events))
    subarray_distribution ./= subarray_distribution[end]
    return mean((target_distribution .- subarray_distribution) .^ 2)
end

function compute_cvm(p_predictions::Array{Float64,1},
                     p_masses::Array{Float64,1},
                     n_neighbours=200,
                     step=50)
#=                     
 Computing Cramer-von Mises (cvm) metric on background events: take average of cvms calculated for each mass bin.
 In each mass bin global prediction's cdf is compared to prediction's cdf in mass bin.

 predictions: array-like, predictions
 masses: array-like, in case of Kaggle tau23mu this is reconstructed mass
 n_neighbours: count of neighbours for event to define mass bin
 step: step through sorted mass-array to define next center of bin
 returns: average cvm value
=#
    @assert length(p_predictions) == length(p_masses)
    # First, reorder by masses
    local predictions_by_mass = p_predictions[sortperm(p_masses)]
    # Second, replace probabilities with order of probability among other events
    local predictions_ordered = sortperm(sortperm(predictions_by_mass))
    # Now, each window forms a group, and we can compute contribution of each group to CvM
    rw = _rolling_window(predictions_ordered, n_neighbours)
    cvms = Array(Float64, 1)
    for window in rw[1:end:step]
        push!(cvms, _cvm(window, length(predictions_ordered)))
    end
    return mean(cvms)
end
# /From evaluation.py

println("Start!")
println()
versioninfo()
println()
println("===> Loading files")
println("Load training data...")
train = readtable("train.csv", quotemark=Char[])
# Build the model over training data
# <MODEL>
#
#choose features:
#
#features = convert(Array, train[:,[:LifeTime, :FlightDistance, :pt, :isolationa]])
#
features = convert(Array{Float64, 2}, train[:,2:4])
println("--------------------------------")
println("Features $(names(train[:,2:4]))")
println("--------------------------------")
labels = convert(Array{Float64,1}, train[:admit])

println("Building model...")
println("--------------------------------")
# Random Forest
@time model = build_forest(labels, features, 3, 50, 1.0)
show(stdout, model)
println()
println("--------------------------------")
println("Model ok!")
# </MODEL>

# # Agreement Test
# println("Load check_agreement...")
# check_agreement = readtable("../input/check_agreement.csv", quotemark=Char[])
# println("Agreement Test")
# features = convert(Array{Float64, 2}, check_agreement[:,2:31])
# agreement_probs = apply_forest(model, features)
# ks = compute_ks(
#     agreement_probs[find(check_agreement[:signal] .==  0)],
#     agreement_probs[find(check_agreement[:signal] .==  1)],
#     check_agreement[find(check_agreement[:signal] .==  0),:weight].data,
#     check_agreement[find(check_agreement[:signal] .==  1),:weight].data
# )
# if ks < ks_cutoff
#     println("The model passed the agreement test with $ks < $ks_cutoff")
# else
#     println("The model failed the agreement test with $ks >= $ks_cutoff")
#     exit(1)
# end

# Correlation Test
# println("Load check_correlation...")
# check_correlation = readtable("../input/check_correlation.csv", quotemark=Char[])
# println("Correlation Test")
# features = convert(Array{Float64, 2}, check_correlation[:,2:31])
# correlation_probs = apply_forest(model, features)
# cvm = compute_cvm(correlation_probs, check_correlation[:mass].data)
# if cvm < cvm_cutoff
#     println("The model passed the correlation test with $cvm < $cvm_cutoff.")
# else
#     println("The model failed the correlation test with $cvm >= $cvm_cutoff.")
#     exit(2)
# end

# Make predictions
println("Loading test data...")
test = readtable("test.csv")
println("Make predictions")
features = convert(Array{Float64, 2}, test[:,2:4])
@time predictions = apply_forest(model, features)
# test[:prediction] = predictions

prediction_class = [if x < 0.5 0 else 1 end for x in predictions];

prediction_df = DataFrame(y_actual = test.admit, y_predicted = prediction_class, prob_predicted = predictions);
prediction_df.correctly_classified = prediction_df.y_actual .== prediction_df.y_predicted

# Accuracy Score
accuracy = mean(prediction_df.correctly_classified)
print("Accuracy of the model is : ",accuracy)

# # Submit
# out = test[:,[:id,:prediction]]
# println("Saving the result...")
# writetable("jsubmission.csv", out, separator=',', header=true)
# println("The End.")
