module GNNCellTypeEstimator

using MetaGraphs
using Graphs
using Flux
using ExpressionData
using GraphNeuralNetworks
using SyntheticExpressionMixtures
using OntologyLookup
using OntologyTrees
using Configurations
using DataFrames
using JLD2

include("utils.jl")

include("model.jl")
using .Model
export GNNCellTypeEstimatorModel

include("prepare_data.jl")
using .DataPreparation
export prepare_data, DataPreparationStrategy

include("train.jl")
using .Training
export TrainingStrategy, gnn_loss_function, eval_loss, train!, train_model

end
