module Training

export TrainingStrategy, gnn_loss_function, eval_loss, train!, train_model

using DataFrames
using MetaGraphs
using Graphs
using Flux
using MLUtils
using GraphNeuralNetworks
using ExpressionData
using OntologyTrees
using Configurations
using SyntheticExpressionMixtures
using JLD2
using RCall
using Statistics

import ..DataPreparation: prepare_data

@option struct TrainingStrategy
    epochs::Int = 200
    η::Float64 = 1e-5
    infotime::Int = 10
    loss_function::Function = gnn_loss_function
    convergence_tolerance::Float64 = 1e-9
    device::Function = Flux.cpu
end

function gnn_loss_function(ŷ, y; delta=1.0)
    return Flux.huber_loss(ŷ, y; delta)
end

function eval_loss(model, data_loader, device; loss_function=gnn_loss_function)
    loss = 0.0
    ntot = 0
    for (g, y) in data_loader
        n = length(y)
        ŷ = model(g)
        loss += loss_function(ŷ, y) * n
        ntot += n
    end
    return (loss = round(loss / ntot; digits=4))
end

function train!(model, (train_loader, test_loader);
                loss_history=[],
                predictions_history=[],
                training_strategy::TrainingStrategy=TrainingStrategy())
    (; epochs, η, infotime, convergence_tolerance, loss_function, device) = training_strategy

    model = device(model)
    opt = Flux.setup(Adam(η), model)

    function report(epoch, predictions_history)
        y_ = model(first(test_loader)[1])
        push!(predictions_history, y_)

        train = eval_loss(model, train_loader, device)
        test = eval_loss(model, test_loader, device)
        @info (; epoch, train, test)
    end

    report(0, predictions_history)
    for epoch in 1:epochs
        for (g, y) in train_loader
            grad = Flux.gradient(model) do model
                ŷ = model(g)
                return loss_function(ŷ, y)
            end

            Flux.update!(opt, model, grad[1])
        end

        push!(loss_history, eval_loss(model, train_loader, device))

        # If the loss has a variation of less than 10^-3 over the last 10 epochs, we can stop
        if length(loss_history) > 10 &&
           var(loss_history[(end - 10):end]) < convergence_tolerance
            @info "Stopping early: loss variance < $convergence_tolerance"
            @info "Final loss: $(loss_history[end])"
            return
        end

        epoch % infotime == 0 && report(epoch, predictions_history)
    end
end

function train_model(base_eset::ExpressionSet, onto_tree::OntologyTree;
                     hidden_channels::Int=24,
                     training_strategy::TrainingStrategy=TrainingStrategy(),
                     syd_config::SYDConfig=SYDConfig(),
                     model_name::String="GNNCellTypeDeconv",
                     output_path::String="output/models",)

    # Prepare the data
    train_loader, test_loader = prepare_data(base_eset,
                                             onto_tree;
                                             syd_config=syd_config)

    # Create the model
    model = GNNCellTypeEstimatorModel(hidden_channels)

    # Train the model
    loss_history = []
    predictions_history = []
    train!(model, (train_loader, test_loader); loss_history, predictions_history,
           training_strategy)

    return model, (; loss_history, predictions_history)
end

end