module Model

using Flux
using GraphNeuralNetworks

struct GNNCellTypeEstimatorModel
    chain::Chain
end

Flux.@layer :expand GNNCellTypeEstimatorModel

__precompile__(false)
function GNNCellTypeEstimatorModel(hidden_channels)
    nn1 = Dense(1 => hidden_channels, relu)
    conv_layer = GINConv(nn1, 0.001f0)
    hidden1 = HeteroGraphConv((:gene, :to, :term) => conv_layer)

    nn2 = Dense(hidden_channels => 1, relu; bias=false)
    conv_layer2 = GINConv(nn2, 0.001f0)
    hidden2 = HeteroGraphConv((:term, :to, :term) => conv_layer2)

    conv_layer3 = GATv2Conv(1 => 1, relu)
    hidden3 = HeteroGraphConv((:term, :to, :term) => conv_layer3)

    chain = Chain(;
                  hidden1=hidden1,
                  hidden2=hidden2,
                  hidden3=hidden3)
    return GNNCellTypeEstimatorModel(chain)
end

function (model::GNNCellTypeEstimatorModel)(g::GNNHeteroGraph)
    x = (gene=hcat(g[:gene].exprs)', term=hcat(g[:term].proportion)')

    l = model.chain.layers

    x = l.hidden1(g, x)
    x = l.hidden2(g, x)
    x = l.hidden3(g, x)

    return x.term'
end

end # module