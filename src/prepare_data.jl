# This module is aimed at preparing data for the GNNCellTypeEstimator model.
# The module export the following functions:
# - prepare_data: Prepare the data for the GNNCellTypeEstimator model
module DataPreparation

export prepare_data, DataPreparationStrategy

using Flux
using MLUtils
using DataFrames
using GraphNeuralNetworks
using ExpressionData
using OntologyTrees
using Graphs
using MetaGraphs
using SyntheticExpressionMixtures
using Configurations

include("utils.jl")

@option struct DataPreparationStrategy
    split_at::Float64 = 0.8
    transform::Function = identity
    collate::Bool = true
    shuffle::Bool = true
end

# This function is to map the expression data of an expression set to a reference ontology tree    
# Creates a tree per sample in the expression set
function map_synthetic_data_to_tree(eset::ExpressionSet,
                                    onto_tree::OntologyTree;
                                    cell_type_col::String="cell type",)::Vector{OntologyTree}
    if !(cell_type_col in names(phenotype_data(eset)))
        error("The expression set is missing the cell type column: $cell_type_col")
    end

    trees = Vector{OntologyTree}()

    graph_genes = [props[:id]
                   for (_, props) in onto_tree.graph.vprops
                   if props[:type] == :gene]

    # Before going over the samples, we can reduce the expression matrix with only the relevant genes
    gx_data = expression_values(eset)
    graph_genes_gx_df = filter(:feature_names => x -> x in graph_genes, gx_data)

    # We can now create the expression matrix
    graph_genes_gx = Matrix(graph_genes_gx_df[!, Not(:feature_names)])

    cell_proportions = get_cell_type_proportions(eset, cell_type_col)

    for i in axes(cell_proportions, 1)
        tree = deepcopy(onto_tree)
        graph_genes_gx_i = graph_genes_gx[:, i]

        for (gene, expression) in zip(graph_genes, graph_genes_gx_i)
            add_expression!(tree.graph, gene, expression)
        end

        # Then we can add the cell type proportions and propagate them
        for (cell_type, proportion) in
            zip(names(cell_proportions[!, Not(:sample_id)]), cell_proportions[i, :])
            add_proportions!(tree.graph, cell_type, proportion)
        end

        propagate_cell_proportions!(tree.graph)

        push!(trees, tree)
    end

    return trees
end

function to_hetero_gnn(onto_tree::OntologyTree)
    graph = onto_tree.graph

    # To construct, the GNNHeteroGraph we need 2 vectors that describes the edges:
    # - one for the genes
    # - one for the cell types
    # These vectors needs to be in the same order
    # That, is (vector1[index1], vector2[index1]) 
    # represents the edge between the gene at index1 and the cell type at index1
    gene_index_vector = [v_index
                         for (v_index, v_props) in graph.vprops
                         if haskey(v_props, :type) && v_props[:type] == :gene]
    cell_index_ref_vector = [v_index
                             for (v_index, v_props) in graph.vprops
                             if haskey(v_props, :type) && v_props[:type] == :term]

    # In the hetero gnn, each node type is indexed separetely 
    # So we create 2 new vectors for each node type
    # We make a dict for each that represents the mapping
    # Then we an create the edge vector with the new indices 

    gene_indices_mapping = Dict([gene_index_vector[index] => index
                                 for index in
                                     eachindex(gene_index_vector)])
    cell_indices_mapping = Dict([cell_index_ref_vector[index] => index
                                 for index in eachindex(cell_index_ref_vector)])

    # Now we can create the edge vectors
    # We are gonna have a single dict where each entries is an edge type
    # The value of each entry is a tuple with the source and destination indices
    # of the edges
    edge_dict = Dict{NTuple{3,Symbol},Tuple{Vector{Int},Vector{Int},Vector{Float64}}}()

    # To find out the edges we need to iterate over the edges
    # and find the new indices of the nodes

    for edge in edges(graph)
        src_e = src(edge)
        dst_e = dst(edge)

        src_type = graph.vprops[src_e][:type]
        dst_type = graph.vprops[dst_e][:type]

        src_index = graph.vprops[src_e][:type] == :gene ?
                    gene_indices_mapping[src_e] :
                    cell_indices_mapping[src_e]

        dst_index = graph.vprops[dst_e][:type] == :gene ?
                    gene_indices_mapping[dst_e] :
                    cell_indices_mapping[dst_e]

        edge_type = (src_type, :to, dst_type)

        # If the edge type doesnt have a weight, set to 1
        if !haskey(graph.eprops, edge)
            graph.eprops[edge] = Dict(:weight => 1)
        end

        if haskey(edge_dict, edge_type)
            push!(edge_dict[edge_type][1], src_index)
            push!(edge_dict[edge_type][2], dst_index)
            push!(edge_dict[edge_type][3], graph.eprops[edge][:weight])
        else
            edge_dict[edge_type] = ([src_index], [dst_index], [graph.eprops[edge][:weight]])
        end
    end

    # Now we have to deal with the node features
    # We are gonna have a single dict where each entries is an node type
    # The value of each entry is a matrix with the features of the nodes
    ndata = Dict{Symbol,DataStore}()

    # Fill the datastore with the gene expression
    gene_expressions = [graph.vprops[gene_index][:expression]
                        for gene_index in gene_index_vector]

    # Transofrm the gene expressions using the log2
    gene_expressions = log2.(gene_expressions .+ 1)

    ndata[:gene] = DataStore(; exprs=gene_expressions)

    # Fill the datastore with 0 for the cell type proportion
    cell_proportions = [0.0 for index in cell_index_ref_vector]

    # Need to put 1.0 for the root term
    og_root_term_index = onto_tree.graph[onto_tree.base_term.obo_id, :id]

    cell_proportions[og_root_term_index] = 1.0

    ndata[:term] = DataStore(; proportion=cell_proportions)

    # We also need to pull the edge weights
    # We are gonna have a single dict where each entries is an edge type
    # The value of each entry is a vector with the edge weights
    edata = Dict{NTuple{3,Symbol},DataStore}()

    # Fill the datastore with the edge weights
    for edge_type in keys(edge_dict)
        edge = edge_dict[edge_type]
        edge_type = (edge_type[1], :to, edge_type[3])

        edata[edge_type] = DataStore(; weights=edge[3])
    end

    # Now we can create the GNNHeteroGraph
    g = GNNHeteroGraph(edge_dict; ndata=ndata, edata=edata)

    return g
end

function prepare_data(base_eset::ExpressionSet, onto_tree::OntologyTree;
                      syd_config::SYDConfig=SYDConfig(),
                      dataprep_strategy::DataPreparationStrategy=DataPreparationStrategy())
    # First transform (defaults to identity)
    eset = dataprep_strategy.transform(base_eset)

    # Generate synthetic data
    @info "Generating synthetic data"
    syd_eset = generate_synthetic_expression_mixtures(eset,
                                                      syd_config)

    # Map the synthetic data to the ontology tree
    @info "Mapping synthetic data to the ontology tree"
    trees = map_synthetic_data_to_tree(syd_eset, onto_tree; cell_type_col="cell_type")

    # Get the target labels from the synthetic data
    # -> note that we have to pass the tree to get the full list of cell types
    y = [(Float32.([prop[:proportion]
                    for prop in values(g.graph.vprops) if prop[:type] == :term]))
         for g in trees]
    y = reduce(hcat, y)

    @info "Creating the graph neural networks"
    gnn_trees = to_hetero_gnn.(trees)

    (; split_at, collate, shuffle) = dataprep_strategy

    train_data, test_data = getobs(splitobs((gnn_trees, y); at=split_at, shuffle=shuffle))

    # Create the data loaders
    train_loader = DataLoader(train_data; shuffle, collate)
    test_loader = DataLoader(test_data; shuffle, collate)

    return train_loader, test_loader
end

end # module