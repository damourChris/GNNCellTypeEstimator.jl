using Test
using GNNCellTypeEstimator
using OntologyLookup
using ExpressionData
using JLD2
using SyntheticExpressionMixtures

# using MetaGraphs
# using Graphs
# using OntologyTrees
# using Configurations
# using DataFrames

# Helper function to create a dummy OntologyTree for testing
function create_dummy_ontologytree(cell_types_terms_ids::Vector{String}=["CL_0009051",
                                                                         "CL_0000988"])
    dummy_onto_tree_path = joinpath(@__DIR__, "data", "test_onto_tree.jld2")

    if isfile(dummy_onto_tree_path)
        return load(dummy_onto_tree_path, "onto_tree")
    end

    base_term = onto_term("cl", "http://purl.obolibrary.org/obo/CL_0000000")
    cell_types_terms_iris = ["http://purl.obolibrary.org/obo/$id"
                             for id in cell_types_terms_ids]
    cell_types_terms = onto_term.("cl", cell_types_terms_iris)

    new_tree = OntologyTree(base_term, cell_types_terms, Term[]; max_parent_limit=50,
                            include_UBERON=false)

    save(dummy_onto_tree_path, "onto_tree", new_tree)

    return new_tree
end

# Test the prepare_data function
@testset "prepare_data tests" begin end

base_eset = rand(ExpressionSet, 10000, 2)

# We need to add cell types to the expression set
phenotype_data(base_eset)[!, "cell_type"] = ["CL_0009051", "CL_0000988"]

onto_tree = create_dummy_ontologytree(["CL_0009051", "CL_0000988"])
syd_config = create_config(Dict("column" => Dict("cell_type" => "cell_type"),
                                "dataset" => Dict("samples" => 150)))

dataprep_strategy = DataPreparationStrategy()

train_loader, test_loader = prepare_data(base_eset, onto_tree; syd_config,
                                         dataprep_strategy)
