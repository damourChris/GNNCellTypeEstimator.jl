function get_cell_types(eset::ExpressionSet, pheno_col::String;
                        cell_seperator=";", value_seperator="=")
    proportions = phenotype_data(eset)[!, pheno_col]
    splits = split.(proportions, cell_seperator)
    cell_types_raw = [split.(x, value_seperator) for x in splits]
    cell_types = unique([strip.(getindex.(x, 1)) for x in cell_types_raw])
    return [String.(cell) for cell in cell_types if cell != ["NA"]][1]
end
function get_cell_type_proportions(eset::ExpressionSet, pheno_col::String;
                                   cell_seperator=";", value_seperator="=")
    proportions = phenotype_data(eset)[!, pheno_col]
    cell_types = get_cell_types(eset, pheno_col)

    splits = split.(proportions, cell_seperator)
    cell_types_raw = [split.(x, value_seperator) for x in splits]

    na_indices = findall(x -> x == [["NA"]], cell_types_raw)

    cell_types_raw = [cell_types_raw[i]
                      for i in eachindex(cell_types_raw) if i ∉ na_indices]

    cell_values = [getindex.(split, 2) for split in cell_types_raw]
    cell_values = [strip.(vals) for vals in cell_values]
    cell_values = [replace.(vals, "%" => "") for vals in cell_values]
    cell_values = [parse.(Float64, value) for value in cell_values]

    # Make a matrix
    cell_values = transpose(hcat(cell_values...))

    df = DataFrame(cell_values, cell_types)

    # Add sample ids  
    snames = sample_names(eset)
    snames = [snames[i]
              for i in eachindex(snames) if i ∉ na_indices]

    df[!, :sample_id] = snames
    return df
end

function add_proportions!(g::MetaGraphs.MetaDiGraph, cell_type::String, proportion::Float64)
    cell_type = replace(cell_type, "_" => ":")
    cell_type_idx = g[cell_type, :id]
    return g.vprops[cell_type_idx][:proportion] = proportion
end

function add_expression!(g::MetaGraphs.MetaDiGraph, gene::String, expression::Float64)
    gene_idx = g[gene, :id]
    return g.vprops[gene_idx][:expression] = Float32(expression)
end

function add_expression!(g::MetaGraphs.MetaDiGraph, gene::Vector{String},
                         expression::Vector{Float64})
    for (gene, expression) in zip(gene, expression)
        add_expression!(g, gene, expression)
    end
end

function propagate_cell_proportions!(graph::MetaGraphs.MetaDiGraph)
    proportions_to_propagate = [v_index
                                for (v_index, v_props) in graph.vprops
                                if haskey(v_props, :proportion)]
    # @info "Proportion propagation started. Propagating $(length(proportions_to_propagate)) proportions"
    for vertex in proportions_to_propagate
        # @info "Propagating proportion for term: $(get_prop(graph, vertex, :term).label)"

        parents_indices = get_graph_parents(graph, vertex)
        if all(isnothing.(parents_indices))
            continue
        end

        proportion = get_prop(graph, vertex, :proportion)
        # @info "Propagating proportion: $proportion for term: $(get_prop(graph, vertex, :term).label) with $(length(parents_indices)) parents"
        for parent_index in parents_indices
            if !haskey(graph.vprops[parent_index], :proportion)
                # @info "Parent does not have a proportion. Setting it to: $proportion"
                set_prop!(graph, parent_index, :proportion, proportion)
            else
                parent_prop = get_prop(graph, parent_index, :proportion)

                set_prop!(graph, parent_index, :proportion,
                          parent_prop + proportion)
                # @info "Parent already has a proportion: $parent_prop. Adding: $proportion"
            end

            propagate_proportion_to_parent!(graph, parent_index, proportion)
        end

        # @info "Propagating proportion: $proportion to parent: $(get_prop(graph, parent_index, :term).label)"
        # propagate_proportion_to_parent!(graph, parent_index, proportion)
    end

    # @info "Proportion propagation done."
    # @info "Setting the proportion of all the remaining cell_type cells to 0"

    # Then we can set the proportion of all the remaining cell_type cells to 0
    for vertex in vertices(graph)
        if haskey(graph.vprops[vertex], :proportion)
            continue
        end

        if graph.vprops[vertex][:type] == :term
            # @info "Setting proportion of cell: $(get_prop(graph, vertex, :term).label) to 0"
            set_prop!(graph, vertex, :proportion, 0.0)
        end
    end
end

function get_graph_parents(graph::MetaGraphs.MetaDiGraph, vertex::Int)
    parents = Int[]
    for edge in edges(graph)
        if src(edge) == vertex
            push!(parents, dst(edge))
        end
    end

    return parents
end

function propagate_proportion_to_parent!(graph::MetaGraphs.MetaDiGraph, vertex::Int,
                                         proportion::Float64)
    if graph.vprops[vertex][:type] !== :term
        return nothing
    end
    vertex_term = get_prop(graph, vertex, :term)
    # @info "Propagating proportion: $proportion for term: $(vertex_term.label)"

    parents_indices = get_graph_parents(graph, vertex)
    if all(isnothing.(parents_indices))
        return
    end

    # @info "To parent: $(parent_term.label)"

    for parent_index in parents_indices
        if !haskey(graph.vprops[parent_index], :proportion)
            # @info "Parent does not have a proportion. Setting it to: $proportion"
            set_prop!(graph, parent_index, :proportion, proportion)
        else
            parent_prop = get_prop(graph, parent_index, :proportion)

            set_prop!(graph, parent_index, :proportion,
                      parent_prop + proportion)
            # @info "Parent already has a proportion: $parent_prop. Adding: $proportion"
        end

        propagate_proportion_to_parent!(graph, parent_index, proportion)
    end

    return nothing
end
