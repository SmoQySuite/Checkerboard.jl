#########################################################
## METHODS FOR CONSTRUCTING CHECKERBOARD DECOMPOSITION ##
#########################################################

@doc raw"""
    checkerboard_decomposition!(neighbor_table::Matrix{Int})

Given a `neighbor_table`, construct the checkerboard decomposition, which results in the columns of `neighbor_table` being re-ordered in-place.
Two additional arrays are also returned:
- `perm::Vector{Int}`: The permutation used to re-order the columns of `neighbor_table` according to the checkerboard decomposition.
- `colors::Matrix{Int}`: Marks the column indice boundaries for each checkerboard color in `neighbor_table`.
"""
function checkerboard_decomposition!(neighbor_table::Matrix{Int})

    # number of neighbor pairs
    N = size(neighbor_table,2)

    # get perm to sort neighbor_table
    sorted_perm, swapped = _sort_neighbor_table!(neighbor_table)

    # apply the new sorted_perm permutation to neighbor_table
    @views @. neighbor_table = neighbor_table[:,sorted_perm]

    # construct checkerboard colors
    color_assignments = _checkerboard_colors(neighbor_table)

    # get checkerboard perm by getting the permuation the sorts the color assignments
    checkerboard_perm = sortperm(color_assignments)

    # get final permutation for checkerboard decomposition
    perm = sorted_perm[checkerboard_perm]

    # re-order neighbor_table, colors and swapped according to the final perm
    @views @. neighbor_table    = neighbor_table[:,checkerboard_perm]
    @views @. color_assignments = color_assignments[checkerboard_perm]
    @views @. swapped           = swapped[perm]

    # undo the swaps that occured in neighbor_table in the _sort_neighbor_table! method
    for n in 1:size(neighbor_table,2)
        if swapped[n]
            tmp                 = neighbor_table[1,n]
            neighbor_table[1,n] = neighbor_table[2,n]
            neighbor_table[2,n] = tmp
        end
    end

    # number of checkerboard colors
    Ncolors = maximum(color_assignments)

    # get the column indice bounds that correspond to views of neighbor_table for each checkerboard color
    colors            = zeros(Int,2,Ncolors)
    colors[1,1]       = 1
    colors[2,Ncolors] = N
    color             = 1 # keeps track of current color
    for n in 1:N
        if color != color_assignments[n]
            colors[2,color]   = n-1
            colors[1,color+1] = n
            color += 1
        end
    end

    return perm, colors
end

#####################################
## PRIVATE FUNCTIONS, NOT EXPORTED ##
#####################################

# Returns the permuation `perm` that sorts `neighbor_table` such that the first row is in strictly ascending order,
# and for constant value in the first row the second row is also in ascending order.
# Also returns an array indicating which columns of `neighbor_table` had the values in the first and second rows swapped.
# Also modifies `neighbor_table` in place such that in each column the larger value is in the first row.
function _sort_neighbor_table!(neighbor_table::Matrix{Int})

    @assert size(neighbor_table,1) == 2

    # number of edges/bonds in lattice
    Nedges = size(neighbor_table,2)

    # number of sites in lattice
    Nsites = maximum(neighbor_table)

    # whether hopping amplitude associated with neighbor pair needs to be conjugated
    swapped = zeros(Bool,Nedges)

    # ensure that smaller index is in first row for each neighbor pair
    for n in 1:Nedges
        if neighbor_table[1,n] > neighbor_table[2,n]
            tmp                 = neighbor_table[1,n]
            neighbor_table[1,n] = neighbor_table[2,n]
            neighbor_table[2,n] = tmp
            swapped[n]          = true
        end
    end

    # assign a weight to each neighbor pair
    weights = Nsites * neighbor_table[1,:] + neighbor_table[2,:]

    # sort neighbor table according to weights
    perm = sortperm(weights)

    return perm, swapped
end


# Constructs the checkerboard decomposition for a given neighbor table.
# This is really just an (inefficient) graph edge coloring algorithm.
function _checkerboard_colors(neighbor_table::Matrix{Int})

    # declare color to store color/color assignments
    colors = zeros(Int, size(neighbor_table,2))
    # checking dimensions
    @assert size(neighbor_table,2) == length(colors)
    @assert size(neighbor_table,1) == 2
    # getting the total number of neighbor pairs
    Nneighbors = size(neighbor_table,2)
    # initialize arrays to store coloring in
    colors = zeros(Int,Nneighbors)
    # intially all neighbors are unassigned to a color
    fill!(colors,0)
    # keeps track of which color is being constructed
    color = 0
    # keep track of number of bonds assigned to color
    nassigned = 0
    # while any edge/bond is not assigned to a color
    while nassigned < Nneighbors
        # increment to next color
        color += 1
        # iterate over edges/bonds in graph/lattice to current color
        for edge in 1:Nneighbors
            # if edge is not assigned to a color
            if colors[edge]==0
                # assign it to current color
                colors[edge] = color
                nassigned   += 1
                # iterate over previous edges
                for prev_edge in 1:edge-1
                    # if previous edge is a color member
                    if colors[prev_edge]==color
                        # if the previous edge of current color intersects current edge
                        if ( neighbor_table[1,edge]==neighbor_table[1,prev_edge] ||
                             neighbor_table[2,edge]==neighbor_table[2,prev_edge] || 
                             neighbor_table[1,edge]==neighbor_table[2,prev_edge] || 
                             neighbor_table[2,edge]==neighbor_table[1,prev_edge] )
                            # remove current edge from color
                            colors[edge] = 0
                            nassigned   -= 1
                            break
                        end
                    end
                end
            end
        end
    end
    return colors
end