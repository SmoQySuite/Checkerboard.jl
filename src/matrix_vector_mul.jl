####################################
## MATRIX-VECTOR MULTIPLY METHODS ##
####################################

@doc raw"""
    checkerboard_lmul!(v::AbstractVector{T}, neighbor_table::Matrix{Int},
        coshΔτt::AbstractVector{E}, sinhΔτt::AbstractVector{E}, colors::Matrix{Int};
        transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:Continuous}

Multiply in-place the vector `v` by the checkerboard matrix.
"""
function checkerboard_lmul!(v::AbstractVector{T}, neighbor_table::Matrix{Int},
    coshΔτt::AbstractVector{E}, sinhΔτt::AbstractVector{E},
    colors::Matrix{Int}; transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued vector by a complex checkerboard matrix!"

    # number of checkerboard colors
    Ncolors = size(colors, 2)

    # how to iterate over neighbors in neighbor_table accounting for whether
    # or not the checkerboard matrix has been transposed
    transposed = inverted*(1-transposed) + (1-inverted)*transposed
    start      = (1-transposed) + transposed*Ncolors
    step       = 1 - 2*transposed
    stop       = (1-transposed)*Ncolors + transposed

    # iterate over columns of B matrix
    for color in start:step:stop

        # perform multiply by checkerboard color
        checkerboard_color_lmul!(v, color, neighbor_table, coshΔτt, sinhΔτt, colors, inverted=inverted)
    end

    return nothing
end


function checkerboard_lmul!(v::AbstractVector{T}, neighbor_table::Matrix{Int},
    coshΔτt::AbstractArray{E}, sinhΔτt::AbstractArray{E},
    colors::Matrix{Int}, L::Int; transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued vector by a complex checkerboard matrix!"
    
    # number of checkerboard colors
    Ncolors = size(colors, 2)

    # how to iterate over neighbors in neighbor_table accounting for whether
    # or not the checkerboard matrix has been transposed
    transposed = inverted*(1-transposed) + (1-inverted)*transposed
    start      = (1-transposed) + transposed*Ncolors
    step       = 1 - 2*transposed
    stop       = (1-transposed)*Ncolors + transposed

    # iterate over columns of B matrix
    for color in start:step:stop

        # perform multiply by checkerboard color
        checkerboard_color_lmul!(v, color, neighbor_table, coshΔτt, sinhΔτt, colors, L, inverted=inverted)
    end

    return nothing
end


@doc raw"""
    checkerboard_color_lmul!(v::AbstractVector{T}, color::Int, neighbor_table::Matrix{Int},
        coshΔτt::AbstractVector{E}, sinhΔτt::AbstractVector{E}, colors::Matrix{Int};
        transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:Continuous}

Multiply in-place the vector `v` by the `color` checkerboard color matrix.
"""
function checkerboard_color_lmul!(v::AbstractVector{T}, color::Int, neighbor_table::Matrix{Int},
    coshΔτt::AbstractVector{E}, sinhΔτt::AbstractVector{E},
    colors::Matrix{Int}; inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued vector by a complex checkerboard matrix!"

    # equals -1 for matrix inverse, +1 otherwise
    inverse = (1-2*inverted)

    # get the range of the checkerboard color
    start = colors[1, color]
    stop  = colors[2, color]

    # iterate over neighbor pairs
    @fastmath @inbounds  for n in start:stop
        # get pair of neighbor sites
        i = neighbor_table[1,n]
        j = neighbor_table[2,n]
        # get the relevant cosh and sinh values
        cᵢⱼ = coshΔτt[n]
        sᵢⱼ = inverse * sinhΔτt[n]
        # get the initial matrix elements
        vᵢ = v[i]
        vⱼ = v[j]
        # in-place multiply
        v[i] = cᵢⱼ * vᵢ + sᵢⱼ * vⱼ
        v[j] = cᵢⱼ * vⱼ + conj(sᵢⱼ) * vᵢ
    end

    return nothing
end


function checkerboard_color_lmul!(v::AbstractVector{T}, color::Int, neighbor_table::Matrix{Int},
    coshΔτt::AbstractVector{E}, sinhΔτt::AbstractVector{E},
    colors::Matrix{Int}, L::Int; inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued vector by a complex checkerboard matrix!"

    # equals -1 for matrix inverse, +1 otherwise
    inverse = (1-2*inverted)

    # get the range of the checkerboard color
    start = colors[1, color]
    stop  = colors[2, color]

    # iterate over neighbor pairs
    @fastmath @inbounds for n in start:stop
        # get pair of neighbor sites
        i = neighbor_table[1,n]
        j = neighbor_table[2,n]
        # get the relevant cosh and sinh values
        cᵢⱼ = coshΔτt[n]
        sᵢⱼ = inverse * sinhΔτt[n]
        # iterate over imaginary time slices
        @simd for τ in 1:L
            # get the indices
            k = (i-1)*L + τ
            l = (j-1)*L + τ
            # get the initial matrix elements
            vᵢ = v[k]
            vⱼ = v[l]
            # in-place multiply
            v[k] = cᵢⱼ * vᵢ + sᵢⱼ       * vⱼ
            v[l] = cᵢⱼ * vⱼ + conj(sᵢⱼ) * vᵢ
        end
    end

    return nothing
end


function checkerboard_color_lmul!(v::AbstractVector{T}, color::Int, neighbor_table::Matrix{Int},
    coshΔτt::AbstractMatrix{E}, sinhΔτt::AbstractMatrix{E},
    colors::Matrix{Int}, L::Int; inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued vector by a complex checkerboard matrix!"

    # equals -1 for matrix inverse, +1 otherwise
    inverse = (1-2*inverted)

    # get the range of the checkerboard color
    start = colors[1, color]
    stop  = colors[2, color]

    # iterate over neighbor pairs
    @fastmath @inbounds for n in start:stop
        # get pair of neighbor sites
        i = neighbor_table[1,n]
        j = neighbor_table[2,n]
        # iterate over imaginary time slices
        @simd for τ in 1:L
            # get the relevant cosh and sinh values
            cᵢⱼ = coshΔτt[τ,n]
            sᵢⱼ = inverse * sinhΔτt[τ,n]
            # get the indices
            k = (i-1)*L + τ
            l = (j-1)*L + τ
            # get the initial matrix elements
            vᵢ = v[k]
            vⱼ = v[l]
            # in-place multiply
            v[k] = cᵢⱼ * vᵢ + sᵢⱼ       * vⱼ
            v[l] = cᵢⱼ * vⱼ + conj(sᵢⱼ) * vᵢ
        end
    end

    return nothing
end