####################################
## MATRIX-MATRIX MULTIPLY METHODS ##
####################################

@doc raw"""
    checkerboard_lmul!(
        # ARGUMENTS
        B::AbstractMatrix{T},
        neighbor_table::Matrix{Int},
        coshΔτt::AbstractVector{E},
        sinhΔτt::AbstractVector{E},
        colors::Matrix{Int};
        # KEYWORD ARGUMENTS
        transposed::Bool=false,
        inverted::Bool=false
    ) where {T<:Continuous, E<:Continuous}

Evaluate the matrix-matrix product in-place `B=Γ⋅B` where `Γ` is the checkerboard matrix.
"""
function checkerboard_lmul!(
    # ARGUMENTS
    B::AbstractMatrix{T},
    neighbor_table::Matrix{Int},
    coshΔτt::AbstractVector{E},
    sinhΔτt::AbstractVector{E},
    colors::Matrix{Int};
    # KEYWORD ARGUMENTS
    transposed::Bool=false,
    inverted::Bool=false
) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued matrix by a complex checkerboard matrix!"

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
        checkerboard_color_lmul!(B, color, neighbor_table, coshΔτt, sinhΔτt, colors, inverted=inverted)
    end

    return nothing
end


@doc raw"""
    checkerboard_color_lmul!(
        # ARGUMENTS
        B::AbstractMatrix{T},
        color::Int,
        neighbor_table::Matrix{Int},
        coshΔτt::AbstractVector{E},
        sinhΔτt::AbstractVector{E},
        colors::Matrix{Int};
        # KEYWORD ARGUMENTS
        inverted::Bool=false
    ) where {T<:Continuous, E<:Continuous}

Evaluate the matrix-matrix product in-place `B=Γ[c]⋅B` where `Γ[c]` is the `color` checkerboard color matrix.
"""
function checkerboard_color_lmul!(
    # ARGUMENTS
    B::AbstractMatrix{T},
    color::Int,
    neighbor_table::Matrix{Int},
    coshΔτt::AbstractVector{E},
    sinhΔτt::AbstractVector{E},
    colors::Matrix{Int};
    # KEYWORD ARGUMENTS
    inverted::Bool=false
) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued matrix by a complex checkerboard matrix!"

    # equals -1 for matrix inverse, +1 otherwise
    inverse = 1 - 2*inverted

    # get the range of the checkerboard color
    start = colors[1, color]
    stop  = colors[2, color]

    # construct views for current checkerboard color
    nt = @view neighbor_table[:,start:stop]
    ch = @view coshΔτt[start:stop]
    sh = @view sinhΔτt[start:stop]

    # iterate over columns of B
    @inbounds @fastmath for c in 1:size(B,2)
        # iterate over neighbor pairs
        @simd for n in 1:size(nt, 2)
            # get the pair of neighboring sites
            i = nt[1,n]
            j = nt[2,n]
            # get relevant cosh and sinh values
            cᵢⱼ = ch[n]
            sᵢⱼ = inverse * sh[n]
            # get relevant matrix elements
            bᵢ = B[i,c]
            bⱼ = B[j,c]
            # perform multiply
            B[i,c] = cᵢⱼ * bᵢ + sᵢⱼ * bⱼ
            B[j,c] = cᵢⱼ * bⱼ + conj(sᵢⱼ) * bᵢ
        end
    end

    return nothing
end


@doc raw"""
    checkerboard_rmul!(
        # ARGUMENTS
        B::AbstractMatrix{T},
        neighbor_table::Matrix{Int},
        coshΔτt::AbstractVector{E},
        sinhΔτt::AbstractVector{E},
        colors::Matrix{Int};
        # KEYWORD ARGUMENTS
        transposed::Bool=false,
        inverted::Bool=false
    ) where {T<:Continuous, E<:Continuous}

Evaluate the matrix-matrix product in-place `B=B⋅Γ` where `Γ` is the checkerboard matrix.
"""
function checkerboard_rmul!(
    # ARGUMENTS
    B::AbstractMatrix{T},
    neighbor_table::Matrix{Int},
    coshΔτt::AbstractVector{E},
    sinhΔτt::AbstractVector{E},
    colors::Matrix{Int};
    # KEYWORD ARGUMENTS
    transposed::Bool=false,
    inverted::Bool=false
) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued matrix by complex a checkerboard matrix!"

    # number of checkerboard colors
    Ncolors = size(colors, 2)

    # how to iterate over neighbors in neighbor_table accounting for whether
    # or not the checkerboard matrix has been transposed
    transposed = inverted*(1-transposed) + (1-inverted)*transposed
    start      = (1-transposed) + transposed*Ncolors
    step       = 1 - 2*transposed
    stop       = (1-transposed)*Ncolors + transposed

    # iterate over columns of B matrix
    for color in stop:-step:start

        # perform multiply by checkerboard color
        checkerboard_color_rmul!(B, color, neighbor_table, coshΔτt, sinhΔτt, colors, inverted=inverted)
    end

    return nothing
end


@doc raw"""
    checkerboard_color_rmul!(
        # ARGUMENTS
        B::AbstractMatrix{T},
        color::Int,
        neighbor_table::Matrix{Int},
        coshΔτt::AbstractVector{E},
        sinhΔτt::AbstractVector{E},
        colors::Matrix{Int};
        # KEYWORD ARGUMENTS
        inverted::Bool=false
    ) where {T<:Continuous, E<:Continuous}

Evaluate the matrix-matrix product in-place `B=B⋅Γ[c]` where `Γ[c]` is the `color` checkerboard color matrix.
"""
function checkerboard_color_rmul!(
    # ARGUMENTS
    B::AbstractMatrix{T},
    color::Int,
    neighbor_table::Matrix{Int},
    coshΔτt::AbstractVector{E},
    sinhΔτt::AbstractVector{E},
    colors::Matrix{Int};
    # KEYWORD ARGUMENTS
    inverted::Bool=false
) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued matrix by a complex checkerboard matrix!"

    # equals -1 for matrix inverse, +1 otherwise
    inverse = 1 - 2*inverted

    # get the range of the checkerboard color
    start = colors[1, color]
    stop  = colors[2, color]

    # construct views for current checkerboard color
    nt = @view neighbor_table[:,start:stop]
    ch = @view coshΔτt[start:stop]
    sh = @view sinhΔτt[start:stop]

    # iterate over neighbor pairs
    @inbounds @fastmath for n in 1:size(nt, 2)
        # get the pair of neighboring sites
        j = nt[1,n]
        i = nt[2,n]
        # get relevant cosh and sinh values
        cᵢⱼ = ch[n]
        sᵢⱼ = inverse * sh[n]
        # iterate over rows of B matrix
        @simd for r in 1:size(B,1)
            # get relevant matrix elements
            bᵢ = B[r,i]
            bⱼ = B[r,j]
            # perform multiply
            B[r,i] = cᵢⱼ * bᵢ + sᵢⱼ * bⱼ
            B[r,j] = cᵢⱼ * bⱼ + conj(sᵢⱼ) * bᵢ
        end
    end

    return nothing
end