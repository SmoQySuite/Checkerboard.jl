"""
    AbstractVecOrMat{T} = Union{AbstractVector{T}, AbstractMatrix{T}}

Abstract type defining union of `AbstractVector` and `AbstractMatrix`.
"""
AbstractVecOrMat{T} = Union{AbstractVector{T}, AbstractMatrix{T}}

"""
    CheckerboardMatrix{T<:Union{AbstractFloat, Complex{<:AbstractFloat}}}

A type to represent a checkerboard decomposition matrix.

# Fields
$(TYPEDFIELDS)
"""
struct CheckerboardMatrix{T<:Continuous}
    
    "Is checkerboard matrix transposed."
    transposed::Bool

    "Is checkerboard matrix inverted."
    inverted::Bool

    "Number of sites in lattice/dimension of checkerboard matrix."
    Nsites::Int

    "Number of neighbors."
    Nneighbors::Int

    "Number of checkerboard colors/groups."
    Ncolors::Int

    "Neighbor table represented by a `(2,Nneighbors)` dimensional matrix, 
    where each column contains a pair of neighboring sites."
    neighbor_table::Matrix{Int}

    "The ``\\cosh(\\Delta\\tau t)`` values."
    coshΔτt::Vector{T}

    "The ``\\sinh(\\Delta\\tau t)`` values."
    sinhΔτt::Vector{T}

    "The checkerboard permutation order relative to the ordering of the original neighbor table."
    perm::Vector{Int}

    "The bounds of each checkerboard color/group in `neighbor_table`."
    colors::Matrix{Int}
end

"""
    CheckerboardMatrix(neighbor_table::Matrix{Int}, t::AbstractVector{T}, Δτ::E;
        transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:AbstractFloat}

Given a `neighbor_table` along with the corresponding hopping amplitudes `t` and discretzation
in imaginary time `Δτ`, construct an instance of the type `CheckerboardMatrix`. 
"""
function CheckerboardMatrix(neighbor_table::Matrix{Int}, t::AbstractVector{T}, Δτ::E;
    transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:AbstractFloat}

    nt           = deepcopy(neighbor_table)
    perm, colors = checkerboard_decomposition!(nt)
    Nsites       = maximum(nt)
    Nneighbors   = size(nt,2)
    Ncolors      = size(colors,2)
    coshΔτt      = zeros(T,Nneighbors)
    sinhΔτt      = zeros(T,Nneighbors)
    update!(coshΔτt, sinhΔτt, t, perm, Δτ)

    return CheckerboardMatrix{T}(transposed, inverted, Nsites, Nneighbors, Ncolors, nt, coshΔτt, sinhΔτt, perm, colors)
end

"""
    CheckerboardMatrix(Γ::CheckerboardMatrix; transposed::Bool=false, inverted::Bool=false, new_matrix::Bool=false)

Construct a new instance of `CheckerboardMatrix` based on a current instance `Γ` of `CheckerboardMatrix`.
If `new_matrix=true` then allocate new `coshΔτt` and `sinhΔτt` arrays.
"""
function CheckerboardMatrix(Γ::CheckerboardMatrix{T}; transposed::Bool=false, inverted::Bool=false, new_matrix::Bool=false) where {T}

    (; Nsites, Nneighbors, Ncolors, neighbor_table, perm, colors) = Γ

    if new_matrix
        coshΔτt = similar(Γ.coshΔτt)
        sinhΔτt = similar(Γ.sinhΔτt)
        copyto!(coshΔτt, Γ.coshΔτt)
        copyto!(sinhΔτt, Γ.sinhΔτt)
    else
        coshΔτt = Γ.coshΔτt
        sinhΔτt = Γ.sinhΔτt
    end

    return CheckerboardMatrix{T}(transposed, inverted, Nsites, Nneighbors, Ncolors, neighbor_table, coshΔτt, sinhΔτt, perm, colors)
end

"""
    checkerboard_matrices(neighbor_table::Matrix{Int}, t::AbstractMatrix{T}, Δτ::E;
        transposed::Bool=false, inverted::Bool=false)

Return a vector of `CheckerboardMatrix`, one for each column of `t`, all sharing the same `neighbor_table`.
"""
function checkerboard_matrices(neighbor_table::Matrix{Int}, t::AbstractMatrix{T}, Δτ::E;
    transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:AbstractFloat}

    @assert size(t,1) == size(neighbor_table,2)

    # number of checkerboard matrices to construct
    L = size(t,2)

    # declare empty vector of checkerboard matrices
    Γs = CheckerboardMatrix{T}[]

    # declare intial checkerboard matrix
    t₁ = @view t[:,1]
    Γ₁ = CheckerboardMatrix(neighbor_table, t₁, Δτ, transposed=transposed, inverted=inverted)

    # add first checkerboard matrix to vector
    push!(Γs, Γ₁)

    # add the rest of the checkerboard matrices to vector
    for l in 2:L
        
        # construct next checkerboard matrix
        Γₗ = CheckerboardMatrix(Γ₁, new_matrix=true)
        tₗ = @view t[:,l]
        update!(Γₗ, tₗ, Δτ)

        # add new checkerboard matrix to vector
        push!(Γs, Γₗ)
    end

    return Γs
end

"""
    update!(Γ::CheckerboardMatrix{T}, t::AbstractVector{T}, Δτ::E) where {T<:Continuous, E<:AbstractFloat}

Update the `CheckerboardMatrix` based on new hopping parameters `t` and discretezation in imaginary time `Δτ`. 
"""
function update!(Γ::CheckerboardMatrix{T}, t::AbstractVector{T}, Δτ::E) where {T<:Continuous, E<:AbstractFloat}

    (; coshΔτt, sinhΔτt, perm) = Γ
    update!(coshΔτt, sinhΔτt, t, perm, Δτ)

    return nothing
end

"""
    update!(Γs::AbstractVector{CheckerboardMatrix{T}}, t::AbstractVector{T}, Δτ::E) where {T<:Continuous, E<:AbstractFloat}

Update a vector of `CheckerboardMatrix` based on new hopping parameters `t` and discretezation in imaginary time `Δτ`. 
"""
function update!(Γs::AbstractVector{CheckerboardMatrix{T}}, t::AbstractMatrix{T}, Δτ::E) where {T<:Continuous, E<:AbstractFloat}

    @assert length(Γs) == size(t,2)
    @assert Γs[1].Nneighbors == size(t,1)

    for l in eachindex(Γs)
        Γₗ = Γs[l]
        tₗ = @view t[:,l]
        update!(Γₗ, tₗ, Δτ)
    end

    return nothing
end

update!(Γ; t, Δτ) = update!(Γ, t, Δτ)

"""
    update!(coshΔτt::AbstractVector{T}, sinhΔτt::AbstractVector{T}, t::AbstractVector{T}, Δτ::E) where {T<:Continuous, E<:AbstractFloat}

Update the `coshΔτt` and `sinhΔτt` associated with a checkerboard decomposition based on new hopping parameters
`t` and discretezation in imaginary time `Δτ`. 
"""
function update!(coshΔτt::AbstractVector{T}, sinhΔτt::AbstractVector{T}, t::AbstractVector{T},
    perm::AbstractVector{Int}, Δτ::E) where {T<:Continuous, E<:AbstractFloat}

    @. coshΔτt = cosh(Δτ*abs(t[perm]))
    @. sinhΔτt = sign(t[perm])*sinh(Δτ*abs(t[perm]))

    return nothing
end

update!(; coshΔτt, sinhΔτt, t, perm, Δτ) = update!(coshΔτt, sinhΔτt, t, perm, Δτ)


########################
## OVERLOADNG METHODS ##
########################

"""
    size(Γ::CheckerboardMatrix)
    
    size(Γ::CheckerboardMatrix, dim::Int)

Return the dimensions of the checkerboard decomposition matrix `Γ`.
"""
size(Γ::CheckerboardMatrix) = (Γ.Nsites, Γ.Nsites)
size(Γ::CheckerboardMatrix, dim::Int) = Γ.Nsites


"""
    transpose(Γ::CheckerboardMatrix)

Return a transposed/adjoint version of the checkerboard matrix `Γ`.
"""
transpose(Γ::CheckerboardMatrix) = CheckerboardMatrix(Γ, transposed=!Γ.transposed)


"""
    adjoint(Γ::CheckerboardMatrix)

Return a transposed/adjoint version of the checkerboard matrix `Γ`.
"""
adjoint(Γ::CheckerboardMatrix) = CheckerboardMatrix(Γ, transposed=!Γ.transposed)


"""
    inv(Γ::CheckerboardMatrix)

Return the inverse of the checkerboard matrix `Γ`.
"""
inv(Γ::CheckerboardMatrix) = CheckerboardMatrix(Γ, inverted=!Γ.inverted)


"""
    mul!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, v::AbstractVecOrMat)

Evaluate the matrix-vector or matrix-matrix product `u=Γ⋅v`.
"""
function mul!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, v::AbstractVecOrMat)

    copyto!(u, v)
    lmul!(Γ, u)

    return nothing
end

"""
    mul!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, v::AbstractVecOrMat, color::Int)

Evaluate the matrix-vector or matrix-matrix product `u=Γ[c]⋅v` where `Γ[c]` is the matrix
associated with the `color` checkerboard color matrix.
"""
function mul!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, v::AbstractVecOrMat, color::Int)

    copyto!(u, v)
    lmul!(Γ, u, color)

    return nothing
end

"""
    mul!(u::AbstractVecOrMat, v::AbstractVecOrMat, Γ::CheckerboardMatrix)

Evaluate the matrix-vector or matrix-matrix product `u=v⋅Γ`.
"""
function mul!(u::AbstractVecOrMat, v::AbstractVecOrMat, Γ::CheckerboardMatrix)

    copyto!(u, v)
    rmul!(u, Γ)

    return nothing
end

"""
    mul!(u::AbstractVecOrMat, v::AbstractVecOrMat, Γ::CheckerboardMatrix, color::Int)

Evaluate the matrix-vector or matrix-matrix product `u=v⋅Γ[c]` where `Γ[c]` is the matrix
associated with the `color` checkerboard color matrix.
"""
function mul!(u::AbstractVecOrMat, v::AbstractVecOrMat, Γ::CheckerboardMatrix, color::Int)

    copyto!(u, v)
    rmul!(u, Γ, color)

    return nothing
end


"""
    lmul!(Γ::CheckerboardMatrix, u::AbstractVecOrMat)

Evaluate in-place the matrix-vector or matrix-matrix product `u=Γ⋅u`, where `u` gets over-written.
"""
function lmul!(Γ::CheckerboardMatrix, u::AbstractVecOrMat)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    checkerboard_lmul!(u, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=inverted)

    return nothing
end

"""
    lmul!(Γ::CheckerboardMatrix, u::AbstractVecOrMat, color::Int)

Evaluate in-place the matrix-vector or matrix-matrix product `u=Γ[c]⋅u`, where `Γ[c]` is the matrix
associated with the `color` checkerboard color matrix.
"""
function lmul!(Γ::CheckerboardMatrix, u::AbstractVecOrMat, color::Int)

    (; inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    checkerboard_color_lmul!(u, color, neighbor_table, coshΔτt, sinhΔτt, colors, inverted=inverted)

    return nothing
end


"""
    rmul!(u::AbstractVecOrMat, Γ::CheckerboardMatrix)

Evaluate in-place the matrix-vector or matrix-matrix product `u=u⋅Γ`, where `u` gets over-written.
"""
function rmul!(u::AbstractVecOrMat, Γ::CheckerboardMatrix)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    checkerboard_rmul!(u, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=inverted)

    return nothing
end

"""
    rmul!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, color::Int)

Evaluate in-place the matrix-vector or matrix-matrix product `u=u⋅Γ[c]`, where `Γ[c]` is the
matrix associated with the `color` checkerboard color matrix.
"""
function rmul!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, color::Int)

    (; inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    checkerboard_color_rmul!(u, color, neighbor_table, coshΔτt, sinhΔτt, colors, inverted=inverted)

    return nothing
end


"""
    ldiv!(Γ::CheckerboardMatrix, v::AbstractVecOrMat)

Evaluate in-place the matrix-vector or matrix-matrix product `v=Γ⁻¹⋅v`.
"""
function ldiv!(Γ::CheckerboardMatrix, v::AbstractVecOrMat)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    checkerboard_lmul!(v, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=!inverted)

    return nothing
end

"""
    ldiv!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, v::AbstractVecOrMat)

Evaluate the matrix-vector or matrix-matrix product `u=Γ⁻¹⋅v`.
"""
function ldiv!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, v::AbstractVecOrMat)

    copyto!(u,v)
    ldiv!(Γ, u)

    return nothing
end

"""
    ldiv!(Γ::CheckerboardMatrix, v::AbstractVecOrMat, color::Int)

Evaluate in-place the matrix-vector or matrix-matrix product `v=Γ⁻¹[c]⋅v` where `Γ⁻¹[c]` is the inverse
of the matrix associated with the `color` checkerboard color matrix.
"""
function ldiv!(Γ::CheckerboardMatrix, v::AbstractVecOrMat, color::Int)

    (; inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    checkerboard_color_lmul!(v, color, neighbor_table, coshΔτt, sinhΔτt, colors, inverted=!inverted)

    return nothing
end

"""
    ldiv!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, v::AbstractVecOrMat, color::Int)

Evaluate the matrix-vector or matrix-matrix product `u=Γ⁻¹[c]⋅v` where `Γ⁻¹[c]` is the inverse of the
matrix associated with the `color` checkerboard color matrix.
"""
function ldiv!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, v::AbstractVecOrMat, color::Int)

    copyto!(u,v)
    ldiv!(Γ, u, color)

    return nothing
end


"""
    rdiv!(u::AbstractVecOrMat, Γ::CheckerboardMatrix)

Evaluate in-place the matrix-vector or matrix-matrix product `u=u⋅Γ⁻¹`, where `u` gets over-written.
"""
function rdiv!(u::AbstractVecOrMat, Γ::CheckerboardMatrix)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    checkerboard_rmul!(u, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=!inverted)

    return nothing
end

"""
    rdiv!(u::AbstractVecOrMat, v::AbstractVecOrMat, Γ::CheckerboardMatrix)

Evaluate the matrix-vector or matrix-matrix product `u=v⋅Γ⁻¹`, where `u` gets over-written.
"""
function rdiv!(u::AbstractVecOrMat, v::AbstractVecOrMat, Γ::CheckerboardMatrix)

    copyto!(u, v)
    rdiv!(u, Γ)

    return nothing
end

"""
    rdiv!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, color::Int)

Evaluate in-place the matrix-vector or matrix-matrix product `u=u⋅Γ⁻¹[c]`, where `Γ[c]` is the
matrix associated with the `color` checkerboard color matrix.
"""
function rdiv!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, color::Int)

    (; inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    checkerboard_color_rmul!(u, color, neighbor_table, coshΔτt, sinhΔτt, colors, inverted=!inverted)

    return nothing
end

"""
    rdiv!(u::AbstractVecOrMat, v::AbstractVecOrMat, Γ::CheckerboardMatrix, color::Int)

Evaluate the matrix-vector or matrix-matrix product `u=v⋅Γ⁻¹[c]`, where `Γ[c]` is the
matrix associated with the `color` checkerboard color matrix.
"""
function rdiv!(u::AbstractVecOrMat, v::AbstractVecOrMat, Γ::CheckerboardMatrix, color::Int)

    copyto!(u,v)
    rdiv!(u, Γ, color)

    return nothing
end