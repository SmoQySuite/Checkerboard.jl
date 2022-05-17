"""
    CheckerboardMatrix{T<:Union{AbstractFloat, Complex{<:AbstractFloat}}}

A type to represent a checkerboard decomposition matrix.

# Fields
$(TYPEDFIELDS)
"""
mutable struct CheckerboardMatrix{T<:Continuous}
    
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

    "Neighbor table."
    neighbor_table::Matrix{Int}

    "The ``\\cosh(\\Delta\\tau t)`` values."
    coshΔτt::Vector{T}

    "The ``\\sinh(\\Delta\\tau t)`` values."
    sinhΔτt::Vector{T}

    "The checkerboard permutation order."
    perm::Vector{Int}

    "The bounds of each checkerboard color/group in `neighbor_table`."
    colors::Matrix{Int}
end

"""
    CheckerboardMatrix(neighbor_table::Matrix{Int}, t::AbstractVector{T}, Δτ::E;
        transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:AbstractFloat}

Given a `neighbor_table` along with the corresponding hopping amplitudes `t` and discretezation
in imaginary time `Δτ`, construct an instance of `CheckerboardMatrix`. 
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
    update!(coshΔτt, sinhΔτt, t, Δτ)

    return CheckerboardMatrix{T}(transposed, inverted, Nsites, Nneighbors, Ncolors, nt, coshΔτt, sinhΔτt, perm, colors)
end

"""
    CheckerboardMatrix(Γ::CheckerboardMatrix; transposed::Bool=false, inverted::Bool=false)

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
    update!(Γ::CheckerboardMatrix{T}, t::AbstractVector{T}, Δτ::E) where {T<:Continuous, E<:AbstractFloat}

Update the `CheckerboardMatrix` based on new hopping parameters `t` and discretezation in imaginary time `Δτ`. 
"""
function update!(Γ::CheckerboardMatrix{T}, t::AbstractVector{T}, Δτ::E) where {T<:Continuous, E<:AbstractFloat}

    (; coshΔτt, sinhΔτt, perm) = Γ
    update!(coshΔτt, sinhΔτt, t, perm, Δτ)

    return nothing
end

update!(Γ; t, Δτ) = update!(Γ, t, Δτ)

"""
    update!(Γ::CheckerboardMatrix{T}, t::AbstractVector{T}, Δτ::E) where {T<:Continuous, E<:AbstractFloat}

Update the `coshΔτt` and `sinhΔτt` associated with a checkerboard decomposition based on new hopping parameters
`t` and discretezation in imaginary time `Δτ`. 
"""
function update!(coshΔτt::AbstractVector{T}, sinhΔτt::AbstractVector{T}, t::AbstractVector{T},
    perm::AbstractVector{Int}, Δτ::E) where {T<:Continuous, E<:AbstractFloat}

    @. coshΔτt = cosh(Δτ*abs(t[perm]))
    @. sinhΔτt = sign(t[perm])*sinh(Δτ*abs(t[perm]))

    return nothing
end

update!(; coshΔτt, sinhΔτt, t, Δτ) = update!(coshΔτt, sinhΔτt, t, Δτ)


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

Return a transposed version of the checkerboard matrix `Γ`.
"""
transpose(Γ::CheckerboardMatrix) = CheckerboardMatrix(Γ, tranposed=!Γ.transposed)


"""
    inv(Γ::CheckerboardMatrix)

Return the inverse of the checkerboard matrix `Γ`.
"""
inv(Γ::CheckerboardMatrix) = CheckerboardMatrix(Γ, inverted=!Γ.inverted)


"""
    mul!(u::AbstractVector, Γ::CheckerboardMatrix, v::AbstractVector)

Evaluate the matrix-vector product `u=Γ⋅v`.
"""
function mul!(u::AbstractVector, Γ::CheckerboardMatrix, v::AbstractVector)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    copyto!(u,v)
    checkerboard_lmul!(u, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=inverted)

    return nothing
end

"""
    mul!(u::AbstractVector, Γ::CheckerboardMatrix, v::AbstractVector, color::Int)

Evaluate the matrix-vector product `u=Γ[c]⋅v` where `Γ[c]` is the matrix associated with
the `color` checkerboard color matrix.
"""
function mul!(u::AbstractVector, Γ::CheckerboardMatrix, v::AbstractVector, color::Int)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    copyto!(u,v)
    checkerboard_color_lmul!(u, color, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=inverted)

    return nothing
end

"""
    mul!(A::AbstractMatrix, Γ::CheckerboardMatrix, B::AbstractMatrix)

Evaluate the matrix-matrix product `A=Γ⋅B`.
"""
function mul!(A::AbstractMatrix, Γ::CheckerboardMatrix, B::AbstractMatrix)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    copyto!(A,B)
    checkerboard_lmul!(A, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=inverted)

    return nothing
end

"""
    mul!(A::AbstractMatrix, Γ::CheckerboardMatrix, B::AbstractMatrix, color::Int)

Evaluate the matrix-matrix product `A=Γ[c]⋅B` where `Γ[c]` is the matrix associated with
the `color` checkerboard color matrix.
"""
function mul!(A::AbstractMatrix, Γ::CheckerboardMatrix, B::AbstractMatrix, color::Int)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    copyto!(A,B)
    checkerboard_color_lmul!(A, color, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=inverted)

    return nothing
end

"""
    mul!(A::AbstractMatrix, B::AbstractMatrix, Γ::CheckerboardMatrix)

Evaluate the matrix-matrix product `A=B⋅Γ`.
"""
function mul!(A::AbstractMatrix, B::AbstractMatrix, Γ::CheckerboardMatrix)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    copyto!(A,B)
    checkerboard_rmul!(A, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=inverted)

    return nothing
end

"""
    mul!(A::AbstractMatrix, B::AbstractMatrix, Γ::CheckerboardMatrix, color::Int)

Evaluate the matrix-matrix product `A=B⋅Γ[c]` where `Γ[c]` is the matrix associated with
the `color` checkerboard color matrix.
"""
function mul!(A::AbstractMatrix, B::AbstractMatrix, Γ::CheckerboardMatrix, color::Int)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    copyto!(A,B)
    checkerboard_color_rmul!(A, color, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=inverted)

    return nothing
end

"""
    ldiv!(u::AbstractVector, Γ::CheckerboardMatrix, v::AbstractVector)

Evaluate the matrix-vector product `u=Γ⁻¹⋅v`.
"""
function ldiv!(u::AbstractVector, Γ::CheckerboardMatrix, v::AbstractVector)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    copyto!(u,v)
    checkerboard_lmul!(u, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=!inverted)

    return nothing
end

"""
    ldiv!(u::AbstractVector, Γ::CheckerboardMatrix, v::AbstractVector, color::Int)

Evaluate the matrix-vector product `u=Γ⁻¹[c]⋅v` where `Γ⁻¹[c]` is the inverse of the matrix associated with the
`color` checkerboard color matrix.
"""
function ldiv!(u::AbstractVector, Γ::CheckerboardMatrix, v::AbstractVector, color::Int)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    copyto!(u,v)
    checkerboard_color_lmul!(u, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=!inverted)

    return nothing
end

"""
    ldiv!(A::AbstractVector, Γ::CheckerboardMatrix, B::AbstractVector)

Evaluate the matrix-matrix product `A=Γ⁻¹⋅B`.
"""
function ldiv!(A::AbstractMatrix, Γ::CheckerboardMatrix, B::AbstractMatrix)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    copyto!(A,B)
    checkerboard_lmul!(A, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=!inverted)

    return nothing
end

"""
    ldiv!(A::AbstractVector, Γ::CheckerboardMatrix, B::AbstractVector, color::Int)

Evaluate the matrix-matrix product `A=Γ⁻¹[c]⋅B` where `Γ⁻¹[c]` is the inverse of the matrix associated with the
`color` checkerboard color matrix.
"""
function ldiv!(A::AbstractMatrix, Γ::CheckerboardMatrix, B::AbstractMatrix, color::Int)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    copyto!(A,B)
    checkerboard_color_lmul!(A, color, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=!inverted)

    return nothing
end


"""
    lmul!(Γ::CheckerboardMatrix, A::AbstractMatrix)

Evaluate in-place the matrix-matrix product `A=Γ⋅A`, where `A` gets over-written.
"""
function lmul!(Γ::CheckerboardMatrix, A::AbstractMatrix)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    checkerboard_lmul!(A, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=inverted)

    return nothing
end

"""
    lmul!(Γ::CheckerboardMatrix, A::AbstractMatrix, color::Int)

Evaluate in-place the matrix-matrix product `A=Γ[c]⋅A`, where `Γ[c]` is the matrix associated with the
`color` checkerboard color matrix.
"""
function lmul!(Γ::CheckerboardMatrix, A::AbstractMatrix, color::Int)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    checkerboard_color_lmul!(A, color, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=inverted)

    return nothing
end


"""
    rmul!(A::AbstractMatrix, Γ::CheckerboardMatrix)

Evaluate in-place the matrix-matrix product `A=A⋅Γ`, where `A` gets over-written.
"""
function rmul!(A::AbstractMatrix, Γ::CheckerboardMatrix)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    checkerboard_rmul!(A, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=inverted)

    return nothing
end

"""
    rmul!(A::AbstractMatrix, Γ::CheckerboardMatrix, color::Int)

Evaluate in-place the matrix-matrix product `A=A⋅Γ[c]`, where `Γ[c]` is the matrix associated with the
`color` checkerboard color matrix.
"""
function rmul!(A::AbstractMatrix, Γ::CheckerboardMatrix, color::Int)

    (; transposed, inverted, neighbor_table, coshΔτt, sinhΔτt, colors) = Γ
    checkerboard_color_rmul!(A, color, neighbor_table, coshΔτt, sinhΔτt, colors, transposed=transposed, inverted=inverted)

    return nothing
end