module Checkerboard

using LinearAlgebra
import Base: size, transpose, inv, adjoint, eltype
import LinearAlgebra: mul!, rmul!, lmul!, ldiv!, rdiv!

@doc raw"""
    Continuous{T<:AbstractFloat}

Real or complex floating-point number types with precision `T`, i.e. `Union{T, Complex{T}}`.
The bare `Continuous` (without a parameter) covers every precision, for example `Float64`,
`ComplexF32`, and `BigFloat`, but excludes integer and rational types.
"""
const Continuous{T<:AbstractFloat} = Union{T, Complex{T}}

# methods to construct the initial checkerboard decomposition
include("construct_checkerboard.jl")
export checkerboard_decomposition!

# low-level routines for in-place matrix-vector and matrix-matrix products
include("matrix_matrix_mul.jl")
include("matrix_vector_mul.jl")
export checkerboard_lmul!, checkerboard_rmul!
export checkerboard_color_lmul!, checkerboard_color_rmul!

# define CheckerboardMatrix type to represent the checkerboard decomposition matrix
include("CheckerboardMatrix.jl")
export CheckerboardMatrix, checkerboard_matrices, update!

end
