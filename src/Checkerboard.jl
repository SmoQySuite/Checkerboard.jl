module Checkerboard

using LinearAlgebra

import Base: size
import LinearAlgebra: mul!, rmul!, lmul!, ldiv!, transpose, inv

"""
Abstract type to represent continuous real or complex numbers.
"""
Continuous = Union{AbstractFloat, Complex{<:AbstractFloat}}

# methods to construct the initial checkerboard decomposition
include("construct_checkerboard.jl")
export checkerboard_decomposition!

# low-level routines for in-place matrix-vector and matrix-matrix products
include("matrix_matrix_mul.jl")
include("matrix_vector_mul.jl")
export checkerboard_lmul!, checkerboard_rmul!
export checkerboard_color_lmul!, checkerboard_color_rmul!

# define CheckerboardMatrix type to represent the checkerboard decomposition matrix
include("checkerboard_matrix.jl")
export CheckerboardMatrix, update!

end
