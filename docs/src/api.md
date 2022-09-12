# API

## Checkerboard Matrix Type

- [`CheckerboardMatrix`](@ref)
- [`checkerboard_matrices`](@ref)
- [`update!`](@ref)

```@docs
CheckerboardMatrix
CheckerboardMatrix(::Matrix{Int}, ::AbstractVector{T}, ::E; ::Bool, ::Bool) where {T, E<:AbstractFloat}
CheckerboardMatrix(::CheckerboardMatrix{T}; ::Bool, ::Bool, new_matrix::Bool=false) where {T}
checkerboard_matrices
update!
```

## Overloaded Functions

- [`size`](@ref)
- [`transpose`](@ref)
- [`adjoint`](@ref)
- [`inv`](@ref)
- [`mul!`](@ref)
- [`lmul!`](@ref)
- [`rmul!`](@ref)
- [`ldiv!`](@ref)
- [`rdiv!`](@ref)

```@docs
Base.size
Base.transpose
Base.adjoint
Base.inv
LinearAlgebra.mul!
LinearAlgebra.lmul!
LinearAlgebra.rmul!
LinearAlgebra.ldiv!
LinearAlgebra.rdiv!
```

## Developer API

- [`Checkerboard.Continuous`](@ref)
- [`Checkerboard.AbstractVecOrMat`](@ref)
- [`checkerboard_decomposition!`](@ref)
- [`checkerboard_lmul!`](@ref)
- [`checkerboard_rmul!`](@ref)
- [`checkerboard_color_lmul!`](@ref)
- [`checkerboard_color_rmul!`](@ref)

```@docs
Checkerboard.AbstractVecOrMat
Checkerboard.Continuous
checkerboard_decomposition!
checkerboard_lmul!
checkerboard_rmul!
checkerboard_color_lmul!
checkerboard_color_rmul!
```