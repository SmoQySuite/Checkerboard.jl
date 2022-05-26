# Usage

This section demonstrates how to use the [Checkerboard.jl](https://github.com/cohensbw/Checkerboard.jl) package.

## The Square Lattice

As an example, we will demonstrate how to construct and apply the checkerboard decomposition method in the case
of a square lattice with isotropic nearest-neighbor hopping.
We begin by importing all relevant packages.

```@example square_lattice
using LinearAlgebra
using BenchmarkTools
using LatticeUtilities
using Checkerboard

# set number of BLAS threads to 1 for fair benchmarking purposes.
BLAS.set_num_threads(1);
```

The next step is to construct the neighbor table, and record the corresponding hopping amplitudes,
for a square lattice with isotropic nearest-neighbor hopping.
We also define a discretization in imaginary time ``\Delta\tau``, which is used as the
small parameter in the checkerboard decomposition approximation.
We will use the package [LatticeUtilities.jl](https://github.com/cohensbw/LatticeUtilities.jl)
to assist with constructing the neighbor table.

```@example square_lattice
# construct square lattice neighbor table
L       = 12 # lattice size
square  = UnitCell(lattice_vecs = [[1.,0.],[0.,1.]], basis_vecs = [[0.,0.]])
lattice = Lattice(L = [L,L], periodic = [true,true])
bond_x  = Bond(orbitals = [1,1], displacement = [1,0])
bond_y  = Bond(orbitals = [1,1], displacement = [0,1])
nt      = build_neighbor_table([bond_x,bond_y], square, lattice)
N       = get_num_sites(square, lattice)

# corresponding hopping for each pair of neighbors in the neighbor table
t = fill(1.0, size(nt,2))

# discretization in imaginary time i.e. the small parameter
# used in the checkerboard approximation
Δτ = 0.1

nt
```

Next, for comparison purposes, we explicitly construct the hopping matrix ``K`` and exponentiate it,
giving us the exact matrix ``e^{-\Delta\tau K}`` that the checkerboard decomposition matrix
``\Gamma`` is intended to approximate.

```@example square_lattice
K = zeros(Float64, N, N)
for c in 1:size(nt,2)
    i      = nt[1,c]
    j      = nt[2,c]
    K[i,j] = -t[c]
    K[j,i] = -conj(t[c])
end

expnΔτK = Hermitian(exp(-Δτ*K))
nothing; # hide
```

An instance of the [`CheckerboardMatrix`](@ref) type representing the checkerboard
decomposition matrix ``\Gamma`` can be instantiated as follows.

```@example square_lattice
Γ = CheckerboardMatrix(nt, t, Δτ)
nothing; # hide
```

It is also straight forward to efficiently construct representations of both the transpose and inverse
of the checkerboard matrix ``\Gamma`` using the [`transpose`](@ref) and [`inv`](@ref) methods.

```@example square_lattice
Γᵀ  = transpose(Γ)
Γ⁻¹ = inv(Γ)
nothing; # hide
```

Matrix-matrix multiplications involving ``\Gamma`` can be performed using the [`mul!`](@ref)
method just as you would with a standard matrix. Here we benchmark the matrix-matrix multiplication
using the checkerboard decomposition as compared to multiplication by the dense matrix ``e^{-\Delta\tau K}``.

```@example square_lattice
I_dense = Matrix{Float64}(I,N,N)
B       = similar(I_dense)
@btime mul!(B, expnΔτK, I_dense);
@btime mul!(B, I_dense, expnΔτK);
@btime mul!(B, Γ,       I_dense);
@btime mul!(B, I_dense, Γ);
```

Using the [@btime](https://juliaci.github.io/BenchmarkTools.jl/stable/reference/#BenchmarkTools.@btime-Tuple) macro
exported by the [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl), we see that performing both
left and right matrix multiplies by the checkerboard matrix ``\Gamma`` is faster than multiplying by the ``e^{-\Delta\tau K}``.
However, we see that right matrix multiplies are significantly faster than left multiplies by ``\Gamma``, a result of
the access order into `I_dense` being unavoidably more efficient for the right multiply.
Therefore, it is important to keep this in mind when using this package to develop QMC codes:
wherever possible do right multiplies by the checkerboard matrix instead of left multiplies.

Let us quickly verify that ``\Gamma`` is a good approximation of ``e^{-\Delta\tau K}``.

```@example square_lattice
I_dense = Matrix{Float64}(I,N,N)
Γ_dense = similar(I_dense)
mul!(Γ_dense, Γ, I_dense)
norm(Γ_dense - expnΔτK) / norm(expnΔτK)
```

It is also possible to do in-place left and right multiplications by ``\Gamma`` using the
[`lmul!`](@ref) and [`rmul!`](@ref) methods respectively.

```@example square_lattice
A = Matrix{Float64}(I,N,N)
B = Matrix{Float64}(I,N,N)
rmul!(A,Γ) # A = A⋅Γ
lmul!(Γ,B) # B = Γ⋅B
A ≈ B
```

The checkerboard matrix is in reality a product of several individually sparse matrices that we will
refer to as checkerboard color matrices, such that ``\Gamma = \prod_{c} \Gamma_c = \Gamma_C \dots \Gamma_1``.
Let us check to see how many checkerboard color matrices there are in our checkerboard decomposition.

```@example square_lattice
Γ.Ncolors
```

It is also possible to multiply by a single one of the ``\Gamma_c`` matrices with the [`mul!`](@ref), [`lmul!`](@ref)
and [`rmul!`](@ref) methods.

```@example square_lattice
Γ₁ = Matrix{Float64}(I, N, N)
Γ₂ = Matrix{Float64}(I, N, N)
Γ₃ = Matrix{Float64}(I, N, N)
Γ₄ = Matrix{Float64}(I, N, N)

mul!(Γ₁, Γ, I_dense, 1)
mul!(Γ₂, I_dense, Γ, 2)
lmul!(Γ, Γ₃, 3)
rmul!(Γ₄, Γ, 4)

Γ_dense ≈ (Γ₄*Γ₃*Γ₂*Γ₁)
```

Lastly, it is easy to update the checkerboard decomposition with new hopping parameter values
using the [`update!`](@ref) method, including disordered values.

```@example square_lattice
@. t = 1.0 + 0.1*randn()
update!(Γ, t, Δτ)
nothing; # hide
```

This is important for QMC simulations of models like the Su-Schrieffer-Heeger model, where an
electron-phonon coupling modulates the hopping amplitude, resulting in the hopping amplitudes changing
over the course of a simulation.