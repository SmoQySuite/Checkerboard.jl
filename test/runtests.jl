using LinearAlgebra
using LatticeUtilities
using Checkerboard
using Test

@testset "Checkerboard.jl" begin
    
    # construct square lattice neighbor table
    L       = 12 # lattice size
    square  = UnitCell(lattice_vecs = [[1.,0.],[0.,1.]], basis_vecs = [[0.,0.]])
    lattice = Lattice(L = [L,L], periodic = [true,true])
    bond_x  = Bond(orbitals = (1,1), displacement = [1,0])
    bond_y  = Bond(orbitals = (1,1), displacement = [0,1])
    nt      = build_neighbor_table([bond_x,bond_y], square, lattice)
    N       = nsites(square, lattice)

    # corresponding hoppig for each pair of neighbors in the neighbor table
    t = fill(1.0, size(nt,2))

    # discretization in imaginary time
    Δτ = 0.1

    # calculate exact exponentiated hopping matrix exp(-Δτ⋅K)
    K = zeros(Float64, L^2, L^2)
    for c in axes(nt,2)
        i      = nt[1,c]
        j      = nt[2,c]
        K[j,i] = -t[c]
        K[i,j] = conj(-t[c])
    end
    expnΔτK = Hermitian(exp(-Δτ*K))

    # define checkerboard matrix
    Γ = CheckerboardMatrix(nt, t, Δτ)

    # take transpose of checkerboard matrix
    Γᵀ = transpose(Γ)

    # take inverse of checkerboard matrix
    Γ⁻¹ = inv(Γ)

    # take inverse of adjoint checkerboard matrix
    Γ⁻ᵀ = transpose(Γ⁻¹)

    # test vector multiplication
    v  = randn(N)
    v′ = zeros(N)
    mul!(v′, Γ, v)
    lmul!(Γ⁻¹, v′)
    @test v′ ≈ v

    # build dense versions of matrices
    I_dense   = Matrix{Float64}(I,L^2,L^2)
    Γ_dense   = similar(I_dense)
    Γᵀ_dense  = similar(I_dense)
    Γ⁻¹_dense = similar(I_dense)
    Γ⁻ᵀ_dense = similar(I_dense)
    mul!(Γ_dense,   Γ,   I_dense)
    mul!(Γᵀ_dense,  Γᵀ,  I_dense)
    mul!(Γ⁻¹_dense, Γ⁻¹, I_dense)
    mul!(Γ⁻ᵀ_dense, Γ⁻ᵀ, I_dense)

    @test Γ_dense ≈ adjoint(Γᵀ_dense)
    @test Γ_dense ≈ inv(Γ⁻¹_dense)
    @test Γ⁻¹_dense ≈ adjoint(Γ⁻ᵀ_dense)
    @test norm(Γ_dense-expnΔτK)/norm(expnΔτK) < 0.01

    A = similar(Γ_dense)
    B = similar(Γ_dense)
    C = similar(Γ_dense)

    mul!(A,Γ,I_dense)
    mul!(B,I_dense,Γ)
    @test A ≈ B

    @. C = rand()
    mul!(A,Γ,C)
    mul!(B,C,Γ)
    @test !(A ≈ B)
end
