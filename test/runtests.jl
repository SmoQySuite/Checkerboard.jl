using LinearAlgebra
using Checkerboard
using LatticeUtilities
using Test

@testset "Checkerboard.jl" begin
    
    # construct square lattice neighbor table for nearest neighbor hopping
    L       = 12
    t0      = 1.0
    Δτ      = 0.1
    square  = UnitCell([[1.,0.],[0.,1.]], [[0.,0.]])
    lattice = Lattice([L,L], [true,true])
    bond_x  = Bond([1,1], [1,0])
    bond_y  = Bond([1,1], [0,1])
    nt      = build_neighbor_table([bond_x,bond_y], square, lattice)
    t       = fill(t0, size(nt,2))

    # calculate exact exponentiated hopping matrix exp(-Δτ⋅K)
    K = zeros(typeof(t0),L^2,L^2)
    for c in 1:size(nt,2)
        i      = nt[1,c]
        j      = nt[2,c]
        K[i,j] = -t[c]
        K[j,i] = -t[c]
    end
    expnΔτK = Hermitian(exp(-Δτ*K))

    # define checkerboard matrix
    Γ = CheckerboardMatrix(nt, t, Δτ)

    # take transpose of checkerboard matrix
    Γᵀ = transpose(Γ)

    # take inverse of checkerboard matrix
    Γ⁻¹ = inv(Γ)

    # build dense versions of matrices
    I_dense   = Matrix{typeof(t0)}(I,L^2,L^2)
    Γ_dense   = similar(I_dense)
    Γᵀ_dense  = similar(I_dense)
    Γ⁻¹_dense = similar(I_dense)
    mul!(Γ_dense,   Γ,   I_dense)
    mul!(Γᵀ_dense,  Γᵀ,  I_dense)
    mul!(Γ⁻¹_dense, Γ⁻¹, I_dense)

    @test Γ_dense ≈ transpose(Γᵀ_dense)
    @test Γ_dense ≈ inv(Γ⁻¹_dense)
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
