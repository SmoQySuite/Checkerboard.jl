using LinearAlgebra
using Checkerboard
using Test

@testset "Checkerboard.jl" begin
    
    # construct square lattice neighbor table
    L  = 12 # lattice size
    nt = Vector{Int}[] # list of neighbors
    for y in 1:L, x in 1:L # add x direction neighbors
        s  = (y-1)*L + x
        x′ = mod1(x+1,L)
        s′ = (y-1)*L + x′
        push!(nt,[s,s′])
    end
    for y in 1:L, x in 1:L # add y direction neighbors
        s  = (y-1)*L + x
        y′ = mod1(y+1,L)
        s′ = (y′-1)*L + x
        push!(nt,[s,s′])
    end
    nt = hcat(nt...) # final neighbor table

    # corresponding hoppig for each pair of neighbors in the neighbor table
    t = fill(1.0, size(nt,2))

    # discretization in imaginary time
    Δτ = 0.1

    # calculate exact exponentiated hopping matrix exp(-Δτ⋅K)
    K = zeros(Float64, L^2, L^2)
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
    I_dense   = Matrix{Float64}(I,L^2,L^2)
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
