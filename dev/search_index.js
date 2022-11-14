var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/#Checkerboard-Matrix-Type","page":"API","title":"Checkerboard Matrix Type","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"CheckerboardMatrix\ncheckerboard_matrices\nupdate!","category":"page"},{"location":"api/","page":"API","title":"API","text":"CheckerboardMatrix\nCheckerboardMatrix(::Matrix{Int}, ::AbstractVector{T}, ::E; ::Bool, ::Bool) where {T, E<:AbstractFloat}\nCheckerboardMatrix(::CheckerboardMatrix{T}; ::Bool, ::Bool, new_matrix::Bool=false) where {T}\ncheckerboard_matrices\nupdate!","category":"page"},{"location":"api/#Checkerboard.CheckerboardMatrix","page":"API","title":"Checkerboard.CheckerboardMatrix","text":"CheckerboardMatrix{T<:Union{AbstractFloat, Complex{<:AbstractFloat}}}\n\nA type to represent a checkerboard decomposition matrix.\n\nFields\n\ntransposed::Bool: If the checkerboard matrix is transposed.\ninverted::Bool: If the checkerboard matrix is inverted.\nNsites::Int: Number of sites/orbitals in lattice.\nNneighbors::Int: Number of neighbors.\nNcolors::Int: Number of checkerboard colors/groups.\nneighbor_table::Matrix{Int}: Neighbor table represented by a (2,Nneighbors) dimension matrix, where each column contains a pair of neighboring sites.\ncoshΔτt::Vector{T}: The cosh(Deltatau t) values.\nsinhΔτt::Vector{T}: The sinh(Deltatau t) values.\nperm::Vector{Int}: The checkerboard permutation order relative to the ordering of the original neighbor table.\ninv_perm::Vector{Int}: Inverse permuation of perm.\ncolors::Matrix{Int}: The bounds of each checkerboard color/group in neighbor_table.\n\n\n\n\n\n","category":"type"},{"location":"api/#Checkerboard.CheckerboardMatrix-Union{Tuple{E}, Tuple{T}, Tuple{Matrix{Int64}, AbstractVector{T}, E}} where {T, E<:AbstractFloat}","page":"API","title":"Checkerboard.CheckerboardMatrix","text":"CheckerboardMatrix(neighbor_table::Matrix{Int}, t::AbstractVector{T}, Δτ::E;\n    transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:AbstractFloat}\n\nGiven a neighbor_table along with the corresponding hopping amplitudes t and discretzation in imaginary time Δτ, construct an instance of the type CheckerboardMatrix. \n\n\n\n\n\n","category":"method"},{"location":"api/#Checkerboard.CheckerboardMatrix-Union{Tuple{CheckerboardMatrix{T}}, Tuple{T}} where T","page":"API","title":"Checkerboard.CheckerboardMatrix","text":"CheckerboardMatrix(Γ::CheckerboardMatrix;\n    transposed::Bool=false, inverted::Bool=false, new_matrix::Bool=false)\n\nConstruct a new instance of CheckerboardMatrix based on a current instance Γ of CheckerboardMatrix. If new_matrix=true then allocate new coshΔτt and sinhΔτt arrays.\n\n\n\n\n\n","category":"method"},{"location":"api/#Checkerboard.checkerboard_matrices","page":"API","title":"Checkerboard.checkerboard_matrices","text":"checkerboard_matrices(neighbor_table::Matrix{Int}, t::AbstractMatrix{T}, Δτ::E;\n    transposed::Bool=false, inverted::Bool=false)\n\nReturn a vector of CheckerboardMatrix, one for each column of t, all sharing the same neighbor_table.\n\n\n\n\n\n","category":"function"},{"location":"api/#Checkerboard.update!","page":"API","title":"Checkerboard.update!","text":"update!(Γ::CheckerboardMatrix{T}, t::AbstractVector{T}, Δτ::E) where {T<:Continuous, E<:AbstractFloat}\n\nUpdate the CheckerboardMatrix based on new hopping parameters t and discretezation in imaginary time Δτ. \n\n\n\n\n\nupdate!(Γs::AbstractVector{CheckerboardMatrix{T}}, t::AbstractVector{T}, Δτ::E) where {T<:Continuous, E<:AbstractFloat}\n\nUpdate a vector of CheckerboardMatrix based on new hopping parameters t and discretezation in imaginary time Δτ. \n\n\n\n\n\nupdate!(coshΔτt::AbstractVector{T}, sinhΔτt::AbstractVector{T}, t::AbstractVector{T}, Δτ::E) where {T<:Continuous, E<:AbstractFloat}\n\nUpdate the coshΔτt and sinhΔτt associated with a checkerboard decomposition based on new hopping parameters t and discretezation in imaginary time Δτ. \n\n\n\n\n\n","category":"function"},{"location":"api/#Overloaded-Functions","page":"API","title":"Overloaded Functions","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"size\ntranspose\nadjoint\ninv\nmul!\nlmul!\nrmul!\nldiv!\nrdiv!","category":"page"},{"location":"api/","page":"API","title":"API","text":"Base.size\nBase.transpose\nBase.adjoint\nBase.inv\nLinearAlgebra.mul!\nLinearAlgebra.lmul!\nLinearAlgebra.rmul!\nLinearAlgebra.ldiv!\nLinearAlgebra.rdiv!","category":"page"},{"location":"api/#Base.size","page":"API","title":"Base.size","text":"size(Γ::CheckerboardMatrix)\n\nsize(Γ::CheckerboardMatrix, dim::Int)\n\nReturn the dimensions of the checkerboard decomposition matrix Γ.\n\n\n\n\n\n","category":"function"},{"location":"api/#Base.transpose","page":"API","title":"Base.transpose","text":"transpose(Γ::CheckerboardMatrix)\n\nReturn a transposed/adjoint version of the checkerboard matrix Γ.\n\n\n\n\n\n","category":"function"},{"location":"api/#Base.adjoint","page":"API","title":"Base.adjoint","text":"adjoint(Γ::CheckerboardMatrix)\n\nReturn a transposed/adjoint version of the checkerboard matrix Γ.\n\n\n\n\n\n","category":"function"},{"location":"api/#Base.inv","page":"API","title":"Base.inv","text":"inv(Γ::CheckerboardMatrix)\n\nReturn the inverse of the checkerboard matrix Γ.\n\n\n\n\n\n","category":"function"},{"location":"api/#LinearAlgebra.mul!","page":"API","title":"LinearAlgebra.mul!","text":"mul!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, v::AbstractVecOrMat)\n\nEvaluate the matrix-vector or matrix-matrix product u=Γ⋅v.\n\n\n\n\n\nmul!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, v::AbstractVecOrMat, color::Int)\n\nEvaluate the matrix-vector or matrix-matrix product u=Γ[c]⋅v where Γ[c] is the matrix associated with the color checkerboard color matrix.\n\n\n\n\n\nmul!(u::AbstractVecOrMat, v::AbstractVecOrMat, Γ::CheckerboardMatrix)\n\nEvaluate the matrix-vector or matrix-matrix product u=v⋅Γ.\n\n\n\n\n\nmul!(u::AbstractVecOrMat, v::AbstractVecOrMat, Γ::CheckerboardMatrix, color::Int)\n\nEvaluate the matrix-vector or matrix-matrix product u=v⋅Γ[c] where Γ[c] is the matrix associated with the color checkerboard color matrix.\n\n\n\n\n\n","category":"function"},{"location":"api/#LinearAlgebra.lmul!","page":"API","title":"LinearAlgebra.lmul!","text":"lmul!(Γ::CheckerboardMatrix, u::AbstractVecOrMat)\n\nEvaluate in-place the matrix-vector or matrix-matrix product u=Γ⋅u, where u gets over-written.\n\n\n\n\n\nlmul!(Γ::CheckerboardMatrix, u::AbstractVecOrMat, color::Int)\n\nEvaluate in-place the matrix-vector or matrix-matrix product u=Γ[c]⋅u, where Γ[c] is the matrix associated with the color checkerboard color matrix.\n\n\n\n\n\n","category":"function"},{"location":"api/#LinearAlgebra.rmul!","page":"API","title":"LinearAlgebra.rmul!","text":"rmul!(u::AbstractVecOrMat, Γ::CheckerboardMatrix)\n\nEvaluate in-place the matrix-vector or matrix-matrix product u=u⋅Γ, where u gets over-written.\n\n\n\n\n\nrmul!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, color::Int)\n\nEvaluate in-place the matrix-vector or matrix-matrix product u=u⋅Γ[c], where Γ[c] is the matrix associated with the color checkerboard color matrix.\n\n\n\n\n\n","category":"function"},{"location":"api/#LinearAlgebra.ldiv!","page":"API","title":"LinearAlgebra.ldiv!","text":"ldiv!(Γ::CheckerboardMatrix, v::AbstractVecOrMat)\n\nEvaluate in-place the matrix-vector or matrix-matrix product v=Γ⁻¹⋅v.\n\n\n\n\n\nldiv!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, v::AbstractVecOrMat)\n\nEvaluate the matrix-vector or matrix-matrix product u=Γ⁻¹⋅v.\n\n\n\n\n\nldiv!(Γ::CheckerboardMatrix, v::AbstractVecOrMat, color::Int)\n\nEvaluate in-place the matrix-vector or matrix-matrix product v=Γ⁻¹[c]⋅v where Γ⁻¹[c] is the inverse of the matrix associated with the color checkerboard color matrix.\n\n\n\n\n\nldiv!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, v::AbstractVecOrMat, color::Int)\n\nEvaluate the matrix-vector or matrix-matrix product u=Γ⁻¹[c]⋅v where Γ⁻¹[c] is the inverse of the matrix associated with the color checkerboard color matrix.\n\n\n\n\n\n","category":"function"},{"location":"api/#LinearAlgebra.rdiv!","page":"API","title":"LinearAlgebra.rdiv!","text":"rdiv!(u::AbstractVecOrMat, Γ::CheckerboardMatrix)\n\nEvaluate in-place the matrix-vector or matrix-matrix product u=u⋅Γ⁻¹, where u gets over-written.\n\n\n\n\n\nrdiv!(u::AbstractVecOrMat, v::AbstractVecOrMat, Γ::CheckerboardMatrix)\n\nEvaluate the matrix-vector or matrix-matrix product u=v⋅Γ⁻¹, where u gets over-written.\n\n\n\n\n\nrdiv!(u::AbstractVecOrMat, Γ::CheckerboardMatrix, color::Int)\n\nEvaluate in-place the matrix-vector or matrix-matrix product u=u⋅Γ⁻¹[c], where Γ[c] is the matrix associated with the color checkerboard color matrix.\n\n\n\n\n\nrdiv!(u::AbstractVecOrMat, v::AbstractVecOrMat, Γ::CheckerboardMatrix, color::Int)\n\nEvaluate the matrix-vector or matrix-matrix product u=v⋅Γ⁻¹[c], where Γ[c] is the matrix associated with the color checkerboard color matrix.\n\n\n\n\n\n","category":"function"},{"location":"api/#Developer-API","page":"API","title":"Developer API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Checkerboard.Continuous\nCheckerboard.AbstractVecOrMat\ncheckerboard_decomposition!\ncheckerboard_lmul!\ncheckerboard_rmul!\ncheckerboard_color_lmul!\ncheckerboard_color_rmul!","category":"page"},{"location":"api/","page":"API","title":"API","text":"Checkerboard.AbstractVecOrMat\nCheckerboard.Continuous\ncheckerboard_decomposition!\ncheckerboard_lmul!\ncheckerboard_rmul!\ncheckerboard_color_lmul!\ncheckerboard_color_rmul!","category":"page"},{"location":"api/#Checkerboard.AbstractVecOrMat","page":"API","title":"Checkerboard.AbstractVecOrMat","text":"AbstractVecOrMat{T} = Union{AbstractVector{T}, AbstractMatrix{T}}\n\nAbstract type defining union of AbstractVector and AbstractMatrix.\n\n\n\n\n\n","category":"type"},{"location":"api/#Checkerboard.Continuous","page":"API","title":"Checkerboard.Continuous","text":"Continuous = Union{AbstractFloat, Complex{<:AbstractFloat}}\n\nAbstract type to represent continuous real or complex numbers.\n\n\n\n\n\n","category":"type"},{"location":"api/#Checkerboard.checkerboard_decomposition!","page":"API","title":"Checkerboard.checkerboard_decomposition!","text":"checkerboard_decomposition!(neighbor_table::Matrix{Int})\n\nGiven a neighbor_table, construct the checkerboard decomposition, which results in the columns of neighbor_table being re-ordered in-place. Two additional arrays are also returned:\n\nperm::Vector{Int}: The permutation used to re-order the columns of neighbor_table according to the checkerboard decomposition.\ncolors::Matrix{Int}: Marks the column indice boundaries for each checkerboard color in neighbor_table.\n\n\n\n\n\n","category":"function"},{"location":"api/#Checkerboard.checkerboard_lmul!","page":"API","title":"Checkerboard.checkerboard_lmul!","text":"checkerboard_lmul!(B::AbstractMatrix{T}, neighbor_table::Matrix{Int},\n    coshΔτt::AbstractVector{T}, sinhΔτt::AbstractVector{T}, colors::Matrix{Int};\n    transposed::Bool=false, inverted::Bool=false) where {T<:Continuous}\n\nEvaluate the matrix-matrix product in-place B=Γ⋅B where Γ is the checkerboard matrix.\n\n\n\n\n\ncheckerboard_lmul!(v::AbstractVector{T}, neighbor_table::Matrix{Int},\n    coshΔτt::AbstractVector{T}, sinhΔτt::AbstractVector{T}, colors::Matrix{Int};\n    transposed::Bool=false, inverted::Bool=false) where {T<:Continuous}\n\nMultiply in-place the vector v by the checkerboard matrix.\n\n\n\n\n\n","category":"function"},{"location":"api/#Checkerboard.checkerboard_rmul!","page":"API","title":"Checkerboard.checkerboard_rmul!","text":"checkerboard_rmul!(B::AbstractMatrix{T}, neighbor_table::Matrix{Int},\n    coshΔτt::AbstractVector{T}, sinhΔτt::AbstractVector{T}, colors::Matrix{Int};\n    transposed::Bool=false, inverted::Bool=false) where {T<:Continuous}\n\nEvaluate the matrix-matrix product in-place B=B⋅Γ where Γ is the checkerboard matrix.\n\n\n\n\n\n","category":"function"},{"location":"api/#Checkerboard.checkerboard_color_lmul!","page":"API","title":"Checkerboard.checkerboard_color_lmul!","text":"checkerboard_color_lmul!(B::AbstractMatrix{T}, color::Int, neighbor_table::Matrix{Int},\n    coshΔτt::AbstractVector{T}, sinhΔτt::AbstractVector{T}, colors::Matrix{Int};\n    inverted::Bool=false) where {T<:Continuous}\n\nEvaluate the matrix-matrix product in-place B=Γ[c]⋅B where Γ[c] is the color checkerboard color matrix.\n\n\n\n\n\ncheckerboard_color_lmul!(v::AbstractVector{T}, color::Int, neighbor_table::Matrix{Int},\n    coshΔτt::AbstractVector{T}, sinhΔτt::AbstractVector{T}, colors::Matrix{Int};\n    transposed::Bool=false, inverted::Bool=false) where {T<:Continuous}\n\nMultiply in-place the vector v by the color checkerboard color matrix.\n\n\n\n\n\n","category":"function"},{"location":"api/#Checkerboard.checkerboard_color_rmul!","page":"API","title":"Checkerboard.checkerboard_color_rmul!","text":"checkerboard_color_rmul!(B::AbstractMatrix{T}, color::Int, neighbor_table::Matrix{Int},\n    coshΔτt::AbstractVector{T}, sinhΔτt::AbstractVector{T}, colors::Matrix{Int};\n    inverted::Bool=false) where {T<:Continuous}\n\nEvaluate the matrix-matrix product in-place B=B⋅Γ[c] where Γ[c] is the color checkerboard color matrix.\n\n\n\n\n\n","category":"function"},{"location":"usage/#Usage","page":"Usage","title":"Usage","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"This section demonstrates how to use the Checkerboard.jl package.","category":"page"},{"location":"usage/#The-Square-Lattice","page":"Usage","title":"The Square Lattice","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"As an example, we will demonstrate how to construct and apply the checkerboard decomposition method in the case of a square lattice with isotropic nearest-neighbor hopping. We begin by importing all relevant packages.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"using LinearAlgebra\nusing BenchmarkTools\nusing LatticeUtilities\nusing Checkerboard\n\n# set number of BLAS threads to 1 for fair benchmarking purposes.\nBLAS.set_num_threads(1);","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"The next step is to construct the neighbor table, and record the corresponding hopping amplitudes, for a square lattice with isotropic nearest-neighbor hopping. We also define a discretization in imaginary time Deltatau, which is used as the small parameter in the checkerboard decomposition approximation. We will use the package LatticeUtilities.jl to assist with constructing the neighbor table.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"# construct square lattice neighbor table\nL       = 12 # lattice size\nsquare  = UnitCell(lattice_vecs = [[1.,0.],[0.,1.]], basis_vecs = [[0.,0.]])\nlattice = Lattice(L = [L,L], periodic = [true,true])\nbond_x  = Bond(orbitals = [1,1], displacement = [1,0])\nbond_y  = Bond(orbitals = [1,1], displacement = [0,1])\nnt      = build_neighbor_table([bond_x,bond_y], square, lattice)\nN       = get_num_sites(square, lattice)\n\n# corresponding hopping for each pair of neighbors in the neighbor table\nt = fill(1.0, size(nt,2))\n\n# discretization in imaginary time i.e. the small parameter\n# used in the checkerboard approximation\nΔτ = 0.1\n\nnt","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Next, for comparison purposes, we explicitly construct the hopping matrix K and exponentiate it, giving us the exact matrix e^-Deltatau K that the checkerboard decomposition matrix Gamma is intended to approximate.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"K = zeros(Float64, N, N)\nfor c in 1:size(nt,2)\n    i      = nt[1,c]\n    j      = nt[2,c]\n    K[i,j] = -t[c]\n    K[j,i] = -conj(t[c])\nend\n\nexpnΔτK = Hermitian(exp(-Δτ*K))\nnothing; # hide","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"An instance of the CheckerboardMatrix type representing the checkerboard decomposition matrix Gamma can be instantiated as follows.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Γ = CheckerboardMatrix(nt, t, Δτ)\nnothing; # hide","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"It is also straight forward to efficiently construct representations of both the transpose and inverse of the checkerboard matrix Gamma using the transpose and inv methods.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Γᵀ  = transpose(Γ)\nΓ⁻¹ = inv(Γ)\nnothing; # hide","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Matrix-matrix multiplications involving Gamma can be performed using the mul! method just as you would with a standard matrix. Here we benchmark the matrix-matrix multiplication using the checkerboard decomposition as compared to multiplication by the dense matrix e^-Deltatau K.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"I_dense = Matrix{Float64}(I,N,N)\nB       = similar(I_dense)\n@btime mul!(B, expnΔτK, I_dense); # 131.958 μs (0 allocations: 0 bytes)\n@btime mul!(B, I_dense, expnΔτK); # 129.375 μs (0 allocations: 0 bytes)\n@btime mul!(B, Γ,       I_dense); # 38.334 μs (0 allocations: 0 bytes)\n@btime mul!(B, I_dense, Γ); # 17.500 μs (0 allocations: 0 bytes)","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Using the @btime macro exported by the BenchmarkTools.jl, we see that performing both left and right matrix multiplies by the checkerboard matrix Gamma is faster than multiplying by the e^-Deltatau K. However, we see that right matrix multiplies are significantly faster than left multiplies by Gamma, a result of the access order into I_dense being unavoidably more efficient for the right multiply. Therefore, it is important to keep this in mind when using this package to develop QMC codes: wherever possible do right multiplies by the checkerboard matrix instead of left multiplies.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Let us quickly verify that Gamma is a good approximation of e^-Deltatau K.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"I_dense = Matrix{Float64}(I,N,N)\nΓ_dense = similar(I_dense)\nmul!(Γ_dense, Γ, I_dense)\nnorm(Γ_dense - expnΔτK) / norm(expnΔτK)","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"It is also possible to do in-place left and right multiplications by Gamma using the lmul! and rmul! methods respectively.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"A = Matrix{Float64}(I,N,N)\nB = Matrix{Float64}(I,N,N)\nrmul!(A,Γ) # A = A⋅Γ\nlmul!(Γ,B) # B = Γ⋅B\nA ≈ B","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"The checkerboard matrix is in reality a product of several individually sparse matrices that we will refer to as checkerboard color matrices, such that Gamma = prod_c Gamma_c = Gamma_C dots Gamma_1. Let us check to see how many checkerboard color matrices there are in our checkerboard decomposition.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Γ.Ncolors","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"It is also possible to multiply by a single one of the Gamma_c matrices with the mul!, lmul! and rmul! methods.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Γ₁ = Matrix{Float64}(I, N, N)\nΓ₂ = Matrix{Float64}(I, N, N)\nΓ₃ = Matrix{Float64}(I, N, N)\nΓ₄ = Matrix{Float64}(I, N, N)\n\nmul!(Γ₁, Γ, I_dense, 1)\nmul!(Γ₂, I_dense, Γ, 2)\nlmul!(Γ, Γ₃, 3)\nrmul!(Γ₄, Γ, 4)\n\nΓ_dense ≈ (Γ₄*Γ₃*Γ₂*Γ₁)","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Lastly, it is easy to update the checkerboard decomposition with new hopping parameter values using the update! method, including disordered values.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"@. t = 1.0 + 0.1*randn()\nupdate!(Γ, t, Δτ)\nnothing; # hide","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"This is important for QMC simulations of models like the Su-Schrieffer-Heeger model, where an electron-phonon coupling modulates the hopping amplitude, resulting in the hopping amplitudes changing over the course of a simulation.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = Checkerboard","category":"page"},{"location":"#Checkerboard.jl","page":"Home","title":"Checkerboard.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for the Checkerboard.jl package exports a flexible implementation of the checkerboard decomposition algorithm for generating a sparse approximation to the exponentiated kinetic energy matrix that appears in quantum Monte Carlo (QMC) simulations of lattice fermions.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install Checkerboard.jl run following in the Julia REPL:","category":"page"},{"location":"","page":"Home","title":"Home","text":"] add Checkerboard","category":"page"}]
}
