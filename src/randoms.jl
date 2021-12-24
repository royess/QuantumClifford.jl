using LinearAlgebra: inv, mul!
using Random: randperm, AbstractRNG, GLOBAL_RNG
using ILog2
import Nemo

##############################
# Random Paulis
##############################

"""A random Pauli operator on n qubits.

Use `realphase=true` to get operators with phase ±1 (excluding ±i)."""
random_pauli(rnd::AbstractRNG, n::Int; nophase=false, realphase=false) = PauliOperator(nophase ? 0x0 : (realphase ? rand(rnd,[0x0,0x2]) : rand(rnd,0x0:0x3)), rand(rnd,Bool,n), rand(rnd,Bool,n))
random_pauli(n::Int; nophase=false, realphase=false) = random_pauli(GLOBAL_RNG, n, nophase=nophase, realphase=realphase)
function random_pauli(rng::AbstractRNG,n::Int,p; nophase=false,nonidbranch=false)
    x = falses(n)
    z = falses(n)
    if nonidbranch
        definite = rand(1:n)
        p=(p/(1-(1-2p)^n) - 1/n/3)*n/(n-1)
    end
    for i in 1:n
        r = rand(rng)
        if nonidbranch && definite==i
            r *= 3p
        end
        if (r<=2p) x[i]=true end
        if (p<r<=3p) z[i]=true end
    end
    PauliOperator(nophase ? 0x0 : rand(rng,0x0:0x3), x, z)
end
random_pauli(n::Int, p; nophase=false, nonidbranch=false) = random_pauli(GLOBAL_RNG,n,p; nophase=nophase,nonidbranch=nonidbranch)

##############################
# Random Binary Matrices
##############################

function random_invertible_gf2(rng::AbstractRNG, n::Int)
    while true
        mat = rand(rng,Bool,n,n)
        gf2_isinvertible(mat) && return mat
    end
end
random_invertible_gf2(n::Int) = random_invertible_gf2(GLOBAL_RNG, n)

##############################
# Random Tableaux and Clifford
##############################

# function random_cnot_clifford(n) = ... #TODO

"""A random Stabilizer/Destabilizer tableau generated by the Bravyi-Maslov Algorithm 2 from [bravyi2020hadamard](@cite)."""
function random_destabilizer(rng::AbstractRNG, n::Int)
    hadamard, perm = quantum_mallows(rng, n)
    had_idxs = findall(i -> hadamard[i], 1:n)
    
    # delta, delta', gamma, gamma' appear in the canonical form
    # of a Clifford operator (Eq. 3/Theorem 1)
    # delta is unit lower triangular, gamma is symmetric
    F1 = zeros(Int8, 2n, 2n)
    F2 = zeros(Int8, 2n, 2n)
    delta   = @view F1[1:n, 1:n]
    delta_p = @view F2[1:n, 1:n]
    prod   = @view F1[n+1:2n, 1:n]
    prod_p = @view F2[n+1:2n, 1:n]
    gamma   = @view F1[1:n, n+1:2n]
    gamma_p = @view F2[1:n, n+1:2n]
    inv_delta   = @view F1[n+1:2n, n+1:2n]
    inv_delta_p = @view F2[n+1:2n, n+1:2n]
    for i in 1:n
        delta[i,i] = 1
        delta_p[i,i] = 1
        gamma_p[i,i] = rand(rng, 0:1)
    end
    
    # gamma_ii is zero if h[i] = 0
    for idx in had_idxs
        gamma[idx, idx] = rand(rng, 0:1)
    end

    # gamma' and delta' are unconstrained on the lower triangular
    fill_tril(rng, gamma_p, n, symmetric = true)
    fill_tril(rng, delta_p, n)

    # off diagonal: gamma, delta must obey conditions C1-C5
    for row in 1:n, col in 1:row-1
        if hadamard[row] && hadamard[col]
            b = rand(rng, 0:1)
            gamma[row, col] = b
            gamma[col, row] = b
            # otherwise delta[row,col] must be zero by C4
            if perm[row] > perm[col]
                 delta[row, col] = rand(rng, 0:1)
            end
        elseif hadamard[row] && (!hadamard[col]) && perm[row] < perm[col]
            # C5 imposes delta[row, col] = 0 for h[row]=1, h[col]=0
            # if perm[row] > perm[col] then C2 imposes gamma[row,col] = 0
            b = rand(rng, 0:1)
            gamma[row, col] = b
            gamma[col, row] = b
        elseif (!hadamard[row]) && hadamard[col]
            delta[row, col] = rand(rng, 0:1)
            # not sure what condition imposes this
            if perm[row] > perm[col]
                 b = rand(rng, 0:1)
                 gamma[row, col] = b
                 gamma[col, row] = b
            end
        elseif (!hadamard[row]) && (!hadamard[col]) && perm[row] < perm[col]
            # C1 imposes gamma[row, col] = 0 for h[row]=h[col] = 0
            # if perm[row] > perm[col] then C3 imposes delta[row,col] = 0
            delta[row, col] = rand(rng, 0:1)
        end
    end

    # now construct the tableau representation for F(I, Gamma, Delta)
    mul!(prod, gamma, delta)
    mul!(prod_p, gamma_p, delta_p)
    inv_delta .= mod.(precise_inv(delta'), 2)
    inv_delta_p .= mod.(precise_inv(delta_p'), 2)
 
    # block matrix form
    F1 .= mod.(F1, 2)
    F2 .= mod.(F2, 2)
    gamma .= 0
    gamma_p .= 0

    # apply qubit permutation S to F2
    perm_inds = vcat(perm, perm .+ n)
    U = F2[perm_inds,:]
    
    # apply layer of hadamards
    lhs_inds = vcat(had_idxs, had_idxs .+ n)
    rhs_inds = vcat(had_idxs .+ n, had_idxs)
    U[lhs_inds, :] = U[rhs_inds, :]
 
    # apply F1
    xzs = mod.(F1 * U,2) .== 1
 
    # random Pauli matrix just amounts to phases on the stabilizer tableau
    phases = rand(rng, [0x0,0x2], 2 * n)
    return Destabilizer(Stabilizer(phases, xzs), noprocessing=true)
end
random_destabilizer(n::Int) =  random_destabilizer(GLOBAL_RNG, n)

"""A random Clifford operator generated by the Bravyi-Maslov Algorithm 2 from [bravyi2020hadamard](@cite)."""
random_clifford(rng::AbstractRNG, n::Int) = CliffordOperator(random_destabilizer(rng, n))
random_clifford(n::Int) = random_clifford(GLOBAL_RNG, n::Int)

"""A random Stabilizer tableau generated by the Bravyi-Maslov Algorithm 2 from [bravyi2020hadamard](@cite)."""
random_stabilizer(rng::AbstractRNG, n::Int) = copy(stabilizerview(random_destabilizer(rng, n))) # TODO be less wasteful: there is no point in creating the whole destabilizer and then just throwing it away
random_stabilizer(n::Int) = random_stabilizer(GLOBAL_RNG, n)
random_stabilizer(rng::AbstractRNG,r::Int,n::Int) = random_stabilizer(rng,n)[randperm(rng,n)[1:r]]
random_stabilizer(r::Int,n::Int) = random_stabilizer(GLOBAL_RNG,n)[randperm(GLOBAL_RNG,n)[1:r]]

function random_singlequbitop(rng::AbstractRNG,n::Int)
    xtox = [falses(n) for i in 1:n]
    ztox = [falses(n) for i in 1:n]
    xtoz = [falses(n) for i in 1:n]
    ztoz = [falses(n) for i in 1:n]
    for i in 1:n
        gate = rand(rng,1:6)
        if gate<5
            xtox[i][i] = true
            xtoz[i][i] = true
            ztox[i][i] = true
            ztoz[i][i] = true
            [xtox,ztox,xtoz,ztoz][gate][i][i] = false
        elseif gate==5
            xtox[i][i] = true
            ztoz[i][i] = true
        else
            xtoz[i][i] = true
            ztox[i][i] = true
        end
    end
    c = CliffordColumnForm(zeros(UInt8,n*2), n,
                         vcat((vcat(x2x.chunks,z2x.chunks)' for (x2x,z2x) in zip(xtox,ztox))...),
                         vcat((vcat(x2z.chunks,z2z.chunks)' for (x2z,z2z) in zip(xtoz,ztoz))...)
        )
end
random_singlequbitop(n::Int) = random_singlequbitop(GLOBAL_RNG,n)


"""Inverting a binary matrix: uses floating point for small matrices and Nemo for large matrices."""
function precise_inv(a)
    n = size(a,1)
    if n<200
        return inv(a)
    else
	    return nemo_inv(a,n)
    end
end

function nemo_inv(a, n)
    binaryring = Nemo.ResidueRing(Nemo.ZZ, 2) # TODO should I use GF(2) instead of ResidueRing(ZZ, 2)?
    M = Nemo.MatrixSpace(binaryring, n, n)
    inverted = inv(M(Matrix{Int}(a))) # Nemo is very picky about input data types
    return (x->x.data).(inverted)
end

"""Sample (h, S) from the distribution P_n(h, S) from Bravyi and Maslov Algorithm 1."""
function quantum_mallows(rng, n) # each one is benchmakred in benchmarks/quantum_mallows.jl
    if n<30 # TODO Do in a prettier way without repetition.
        quantum_mallows_int(rng, n)
    elseif n<500
        quantum_mallows_float(rng, n)
    else
        quantum_mallows_bigint(rng, n)
    end
end

function quantum_mallows_int(rng, n)
    arr = collect(1:n)
    hadamard = falses(n)
    perm = zeros(Int64, n)
    for idx in 1:n
        m = length(arr)
        # sample h_i from given prob distribution
        k = rand(rng, 1:(UInt(4)^m-1))
        l = Int64(ilog2(k))+1
        # can also be written as
        #k = rand(rng, 2:UInt(4)^m)
        #l = (ispow2(k) ? ilog2(k) : ilog2(k) + 1)
        weight = 2 * m - l
        hadamard[idx] = (weight < m)
        k = weight < m ? weight : 2*m - weight - 1
        perm[idx] = popat!(arr, k + 1)
    end
    return hadamard, perm
end

function quantum_mallows_float(rng, n)
    arr = collect(1:n)
    hadamard = falses(n)
    perm = zeros(Int64, n)
    for idx in 1:n
        m = length(arr)
        # sample h_i from given prob distribution
        k = rand(rng)*(4.0^m-1) + 1
        l = ceil(log2(k))
        weight = Int64(2 * m - l)
        hadamard[idx] = (weight < m)
        k = weight < m ? weight : 2*m - weight - 1
        perm[idx] = popat!(arr, k + 1)
    end
    return hadamard, perm
end

function quantum_mallows_bigint(rng, n)
    arr = collect(1:n)
    hadamard = falses(n)
    perm = zeros(Int64, n)
    for idx in 1:n
        m = length(arr)
        # sample h_i from given prob distribution
        k = rand(rng, 2:BigInt(4)^m)
        l = Int64(ispow2(k) ? ilog2(k) : ilog2(k) + 1)
        # TODO This should be faster, but it is not:
        #k = rand(rng, 1:(BigInt(4)^m-1))
        #l = Int64(ilog2(k))+1
        # To compare to float implementations:
        # function f1(r,m) k = r*(4.0^m-1) + 1; l = ceil(log2(k)) end
        # function function f2(k,m) l = (ispow2(k) ? ilog2(k) : ilog2(k) + 1) end
        # m = 3
        # rs = 0:0.0025:0.999999; plot(rs, f1.(rs, m))
        # rs = 2:4^m; plot!((rs.-1)./(maximum(rs)-1), f2.(rs, 1),line=false,marker=true,legend=false)
        weight = Int64(2 * m - l)
        hadamard[idx] = (weight < m)
        k = weight < m ? weight : 2*m - weight - 1
        perm[idx] = popat!(arr, k + 1)
    end
    return hadamard, perm
end

"""Assign (symmetric) random ints to off diagonals of matrix."""
function fill_tril(rng, matrix, n; symmetric::Bool=false)
    # Add (symmetric) random ints to off diagonals
    for row in 1:n, col in 1:row-1
        b = rand(rng, 0:1)
        matrix[row, col] = b
        if symmetric
            matrix[col, row] = b
        end
    end
    matrix
end
