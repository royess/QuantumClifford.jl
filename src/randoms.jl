using Random: randperm, AbstractRNG, GLOBAL_RNG
using ILog2
import Nemo

##############################
# Random Paulis
##############################

"""A random Pauli operator on n qubits.

Use `realphase=true` to get operators with phase ±1 (excluding ±i).
`nophase=true` sets the phase to +1.

Optionally, a "flip" probability `p` can be provided specified,
in which case each bit is set to I with probability `1-p` and to
X, Y, or Z, each with probability `p`. Useful for simulating Pauli noise."""
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

"""A random Stabilizer/Destabilizer tableau generated by the Bravyi-Maslov Algorithm 2 from [bravyi2020hadamard](@cite).

`random_destabilizer(n)` gives a n-qubit tableau of rank `n`.
`random_destabilizer(r,n)` gives a n-qubit tableau of rank `r`."""
function random_destabilizer(rng::AbstractRNG, n::Int; phases::Bool=true)
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
        gamma_p[i,i] = rand(rng, 0x0:0x1)::UInt8
    end

    # gamma_ii is zero if h[i] = 0
    for idx in had_idxs
        gamma[idx, idx] = rand(rng, 0x0:0x1)::UInt8
    end

    # gamma' and delta' are unconstrained on the lower triangular
    fill_tril(rng, gamma_p, n, symmetric = true)
    fill_tril(rng, delta_p, n)

    # off diagonal: gamma, delta must obey conditions C1-C5
    for row in 1:n, col in 1:row-1
        if hadamard[row] && hadamard[col]
            gamma[row, col] = gamma[col, row] = rand(rng, 0x0:0x1)::UInt8
            # otherwise delta[row,col] must be zero by C4
            if perm[row] > perm[col]
                 delta[row, col] = rand(rng, 0x0:0x1)::UInt8
            end
        elseif hadamard[row] && (!hadamard[col]) && perm[row] < perm[col]
            # C5 imposes delta[row, col] = 0 for h[row]=1, h[col]=0
            # if perm[row] > perm[col] then C2 imposes gamma[row,col] = 0
            gamma[row, col] = gamma[col, row] = rand(rng, 0x0:0x1)::UInt8
        elseif (!hadamard[row]) && hadamard[col]
            delta[row, col] = rand(rng, 0x0:0x1)::UInt8
            # not sure what condition imposes this
            if perm[row] > perm[col]
                 gamma[row, col] = gamma[col, row] = rand(rng, 0x0:0x1)::UInt8
            end
        elseif (!hadamard[row]) && (!hadamard[col]) && perm[row] < perm[col]
            # C1 imposes gamma[row, col] = 0 for h[row]=h[col] = 0
            # if perm[row] > perm[col] then C3 imposes delta[row,col] = 0
            delta[row, col] = rand(rng, 0x0:0x1)::UInt8
        end
    end

    # now construct the tableau representation for F(I, Gamma, Delta)
    mul!(prod, gamma, delta)
    mul!(prod_p, gamma_p, delta_p)
    inv_delta .= precise_inv(delta')
    inv_delta_p .= precise_inv(delta_p')

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
    U[lhs_inds, :] .= U[rhs_inds, :]

    # apply F1
    xzs = mod.(F1 * U,2) .== 1

    # random Pauli matrix just amounts to phases on the stabilizer tableau
    phasesarray::Vector{UInt8} = if phases rand(rng, [0x0,0x2], 2n) else zeros(UInt8, 2n) end
    return Destabilizer(Tableau(phasesarray, xzs))
end
random_destabilizer(n::Int; phases::Bool=true) = random_destabilizer(GLOBAL_RNG, n; phases)
random_destabilizer(rng::AbstractRNG, r::Int, n::Int; phases::Bool=true) = MixedDestabilizer(random_destabilizer(rng,n;phases),r)
random_destabilizer(r::Int, n::Int; phases::Bool=true) = random_destabilizer(GLOBAL_RNG,r,n; phases)

"""A random Clifford operator generated by the Bravyi-Maslov Algorithm 2 from [bravyi2020hadamard](@cite)."""
random_clifford(rng::AbstractRNG, n::Int; phases::Bool=true) = CliffordOperator(random_destabilizer(rng, n; phases))
random_clifford(n::Int; phases::Bool=true) = random_clifford(GLOBAL_RNG, n::Int; phases)

"""A random Stabilizer tableau generated by the Bravyi-Maslov Algorithm 2 from [bravyi2020hadamard](@cite)."""
random_stabilizer(rng::AbstractRNG, n::Int; phases::Bool=true) = copy(stabilizerview(random_destabilizer(rng, n; phases))) # TODO be less wasteful: there is no point in creating the whole destabilizer and then just throwing it away
random_stabilizer(n::Int; phases::Bool=true) = random_stabilizer(GLOBAL_RNG, n; phases)
random_stabilizer(rng::AbstractRNG,r::Int,n::Int; phases::Bool=true) = random_stabilizer(rng,n; phases)[randperm(rng,n)[1:r]]
random_stabilizer(r::Int,n::Int; phases::Bool=true) = random_stabilizer(GLOBAL_RNG,n; phases)[randperm(GLOBAL_RNG,n)[1:r]]

"""Inverting a binary matrix: uses floating point for small matrices and Nemo for large matrices."""
function precise_inv(a)::Matrix{UInt8}
    n = size(a,1)
    if n<200
        return UInt8.(mod.(inv(a),0x2))
    else
	    return nemo_inv(a,n)
    end
end

function nemo_inv(a, n)::Matrix{UInt8}
    binaryring = Nemo.residue_ring(Nemo.ZZ, 2) # TODO should I use GF(2) instead of ResidueRing(ZZ, 2)?
    M = Nemo.MatrixSpace(binaryring, n, n)
    inverted = inv(M(Matrix{Int}(a))) # Nemo is very picky about input data types
    return (x->mod(UInt8(x.data),0x2)).(inverted)
end

"""Sample (h, S) from the distribution P_n(h, S) from Bravyi and Maslov Algorithm 1."""
function quantum_mallows(rng, n) # each one is benchmakred in benchmarks/quantum_mallows.jl
    arr = collect(1:n)
    hadamard = falses(n)
    perm = zeros(Int64, n)
    for idx in 1:n
        m = length(arr)
        # sample h_i from given prob distribution
        l = sample_geometric_2(rng, 2 * m)
        weight = 2 * m - l
        hadamard[idx] = (weight < m)
        k = weight < m ? weight : 2*m - weight - 1
        perm[idx] = popat!(arr, k + 1)
    end
    return hadamard, perm
end

""" This function samples a number from 1 to `n` where `n >= 1`
    probability of outputting `i` is proportional to `2^i`"""
function sample_geometric_2(rng, n::Integer)
    n < 1 && throw(DomainError(n))
    if n<30
        k = rand(rng, 2:UInt(2)^n)
        return ilog2(k, RoundUp)
    elseif n<500
        k = rand(rng)*(2.0^n-1) + 1
        return Int(ceil(log2(k)))
    else
        k = rand(rng, 2:BIG_INT_TWO[]^n)
        return ilog2(k, RoundUp)
    end
end

"""Assign (symmetric) random ints to off diagonals of matrix."""
function fill_tril(rng, matrix, n; symmetric::Bool=false)
    # Add (symmetric) random ints to off diagonals
    @inbounds for row in 1:n, col in 1:row-1
        b = rand(rng, Bool)
        matrix[row, col] = b
        if symmetric
            matrix[col, row] = b
        end
    end
    matrix
end
