"""
A module for simulating noisy Clifford circuits.
"""
module NoisyCircuits

# TODO the current interfaces of this module are a bit clunky when it comes to random events
# - applyop! returns (new_state, confirmation_of_no_detected_errors)
# - two sources of randomness are not tracked well currently
#   - which noise operator is applied (applied in a purely MC fashion)
#   - when a measurement is not commuting, which branch is taken (purely MC for now as well)
#
# TODO should we call things `apply!` instead of applyop and applynoise?
#
# TODO how important it is to distinguish measuring X₁ and then X₂ from measuring X₁X₂ when doing coincidence measurements

using QuantumClifford

using StatsBase: countmap

export Operation, AbstractGate, AbstractBellMeasurement, AbstractNoise,
       UnbiasedUncorrelatedNoise, NoiseOp, NoiseOpAll, VerifyOp,
       SparseGate, NoisyGate,
       BellMeasurement, NoisyBellMeasurement, BellMeasurementAndReset, BellMeasurementAndNoisyReset,
       affectedqubits, applyop!, applynoise!,
       mctrajectory!, mctrajectories,
       petrajectory, petrajectories

abstract type Operation end
abstract type AbstractGate <: Operation end
abstract type AbstractBellMeasurement <: Operation end

abstract type AbstractNoise end

"""Depolarization noise model with total probability of error `3*errprobthird`."""
struct UnbiasedUncorrelatedNoise{T} <: AbstractNoise
    errprobthird::T
end

"""An operator that applies the given `noise` model to the qubits at the selected `indices`."""
struct NoiseOp <: Operation
    noise::AbstractNoise
    indices::AbstractVector{Int}
end

"""An operator that applies the given `noise` model to all qubits."""
struct NoiseOpAll <: Operation
    noise::AbstractNoise
end

"""A Clifford gate, applying the given `cliff` operator to the qubits at the selected `indices`."""
struct SparseGate <: AbstractGate
    cliff::CliffordOperator # TODO do not hardcode this type of clifford op
    indices::AbstractVector{Int}
end

"""A gate consisting of the given `noise` applied after the given perfect Clifford `gate`."""
struct NoisyGate <: AbstractGate
    gate::AbstractGate # TODO should the type be more specific
    noise::AbstractNoise # TODO should the type be more specific
end

"""A Bell measurement performing the correlation measurement corresponding to the given `paulis` projections on the qubits at the selected indices."""
struct BellMeasurement <: AbstractBellMeasurement
    pauli::AbstractVector{PauliOperator}
    indices::AbstractVector{Int}
end

"""A perfect Bell measurement performed after the application of the given `noise` on the measured qubits."""
struct NoisyBellMeasurement <: AbstractBellMeasurement
    meas::AbstractBellMeasurement # TODO should the type be more specific
    noise::AbstractNoise # TODO should the type be more specific
end

"""Performing a Bell measurement followed by resetting the measured qubits to the given state `resetto`."""
struct BellMeasurementAndReset <: AbstractBellMeasurement # TODO Do we need a new type or should all non-terminal measurements implicitly have a reset?
    meas::AbstractBellMeasurement # TODO is this the cleanest way to specify the type
    resetto::Stabilizer
end

"""Performing a Bell measurement followed by resetting the measured qubits to the given state `resetto` followed by the same qubits being affected by the given `noise`."""
struct BellMeasurementAndNoisyReset <: AbstractBellMeasurement # TODO Do we need a new type or should all non-terminal measurements implicitly have a reset?
    meas::AbstractBellMeasurement # TODO is this the cleanest way to specify the type
    resetto::Stabilizer
    noise::AbstractNoise # TODO should the type be more specific
end

"""A "probe" to verify that the state of the qubits corresponds to a desired `good_state`, e.g. at the end of the execution of a circuit."""
struct VerifyOp <: Operation
    good_state::Stabilizer
    indices::AbstractVector{Int}
end

"""A dictionary of possible statuses returned by `applyop!`."""
const statuses = Dict(0=>:continue, 1=>:detected_failure, 2=>:undetected_failure, 3=>:true_success)
const s_continue = 0
const s_detected_failure = 1
const s_undetected_failure = 2
const s_true_success = 3

"""A method giving the qubits acted upon by a given operation. Part of the Noise interface."""
function affectedqubits end
affectedqubits(g::NoisyGate) = affectedqubits(g.gate)
affectedqubits(g::SparseGate) = g.indices
affectedqubits(m::BellMeasurement) = m.indices
affectedqubits(m::BellMeasurementAndReset) = affectedqubits(m.meas)
affectedqubits(m::BellMeasurementAndNoisyReset) = affectedqubits(m.meas)
affectedqubits(m::NoisyBellMeasurement) = affectedqubits(m.meas)
affectedqubits(n::NoiseOp) = n.indices
affectedqubits(v::VerifyOp) = v.indices

"""A method modifying a given state by applying the given operation. Non-deterministic, part of the Monte Carlo interface."""
function applyop! end

function applyop!(s::Stabilizer, g::NoisyGate)
    s = applynoise!(
            applyop!(s,g.gate)[1],
            g.noise,
            affectedqubits(g.gate)),
    return s, s_continue
end

applyop!(s::Stabilizer, g::SparseGate) = (apply!(s,g.cliff,affectedqubits(g)), s_continue)

applyop!(s::Stabilizer, m::NoisyBellMeasurement) =  applyop!(
    applynoise!(s,m.noise,affectedqubits(m)),
    m.meas)

# TODO this seems unnecessarily complicated
function applyop!(s::Stabilizer, m::BellMeasurement) # TODO is it ok to just measure XX instead of measuring XI and IX separately? That would be much faster
    n = nqubits(s)
    indices = affectedqubits(m)
    res = 0x00
    for (pauli, index) in zip(m.pauli,affectedqubits(m))
        if pauli==X # TODO this is not an elegant way to choose between X and Z coincidence measurements
            op = single_x(n,index) # TODO this is pretty terribly inefficient... use some sparse check
        else
            op = single_z(n,index)
        end # TODO permit Y operators and permit negative operators
        s,anticom,r = project!(s,op)
        if isnothing(r)
            if rand()>0.5 # TODO this seems stupid, float not necessary
                r = s.phases[anticom] = 0x00
            else
                r = s.phases[anticom] = 0x02
            end
        end
        res ⊻= r
    end
    if res==0x0
        return s, s_continue
    else
        return s, s_detected_failure
    end
end

function applyop!(s::Stabilizer, mr::BellMeasurementAndReset)
    s,res = applyop!(s,mr.meas)
    if !res
        return s,res
    else
        # TODO is the traceout necessary given that we just performed measurements?
        traceout!(s,mr.meas.indices)# TODO it seems like a bad idea not to keep track of the rank here
        n = nqubits(s) # TODO implement lastindex so we can just use end
        for (ii,i) in enumerate(affectedqubits(mr))
            for j in [1,2]
                s[n-j+1,i] = mr.resetto[j,ii]
            end
        end
        return s,s_continue
    end
end

function applyop!(s::Stabilizer, mr::BellMeasurementAndNoisyReset)
    s,res = applyop!(s,mr.meas)
    if !res
        return s,res
    else
        # TODO is the traceout necessary given that we just performed measurements?
        traceout!(s,affectedqubits(mr))# TODO it seems like a bad idea not to keep track of the rank here
        n = nqubits(s) # TODO implement lastindex so we can just use end
        for (ii,i) in enumerate(affectedqubits(mr))
            for j in [1,2]
                s[n-j+1,i] = mr.resetto[j,ii]
            end
        end
        return applynoise!(s,mr.noise,affectedqubits(mr)), s_continue
    end
end

"""A method modifying a given state by applying the corresponding noise model. Non-deterministic, part of the Noise interface."""
function applynoise! end

function applynoise!(s::Stabilizer,noise::UnbiasedUncorrelatedNoise,indices::AbstractVector{Int})
    n = nqubits(s)
    infid = noise.errprobthird
    for i in indices
        r = rand()
        if r<infid
            apply!(s,single_x(n,i)) # TODO stupidly inefficient, do it sparsely
        end
        if infid<=r<2infid
            apply!(s,single_z(n,i)) # TODO stupidly inefficient, do it sparsely
        end
        if 2infid<=r<3infid
            apply!(s,single_x(n,i)) # TODO stupidly inefficient, do it sparsely
            apply!(s,single_z(n,i)) # TODO stupidly inefficient, do it sparsely
        end
    end
    s
end

function applyop!(s::Stabilizer, mr::NoiseOpAll)
    n = nqubits(s)
    return applynoise!(s, mr.noise, 1:n), s_continue
end

function applyop!(s::Stabilizer, mr::NoiseOp)
    return applynoise!(s, mr.noise, affectedqubits(mr)), s_continue
end

# TODO this one needs more testing
function applyop!(s::Stabilizer, v::VerifyOp) # XXX It assumes the other qubits are measured or traced out
    # TODO QuantumClifford should implement some submatrix comparison
    n = nqubits(s) #  TODO QuantumClifford: implement lastindex(s)
    m = nqubits(v.good_state)
    s, _ = canonicalize_rref!(s,v.indices)
    for i in 1:m
        (s.phases[n-i+1]==v.good_state.phases[m-i+1]) || return s, s_undetected_failure
        for (j,q) in zip(1:m,v.indices)
            (s[n-i+1,q]==v.good_state[m-i+1,j]) || return s, s_undetected_failure
        end
    end
    return s, s_true_success
end

"""Run a single Monte Carlo sample, starting with (and modifying) `initialstate` by applying the given `circuit`. Uses `applyop!` under the hood."""
function mctrajectory!(initialstate::Stabilizer,circuit::AbstractVector{Operation})
    state = initialstate
    for op in circuit
        #println(typeof(op))
        state, cont = applyop!(state, op)
        #println("#",typeof(state))
        if cont!=s_continue
            return state, cont
        end
    end
    return state, s_continue
end

"""Run multiple Monte Carlo trajectories and report the aggregate final statuses of each."""
function mctrajectories(initialstate::Stabilizer,circuit::AbstractVector{Operation};trajectories=500)
    counts = countmap([mctrajectory!(copy(initialstate),circuit)[2] for i in 1:trajectories]) # TODO use threads or at least a generator
    return merge(Dict([(v=>0) for v in values(statuses)]),
                 Dict([statuses[k]=>v for (k,v) in counts]))
end

"""Compute all possible new states after the application of the given operator. Reports the probability of each one of them. Deterministic, part of the Perturbative Expansion interface."""
function applyop_branches end

applyop_branches(s::Stabilizer, g::SparseGate; max_order=1) = [(applyop!(copy(s),g)...,1,0)] # there are no fall backs on purpose, otherwise it is easy to mistakenly make a non-deterministic version of this method
applyop_branches(s::Stabilizer, v::VerifyOp; max_order=1) = [(applyop!(copy(s),v)...,1,0)] 

"""Compute all possible new states after the application of the given noise model. Reports the probability of each one of them. Deterministic, part of the Noise interface."""
function applynoise_branches end

function applynoise_branches(s::Stabilizer,noise::UnbiasedUncorrelatedNoise,indices::AbstractVector{Int}; max_order=1)
    n = nqubits(s)
    l = length(indices)
    infid = noise.errprobthird
    if l==0
        return [s,one(infid)]
    end
    no_error1 = 1-3*infid
    no_error = no_error1^l
    single_error = no_error1^(l-1)*infid
    results = [(copy(s),no_error,0)] # state, prob, order
    if max_order==0
        return results
    end
    for i in indices # TODO max_order>1 is not currently implemented
        push!(results,(apply!(copy(s),single_x(n,i)), single_error, 1)) # TODO stupidly inefficient, do it sparsely
        push!(results,(apply!(copy(s),single_z(n,i)), single_error, 1)) # TODO stupidly inefficient, do it sparsely
        push!(results,(apply!(apply!(copy(s),single_x(n,i)),single_z(n,i)), single_error, 1)) # TODO stupidly inefficient, do it sparsely
    end
    results
end

function applyop_branches(s::Stabilizer, nop::NoiseOpAll; max_order=1)
    n = nqubits(s)
    return [(state, s_continue, prob, order) for (state, prob, order) in applynoise_branches(s, nop.noise, 1:n, max_order=max_order)]
end

function applyop_branches(s::Stabilizer, nop::NoiseOp; max_order=1)
    return [(state, s_continue, prob, order) for (state, prob, order) in applynoise_branches(s, nop.noise, affectedqubits(nop), max_order=max_order)]
end

function applyop_branches(s::Stabilizer, g::NoisyGate; max_order=1)
    news, _,_,_ = applyop_branches(s,g.gate,max_order=max_order)[1] # TODO this assumes only one always successful branch for the gate
    return [(state, s_continue, prob, order) for (state, prob, order) in applynoise_branches(news, g.noise, affectedqubits(g), max_order=max_order)]
end

# TODO this can be much faster if we perform the flip on the classical bit after measurement, when possible
function applyop_branches(s::Stabilizer, m::NoisyBellMeasurement; max_order=1)
    return [(state, success, nprob*mprob, order)
            for (mstate, success, mprob, morder) in applyop_branches(s, m.meas, max_order=max_order)
            for (state, nprob, order) in applynoise_branches(mstate, m.noise, affectedqubits(m), max_order=max_order-morder)]
end

# TODO a lot of repetition with applyop!
function applyop_branches(s::Stabilizer, m::BellMeasurement; max_order=1) # TODO is it ok to just measure XX instead of measuring XI and IX separately? That would be much faster
    n = nqubits(s)
    [(ns,iseven(r>>1) ? s_continue : s_detected_failure, p,0)
     for (ns,r,p) in _applyop_branches_measurement([(s,0x0,1.0)],m.pauli,affectedqubits(m),n)]
end

# TODO XXX THIS IS PARTICULARLY INEFFICIENT recurrent implementation
function _applyop_branches_measurement(branches, paulis, qubits, n)
    if length(paulis) == 0
        return branches
    end

    new_branches = []
    pauli = paulis[1]
    otherpaulis = paulis[2:end]
    index = qubits[1]
    otherqubits = qubits[2:end]
    if pauli==X # TODO this is not an elegant way to choose between X and Z coincidence measurements
        op = single_x(n,index) # TODO this is pretty terribly inefficient... use some sparse check
    else
        op = single_z(n,index)
    end # TODO permit Y operators and permit negative operators

    for (s,r0,p) in branches
        s,anticom,r = project!(s,op)
        if isnothing(r)
            s1 = s
            s2 = copy(s)
            r1 = s1.phases[anticom] = 0x00
            r2 = s2.phases[anticom] = 0x02
            push!(new_branches, (s1,r0+r1,p/2))
            push!(new_branches, (s2,r0+r2,p/2))
        else
            push!(new_branches, (s,r0+r,p))
        end
    end

    return _applyop_branches_measurement(new_branches, otherpaulis, otherqubits, n)
end

# TODO a lot of repetition with applyop!
function applyop_branches(s::Stabilizer, mr::BellMeasurementAndReset; max_order=1)
    branches = applyop_branches(s,mr.meas, max_order=max_order)
    s = branches[1][1] # relies on the order of the branches, does not reset the branch with success==false, assumes order=0
    branches = [(_reset!(s,affectedqubits(mr).mr.resetto),succ,prob,order) for (s,succ,prob,order) in branches]
    branches
end

# TODO a lot of repetition with applyop!
function applyop_branches(s::Stabilizer, mr::BellMeasurementAndNoisyReset; max_order=1)
    branches = applyop_branches(s,mr.meas)
    # TODO can skip the inner loop if succ=false
    noise_branches = [(state, succ, prob*mprob, order) 
                      for (ms, succ, mprob, prime_order) in branches
                      for (state, prob, order) in applynoise_branches(_reset!(ms,affectedqubits(mr),mr.resetto), mr.noise, affectedqubits(mr), max_order=max_order-prime_order)
                     ]
end

function _reset!(s, qubits, resetto)
    # TODO is the traceout necessary given that we just performed measurements?
    traceout!(s,qubits)# TODO it seems like a bad idea not to keep track of the rank here
    n = nqubits(s) # TODO implement lastindex so we can just use end
    for (ii,i) in enumerate(qubits)
        for j in [1,2]
            s[n-j+1,i] = resetto[j,ii]
        end
    end
    return s
end

"""Run a perturbative expansion to a given order. Uses applyop_branches under the hood."""
function petrajectory(state, circuit; branch_weight=1.0, current_order=0, max_order=1)
    next_op = circuit[1]
    rest_of_circuit = circuit[2:end]

    status_probs = zeros(typeof(branch_weight), length(statuses)-1)

    # applyop_all returns all branches of the noise model
    p = 0
    for (i,(newstate, status, prob, order)) in enumerate(applyop_branches(state, next_op, max_order=max_order-current_order))
        p+=prob
        if status==s_continue # TODO is the copy below necessary?
            out_probs = petrajectory(copy(newstate), rest_of_circuit,
                branch_weight=branch_weight*prob, current_order=current_order+order, max_order=max_order)
            status_probs .+= out_probs
        else
            status_probs[status] += prob*branch_weight
        end
    end

    return status_probs
end

"""Run a perturbative expansion to a given order. This is the main public fuction for the perturbative expansion approach."""
function petrajectories(state, circuit; branch_weight=1.0, max_order=1)
    status_probs = petrajectory(state, circuit; branch_weight=branch_weight, current_order=0, max_order=max_order)
    Dict([statuses[i]=>status_probs[i] for i in eachindex(status_probs)])
end

end
