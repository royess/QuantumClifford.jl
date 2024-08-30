import Nemo: matrix_space
import LinearAlgebra

"""
Lifted product codes [panteleev2021degenerate](@cite) [panteleev2022asymptotically](@cite)

- `A::PermGroupRingMatrix`: the first base matrix for constructing the lifted product code, whose elements are in a permutation group ring;
- `B::PermGroupRingMatrix`: the second base matrix for constructing the lifted product code, whose elements are in the same permutation group ring as `A`;
- `repr::Function`: a function that converts the permutation group ring element to a matrix;
 default to be [`permutation_repr`](@ref) for GF(2)-algebra.

A lifted product code is constructed by hypergraph product of the two lifted codes `c₁` and `c₂`.
Here, the hypergraph product is taken over a group ring, which serves as the base ring for both lifted codes.
After the hypergraph product, the parity-check matrices are lifted by `repr`.
The lifting is achieved by applying `repr` to each element of the matrix resulted from the hypergraph product,
which is mathematically a linear map from a group algebra element to a binary matrix.

## Constructors

A lifted product code can be constructed via the following approaches:

1. From two matrices of permutation group elements.

2. From two matrices of permutations, where a permutation will be considered as a permutation group element by given a unit coefficient.

3. From two matrices of integers, where an integer represent the a cyclic permutation by its offset.

4. From two (classical) lifted codes. The base matrices of them will be used for product construction.

## Code instances

A [[882, 24, d ≤ 24]] code from https://arxiv.org/abs/2202.01702v3, following the 1st constructor.
During the construction, we do arithmetic operations to get the permutation group ring elements
with `x` being the offset-1 cyclic permutation and `R(1)` being the unit.

```jldoctest
julia> ENV["NEMO_PRINT_BANNER"] = "false"; using Nemo: GF, Perm; import LinearAlgebra;

julia> l = 63; R = PermutationGroupRing(GF(2), l); x = R(cyclic_permutation(1, l));

julia> A = zeros(R, 7, 7);

julia> A[LinearAlgebra.diagind(A)] .= x^27;

julia> A[LinearAlgebra.diagind(A, -1)] .= x^54;

julia> A[LinearAlgebra.diagind(A, 6)] .= x^54;

julia> A[LinearAlgebra.diagind(A, -2)] .= R(1);

julia> A[LinearAlgebra.diagind(A, 5)] .= R(1);

julia> B = reshape([1 + x + x^6], (1, 1));

julia> c2 = LPCode(A, B);

julia> code_n(c2), code_k(c2)
(882, 24)
```

A [[175, 19, d ≤ 0]] code from paper http://arxiv.org/abs/2111.07029, following the 3rd constructor.

```jldoctest
julia> base_matrix = [0 0 0 0; 0 1 2 5; 0 6 3 1]; l = 7;

julia> c1 = LPCode(base_matrix, l .- base_matrix', l);

julia> code_n(c1), code_k(c1)
(175, 19)
```

## Examples of code subfamilies

- When the base matrices of the `LPCode` are one-by-one, the code is called a two-block group-algebra code [`two_block_group_algebra_codes`](@ref).
- When the base matrices of the `LPCode` are one-by-one and their elements are cyclic permuatations, the code is called a generalized bicycle code [`generalized_bicycle_codes`](@ref).
- More specially, when the two matrices are adjoint to each other, the code is called a bicycle code [`bicycle_codes`](@ref).

See also: [`LiftedCode`](@ref), [`PermGroupRing`](@ref), [`two_block_group_algebra_codes`](@ref),
[`generalized_bicycle_codes`](@ref), [](@ref), [`bicycle_codes`](@ref)
"""
struct LPCode <: AbstractECC
    A::PermGroupRingMatrix
    B::PermGroupRingMatrix
    repr::Function

    function LPCode(A::PermGroupRingMatrix, B::PermGroupRingMatrix, repr::Function)
        A[1, 1].parent == B[1, 1].parent || error("The base rings of the two codes must be the same")
        new(A, B, repr)
    end

    function LPCode(c₁::LiftedCode, c₂::LiftedCode, repr::Function)
        c₁.A[1, 1].parent == c₂.A[1, 1].parent || error("The base rings of the two codes must be the same")
        new(c₁.A, c₂.A, repr)
    end
end

function LPCode(A::PermGroupRingMatrix, B::PermGroupRingMatrix)
    A[1, 1].parent == B[1, 1].parent || error("The base rings of the two codes must be the same")
    LPCode(A, B, permutation_repr)
end

function LPCode(perm_array1::Matrix{<:Perm}, perm_array2::Matrix{<:Perm})
    LPCode(LiftedCode(perm_array1), LiftedCode(perm_array2))
end

function LPCode(shift_array1::Matrix{Int}, shift_array2::Matrix{Int}, l::Int)
    LPCode(LiftedCode(shift_array1, l), LiftedCode(shift_array2, l))
end

function LPCode(c₁::LiftedCode, c₂::LiftedCode)
    c₁.A[1, 1].parent == c₂.A[1, 1].parent || error("The base rings of the two codes must be the same")
    LPCode(c₁, c₂, c₁.repr) # use the same `repr` as the first code
end

iscss(::Type{LPCode}) = true

function parity_checks_xz(c::LPCode)
    hx, hz = hgp(c.A, c.B')
    hx, hz = lift(c.repr, hx), lift(c.repr, hz)
    return hx, hz
end

parity_checks_x(c::LPCode) = parity_checks_xz(c)[1]

parity_checks_z(c::LPCode) = parity_checks_xz(c)[2]

parity_checks(c::LPCode) = parity_checks(CSS(parity_checks_xz(c)...))

code_n(c::LPCode) = size(c.repr(parent(c.A[1, 1])(0)), 2) * (size(c.A, 2) * size(c.B, 1) + size(c.A, 1) * size(c.B, 2))

code_s(c::LPCode) = size(c.repr(parent(c.A[1, 1])(0)), 1) * (size(c.A, 1) * size(c.B, 1) + size(c.A, 2) * size(c.B, 2))

"""
Two-block group algebra (2GBA) codes.
"""
function two_block_group_algebra_codes(a::PermGroupRingElem, b::PermGroupRingElem)
    A = reshape([a], (1, 1))
    B = reshape([b], (1, 1))
    LPCode(A, B)
end

"""
Generalized bicycle codes.

```jldoctest
julia> c = generalized_bicycle_codes([0, 15, 20, 28, 66], [0, 58, 59, 100, 121], 127);

julia> code_n(c), code_k(c)
(254, 28)
```
"""
function generalized_bicycle_codes(a_shifts::Array{Int}, b_shifts::Array{Int}, l::Int)
    R = PermutationGroupRing(GF(2), l)
    a = sum(R(cyclic_permutation(n, l)) for n in a_shifts)
    b = sum(R(cyclic_permutation(n, l)) for n in b_shifts)
    two_block_group_algebra_codes(a, b)
end

"""
Bicycle codes.
"""
function bicycle_codes(a_shifts::Array{Int}, l::Int)
    R = PermutationGroupRing(GF(2), l)
    a = sum(R(cyclic_permutation(n, l)) for n in a_shifts)
    two_block_group_algebra_codes(a, a')
end
