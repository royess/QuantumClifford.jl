function LPCode(args...; kwargs...)
    ext = Base.get_extension(QuantumClifford, :QuantumCliffordHeckeExt)
    if isnothing(ext)
        throw("The `LPCode` depends on the package `AbstractAlgebra` and `Hecke` but you have not installed or imported them yet. Immediately after you import `AbstractAlgebra` and `Hecke`, the `LPCode` will be available.")
    end
    return ext.LPCode(args...; kwargs...)
end

function two_block_group_algebra_codes end

function generalized_bicycle_codes end

function bicycle_codes end
