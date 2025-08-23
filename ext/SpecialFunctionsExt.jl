module SpecialFunctionsExt

#   This file is part of Cartan.jl
#   It is licensed under the GPL license
#   Cartan Copyright (C) 2023 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com

using Grassmann, Cartan
isdefined(Cartan, :Requires) ? (import Cartan: SpecialFunctions) : (using SpecialFunctions)

for fun ∈ (:gamma,:loggamma,:logfactorial,:digamma,:invdigamma,:trigamma,:expinti,:expintx,:sinint,:cosint,:erf,:erfc,:erfcinv,:erfcx,:logerfc,:logerfcx,:erfi,:erfinv,:dawson,:faddeeva,:airyai,:airyaiprime,:airybi,:airybiprime,:airyaix,:airyaiprimex,:airybix,:airybiprimex,:besselj0,:besselj1,:bessely0,:bessely1,:jinc,:ellipk,:ellipe,:eta,:zeta)
    @eval begin
        SpecialFunctions.$fun(t::TensorField) = TensorField(base(t), $fun.(fiber(t)))
        SpecialFunctions.$fun(t::LocalTensor) = LocalTensor(base(t), $fun(fiber(t)))
    end
end
for fun ∈ (:polygamma,:gamma,:loggamma,:besselj,:besseljx,:sphericalbesselj,:bessely,:besselyx,:sphericalbessely,:hankelh1,:hankelh1x,:hankelh2,:hankelh2x,:besseli,:besselix,:besselk,:besselkx)
    @eval begin
        SpecialFunctions.$fun(m,t::TensorField) = TensorField(base(t), $fun.(m,fiber(t)))
        SpecialFunctions.$fun(m,t::LocalTensor) = LocalTensor(base(t), $fun(m,fiber(t)))
    end
end
for fun ∈ (:gamma,:loggamma)
    @eval begin
        SpecialFunctions.$fun(a::TensorField,z) = TensorField(base(a), $fun.(fiber(a),z))
        SpecialFunctions.$fun(a::LocalTensor,z) = LocalTensor(base(a), $fun(fiber(a),z))
    end
end
for fun ∈ (:gamma,:loggamma,:beta,:logbeta,:logabsbeta,:logabsbinomial,:expint,:erf)
    @eval begin
        SpecialFunctions.$fun(x::TensorField,y::TensorField) = TensorField(base(x), $fun.(fiber(x),fiber(y)))
        SpecialFunctions.$fun(x::LocalTensor,y::LocalTensor) = LocalTensor(base(x), $fun(fiber(x),fiber(y)))
    end
end
SpecialFunctions.besselh(nu,k,t::TensorField) = TensorField(base(t), besselh.(nu,k,fiber(t)))
SpecialFunctions.besselh(nu,k,t::LocalTensor) = LocalTensor(base(t), besselh(nu,k,fiber(t)))

# :gamma_inc,:gamma_inc_inv,:beta_inc,:beta_inc_inv

end # module
