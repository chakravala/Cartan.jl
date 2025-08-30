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
        SpecialFunctions.$fun(x::TensorField) = TensorField(base(x), SpecialFunctions.$fun.(fiber(x)))
        SpecialFunctions.$fun(x::LocalTensor) = LocalTensor(base(x), SpecialFunctions.$fun(fiber(x)))
    end
end
for fun ∈ (:polygamma,:gamma,:gamma_inc,:loggamma,:beta,:beta_inc,:beta_inc_inv,:logbeta,:logabsbeta,:logabsbinomial,:expint,:erf,:besselj,:besseljx,:sphericalbesselj,:bessely,:besselyx,:sphericalbessely,:hankelh1,:hankelh1x,:hankelh2,:hankelh2x,:besseli,:besselix,:besselk,:besselkx)
    @eval begin
        SpecialFunctions.$fun(x,y::TensorField) = TensorField(base(y), SpecialFunctions.$fun.(x,fiber(y)))
        SpecialFunctions.$fun(x,y::LocalTensor) = LocalTensor(base(y), SpecialFunctions.$fun(x,fiber(y)))
        SpecialFunctions.$fun(x::TensorField,y) = TensorField(base(x), SpecialFunctions.$fun.(fiber(x),y))
        SpecialFunctions.$fun(x::LocalTensor,y) = LocalTensor(base(x), SpecialFunctions.$fun(fiber(x),y))
        SpecialFunctions.$fun(x::TensorField,y::TensorField) = TensorField(base(x), SpecialFunctions.$fun.(fiber(x),fiber(y)))
        SpecialFunctions.$fun(x::LocalTensor,y::LocalTensor) = LocalTensor(base(x), SpecialFunctions.$fun(fiber(x),fiber(y)))
    end
end
SpecialFunctions.gamma_inc(x::TensorField,y,z) = TensorField(base(x), SpecialFunctions.gamma_inc.(fiber(x),y,z))
SpecialFunctions.gamma_inc(x::LocalTensor,y,z) = LocalTensor(base(x), SpecialFunctions.gamma_inc(fiber(x),y,z))
SpecialFunctions.gamma_inc(x,y::TensorField,z) = TensorField(base(y), SpecialFunctions.gamma_inc.(x,k,fiber(y)))
SpecialFunctions.gamma_inc(x,y::LocalTensor,z) = LocalTensor(base(y), SpecialFunctions.gamma_inc(x,k,fiber(y)))
SpecialFunctions.gamma_inc(x::TensorField,y::TensorField,z) = TensorField(base(x), SpecialFunctions.gamma_inc.(fiber(x),fiber(y),z))
SpecialFunctions.gamma_inc(x::LocalTensor,y::LocalTensor,z) = LocalTensor(base(x), SpecialFunctions.gamma_inc(fiber(x),fiber(y),z))
for fun ∈ (:gamma_inc_inv,:beta_inc,:beta_inc_inv)
    @eval begin
        SpecialFunctions.$fun(x,y,z::TensorField) = TensorField(base(z), SpecialFunctions.$fun.(x,y,fiber(z)))
        SpecialFunctions.$fun(x,y,z::LocalTensor) = LocalTensor(base(z), SpecialFunctions.$fun(x,y,fiber(z)))
        SpecialFunctions.$fun(x,y::TensorField,z) = TensorField(base(y), SpecialFunctions.$fun.(x,fiber(y),z))
        SpecialFunctions.$fun(x,y::LocalTensor,z) = LocalTensor(base(y), SpecialFunctions.$fun(x,fiber(y),z))
        SpecialFunctions.$fun(x::TensorField,y,z::Number) = TensorField(base(x), SpecialFunctions.$fun.(fiber(x),y,z))
        SpecialFunctions.$fun(x::LocalTensor,y,z) = LocalTensor(base(x), SpecialFunctions.$fun(fiber(x),y,z))
        SpecialFunctions.$fun(x,y::TensorField,z::TensorField) = TensorField(base(y), SpecialFunctions.$fun.(x,fiber(y),fiber(z)))
        SpecialFunctions.$fun(x,y::LocalTensor,z::LocalTensor) = LocalTensor(base(y), SpecialFunctions.$fun(x,fiber(y),fiber(z)))
        SpecialFunctions.$fun(x::TensorField,y,z::TensorField) = TensorField(base(x), SpecialFunctions.$fun.(fiber(x),y,fiber(z)))
        SpecialFunctions.$fun(x::LocalTensor,y,z::LocalTensor) = LocalTensor(base(x), SpecialFunctions.$fun(fiber(x),y,fiber(z)))
        SpecialFunctions.$fun(x::TensorField,y::TensorField,z) = TensorField(base(x), SpecialFunctions.$fun.(fiber(x),fiber(y),z))
        SpecialFunctions.$fun(x::LocalTensor,y::LocalTensor,z) = LocalTensor(base(x), SpecialFunctions.$fun(fiber(x),fiber(y),z))
        SpecialFunctions.$fun(x::TensorField,y::TensorField,z::TensorField) = TensorField(base(x), SpecialFunctions.$fun.(fiber(x),fiber(y),fiber(z)))
        SpecialFunctions.$fun(x::LocalTensor,y::LocalTensor,z::LocalTensor) = LocalTensor(base(x), SpecialFunctions.$fun(fiber(x),fiber(y),fiber(z)))
    end
end
for fun ∈ (:beta_inc,:beta_inc_inv)
    @eval begin
        SpecialFunctions.$fun(a::LocalTensor,b::LocalTensor,x::LocalTensor,y::LocalTensor) = LocalTensor(base(a), SpecialFunctions.$fun(fiber(a),fiber(b),fiber(x),fiber(y)))
        SpecialFunctions.$fun(a::TensorField,b::TensorField,x::TensorField,y::TensorField) = TensorField(base(a), SpecialFunctions.$fun.(fiber(a),fiber(b),fiber(x),fiber(y)))
        SpecialFunctions.$fun(a::LocalTensor,b::LocalTensor,x,y) = LocalTensor(base(a), SpecialFunctions.$fun(fiber(a),fiber(b),x,y))
        SpecialFunctions.$fun(a::TensorField,b::TensorField,x,y) = TensorField(base(a), SpecialFunctions.$fun.(fiber(a),fiber(b),x,y))
        SpecialFunctions.$fun(a::LocalTensor,b::LocalTensor,x::LocalTensor,y) = LocalTensor(base(a), SpecialFunctions.$fun(fiber(a),fiber(b),fiber(x),y))
        SpecialFunctions.$fun(a::TensorField,b::TensorField,x::TensorField,y) = TensorField(base(a), SpecialFunctions.$fun.(fiber(a),fiber(b),fiber(x),y))
        SpecialFunctions.$fun(a::LocalTensor,b::LocalTensor,x,y::LocalTensor) = LocalTensor(base(a), SpecialFunctions.$fun(fiber(a),fiber(b),x,fiber(y)))
        SpecialFunctions.$fun(a::TensorField,b::TensorField,x,y::TensorField) = TensorField(base(a), SpecialFunctions.$fun.(fiber(a),fiber(b),x,fiber(y)))
        SpecialFunctions.$fun(a::Number,b::Number,x::LocalTensor,y) = LocalTensor(base(x), SpecialFunctions.$fun(a,b,fiber(x),y))
        SpecialFunctions.$fun(a::Number,b::Number,x::TensorField,y) = TensorField(base(x), SpecialFunctions.$fun.(a,b,fiber(x),y))
        SpecialFunctions.$fun(a::Number,b::Number,x::Number,y::LocalTensor) = LocalTensor(base(y), SpecialFunctions.$fun(a,b,x,fiber(y)))
        SpecialFunctions.$fun(a::Number,b::Number,x::Number,y::TensorField) = TensorField(base(y), SpecialFunctions.$fun.(a,b,x,fiber(y)))
        SpecialFunctions.$fun(a::Number,b::Number,x::LocalTensor,y::LocalTensor) = LocalTensor(base(x), SpecialFunctions.$fun(a,b,fiber(x),fiber(y)))
        SpecialFunctions.$fun(a::Number,b::Number,x::TensorField,y::TensorField) = TensorField(base(x), SpecialFunctions.$fun.(a,b,fiber(x),fiber(y)))
    end
end
SpecialFunctions.besselh(x,k,y::TensorField) = TensorField(base(y), SpecialFunctions.besselh.(x,k,fiber(y)))
SpecialFunctions.besselh(x,k,y::LocalTensor) = LocalTensor(base(y), SpecialFunctions.besselh(x,k,fiber(y)))
SpecialFunctions.besselh(x::TensorField,k,y) = TensorField(base(x), SpecialFunctions.besselh.(fiber(x),k,y))
SpecialFunctions.besselh(x::LocalTensor,k,y) = LocalTensor(base(x), SpecialFunctions.besselh(fiber(x),k,y))
SpecialFunctions.besselh(x::TensorField,k,y::TensorField) = TensorField(base(x), SpecialFunctions.besselh.(fiber(x),k,fiber(y)))
SpecialFunctions.besselh(x::LocalTensor,k,y::LocalTensor) = LocalTensor(base(x), SpecialFunctions.besselh(fiber(x),k,fiber(y)))

end # module
