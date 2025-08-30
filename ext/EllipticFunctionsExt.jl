module EllipticFunctionsExt

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
isdefined(Cartan, :Requires) ? (import Cartan: EllipticFunctions) : (using EllipticFunctions)

for fun ∈ (:qfromtau,:taufromq,:etaDedekind,:lambda,:kleinj,:kleinjinv,:ellipticE,:ellipticK,:EisensteinE2,:EisensteinE4,:EisensteinE6)
    @eval begin
        EllipticFunctions.$fun(x::TensorField) = TensorField(base(x), EllipticFunctions.$fun.(fiber(x)))
        EllipticFunctions.$fun(x::LocalTensor) = LocalTensor(base(x), EllipticFunctions.$fun(fiber(x)))
    end
end
for fun ∈ (:ellipticE,:ellipticF,:ellipticZ,:jtheta1,:jtheta2,:jtheta3,:jtheta4,:jtheta1dash)
    @eval begin
        EllipticFunctions.$fun(x::Number,y::TensorField) = TensorField(base(y), EllipticFunctions.$fun.(x,fiber(y)))
        EllipticFunctions.$fun(x::Number,y::LocalTensor) = LocalTensor(base(y), EllipticFunctions.$fun(x,fiber(y)))
        EllipticFunctions.$fun(x::TensorField,y::Number) = TensorField(base(x), EllipticFunctions.$fun.(fiber(x),y))
        EllipticFunctions.$fun(x::LocalTensor,y::Number) = LocalTensor(base(x), EllipticFunctions.$fun(fiber(x),y))
        EllipticFunctions.$fun(x::TensorField,y::TensorField) = TensorField(base(x), EllipticFunctions.$fun.(fiber(x),fiber(y)))
        EllipticFunctions.$fun(x::LocalTensor,y::LocalTensor) = LocalTensor(base(x), EllipticFunctions.$fun(fiber(x),fiber(y)))
    end
end
for fun ∈ (:ljtheta1,:ljtheta2,:ljtheta3,:ljtheta4,:am)
    @eval begin
        EllipticFunctions.$fun(x::Number,y::TensorField) = TensorField(base(y), EllipticFunctions.$fun(x,fiber(y)))
        EllipticFunctions.$fun(x::Number,y::LocalTensor) = LocalTensor(base(y), EllipticFunctions.$fun(x,fiber(y)))
        EllipticFunctions.$fun(x::TensorField,y::Number) = TensorField(base(x), EllipticFunctions.$fun(fiber(x),y))
        EllipticFunctions.$fun(x::LocalTensor,y::Number) = LocalTensor(base(x), EllipticFunctions.$fun(fiber(x),y))
        EllipticFunctions.$fun(x::TensorField,y::TensorField) = TensorField(base(x), EllipticFunctions.$fun.(fiber(x),fiber(y)))
        EllipticFunctions.$fun(x::LocalTensor,y::LocalTensor) = LocalTensor(base(x), EllipticFunctions.$fun(fiber(x),fiber(y)))
    end
end
for fun ∈ (:CarlsonRD,:CarlsonRF,:CarlsonRG,:ellipticPI)
    @eval begin
        EllipticFunctions.$fun(x::Number,y::Number,z::TensorField) = TensorField(base(z), EllipticFunctions.$fun.(x,y,fiber(z)))
        EllipticFunctions.$fun(x::Number,y::Number,z::LocalTensor) = LocalTensor(base(z), EllipticFunctions.$fun(x,y,fiber(z)))
        EllipticFunctions.$fun(x::Number,y::TensorField,z::Number) = TensorField(base(y), EllipticFunctions.$fun.(x,fiber(y),z))
        EllipticFunctions.$fun(x::Number,y::LocalTensor,z::Number) = LocalTensor(base(y), EllipticFunctions.$fun(x,fiber(y),z))
        EllipticFunctions.$fun(x::TensorField,y::Number,z::Number) = TensorField(base(x), EllipticFunctions.$fun.(fiber(x),y,z))
        EllipticFunctions.$fun(x::LocalTensor,y::Number,z::Number) = LocalTensor(base(x), EllipticFunctions.$fun(fiber(x),y,z))
        EllipticFunctions.$fun(x::Number,y::TensorField,z::TensorField) = TensorField(base(y), EllipticFunctions.$fun.(x,fiber(y),fiber(z)))
        EllipticFunctions.$fun(x::Number,y::LocalTensor,z::LocalTensor) = LocalTensor(base(y), EllipticFunctions.$fun(x,fiber(y),fiber(z)))
        EllipticFunctions.$fun(x::TensorField,y::Number,z::TensorField) = TensorField(base(x), EllipticFunctions.$fun.(fiber(x),y,fiber(z)))
        EllipticFunctions.$fun(x::LocalTensor,y::Number,z::LocalTensor) = LocalTensor(base(x), EllipticFunctions.$fun(fiber(x),y,fiber(z)))
        EllipticFunctions.$fun(x::TensorField,y::TensorField,z::Number) = TensorField(base(x), EllipticFunctions.$fun.(fiber(x),fiber(y),z))
        EllipticFunctions.$fun(x::LocalTensor,y::LocalTensor,z::Number) = LocalTensor(base(x), EllipticFunctions.$fun(fiber(x),fiber(y),z))
        EllipticFunctions.$fun(x::TensorField,y::TensorField,z::TensorField) = TensorField(base(x), EllipticFunctions.$fun.(fiber(x),fiber(y),fiber(z)))
        EllipticFunctions.$fun(x::LocalTensor,y::LocalTensor,z::LocalTensor) = LocalTensor(base(x), EllipticFunctions.$fun(fiber(x),fiber(y),fiber(z)))
    end
end
for fun ∈ (:CarlsonRJ,:jtheta_ab)
    @eval begin
        EllipticFunctions.$fun(a::LocalTensor,b::LocalTensor,x::LocalTensor,y::LocalTensor) = LocalTensorField(base(a), EllipticFunctions.$fun(fiber(a),fiber(b),fiber(x),fiber(y)))
        EllipticFunctions.$fun(a::TensorField,b::TensorField,x::TensorField,y::TensorField) = TensorField(base(a), EllipticFunctions.$fun.(fiber(a),fiber(b),fiber(x),fiber(y)))
        EllipticFunctions.$fun(a::LocalTensor,b::LocalTensor,x::Number,y::Number) = LocalTensor(base(a), EllipticFunctions.$fun(fiber(a),fiber(b),x,y))
        EllipticFunctions.$fun(a::TensorField,b::TensorField,x::Number,y::Number) = TensorField(base(a), EllipticFunctions.$fun.(fiber(a),fiber(b),x,y))
        EllipticFunctions.$fun(a::LocalTensor,b::LocalTensor,x::LocalTensor,y::Number) = LocalTensor(base(a), EllipticFunctions.$fun(fiber(a),fiber(b),fiber(x),y))
        EllipticFunctions.$fun(a::TensorField,b::TensorField,x::TensorField,y::Number) = TensorField(base(a), EllipticFunctions.$fun.(fiber(a),fiber(b),fiber(x),y))
        EllipticFunctions.$fun(a::LocalTensor,b::LocalTensor,x::Number,y::LocalTensor) = LocalTensor(base(a), EllipticFunctions.$fun(fiber(a),fiber(b),x,fiber(y)))
        EllipticFunctions.$fun(a::TensorField,b::TensorField,x::Number,y::TensorField) = TensorField(base(a), EllipticFunctions.$fun.(fiber(a),fiber(b),x,fiber(y)))
        EllipticFunctions.$fun(a::Number,b::Number,x::LocalTensor,y::Number) = LocalTensor(base(x), EllipticFunctions.$fun(a,b,fiber(x),y))
        EllipticFunctions.$fun(a::Number,b::Number,x::Number,y::LocalTensor) = LocalTensor(base(y), EllipticFunctions.$fun(a,b,x,fiber(y)))
        EllipticFunctions.$fun(a::Number,b::Number,x::Number,y::TensorField) = TensorField(base(y), EllipticFunctions.$fun.(a,b,x,fiber(y)))
        EllipticFunctions.$fun(a::Number,b::Number,x::LocalTensor,y::LocalTensor) = LocalTensor(base(x), EllipticFunctions.$fun(a,b,fiber(x),fiber(y)))
        EllipticFunctions.$fun(a::Number,b::Number,x::TensorField,y::TensorField) = TensorField(base(x), EllipticFunctions.$fun.(a,b,fiber(x),fiber(y)))
    end
end
EllipticFunctions.CarlsonRJ(a::Number,b::Number,x::TensorField,y::Number) = TensorField(base(x), EllipticFunctions.CarlsonRJ.(a,b,fiber(x),y))
EllipticFunctions.jtheta_ab(a::Number,b::Number,x::TensorField,y::Number) = TensorField(base(x), EllipticFunctions.jtheta_ab.(a,b,fiber(x),y))
for fun ∈ (:wsigma,:wzeta,:thetaC,:thetaD,:thetaN,:thetaS)
    @eval begin
        EllipticFunctions.$fun(x::TensorField;args...) = TensorField(base(x), EllipticFunctions.$fun(fiber(x);args...))
        EllipticFunctions.$fun(x::LocalTensor;args...) = LocalTensor(base(x), EllipticFunctions.$fun(fiber(x);args...))
    end
end
EllipticFunctions.wp(x::TensorField;args...) = TensorField(base(x), EllipticFunctions.wp.(fiber(x);args...))
EllipticFunctions.wp(x::LocalTensor;args...) = LocalTensor(base(x), EllipticFunctions.wp(fiber(x);args...))
EllipticFunctions.jellip(kind::String,x::TensorField;args...) = TensorField(base(x), EllipticFunctions.jellip(kind,fiber(x);args...))
EllipticFunctions.jellip(kind::String,x::LocalTensor;args...) = LocalTensor(base(x), EllipticFunctions.jellip(kind,fiber(x);args...))

end # module
