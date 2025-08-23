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
        EllipticFunctions.$fun(t::TensorField) = TensorField(base(t), $fun.(fiber(t)))
        EllipticFunctions.$fun(t::LocalTensor) = LocalTensor(base(t), $fun(fiber(t)))
    end
end
for fun ∈ (:ellipticE,:ellipticF,:ellipticZ)
    @eval begin
        EllipticFunctions.$fun(m,t::TensorField) = TensorField(base(t), $fun.(m,fiber(t)))
        EllipticFunctions.$fun(m,t::LocalTensor) = LocalTensor(base(t), $fun(m,fiber(t)))
    end
end
for fun ∈ (:ljtheta1,:jtheta1,:ljtheta2,:jtheta2,:ljtheta3,:jtheta3,:ljtheta4,:jtheta4,:jtheta1dash,:am)
    @eval begin
        EllipticFunctions.$fun(z::TensorField,q) = TensorField(base(z), $fun(fiber(z),q))
        EllipticFunctions.$fun(z::LocalTensor,q) = LocalTensor(base(z), $fun(fiber(z),q))
    end
end
EllipticFunctions.jtheta_ab(a,b,z::TensorField,q) = TensorField(base(z), jtheta_ab(a,b,fiber(z),q))
EllipticFunctions.jtheta_ab(a,b,z::LocalTensor,q) = LocalTensor(base(z), jtheta_ab(a,b,fiber(z),q))
EllipticFunctions.ellipticPI(nu,k,t::TensorField) = TensorField(base(t), ellipticPI.(nu,k,fiber(t)))
EllipticFunctions.ellipticPI(nu,k,t::LocalTensor) = LocalTensor(base(t), ellipticPI(nu,k,fiber(t)))
for fun ∈ (:wp,:wsigma,:wzeta,:thetaC,:thetaD,:thetaN,:thetaS)
    @eval begin
        EllipticFunctions.$fun(z::TensorField;args...) = TensorField(base(z), $fun(fiber(z);args...))
        EllipticFunctions.$fun(z::LocalTensor;args...) = LocalTensor(base(z), $fun(fiber(z);args...))
    end
end
EllipticFunctions.jellip(kind,u::TensorField;args...) = TensorField(base(u), jellip(kind,fiber(u);args...))
EllipticFunctions.jellip(kind,u::LocalTensor;args...) = LocalTensor(base(u), jellip(kind,fiber(u);args...))

end # module
