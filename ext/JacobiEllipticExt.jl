module JacobiEllipticExt

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
isdefined(Cartan, :Requires) ? (import Cartan: JacobiElliptic) : (using JacobiElliptic)

JacobiElliptic.CarlsonAlg.ellipke(t::TensorField) = TensorField(base(t), CarlsonAlg.ellipke.(fiber(t)))
JacobiElliptic.CarlsonAlg.ellipke(t::LocalTensor) = LocalTensor(base(t), CarlsonAlg.ellipke(fiber(t)))
for fun ∈ (:K,:E)
    @eval begin
        JacobiElliptic.$fun(t::TensorField) = TensorField(base(t), $fun.(fiber(t)))
        JacobiElliptic.$fun(t::LocalTensor) = LocalTensor(base(t), $fun(fiber(t)))
    end
end
for fun ∈ (:F,:E,:Pi,:J,:am,:sn,:cn,:dn,:sd,:cd,:nd,:dc,:sc,:nc,:ns,:ds,:cs,:ss,:dd,:cc,:nn,:asn) #:acn
    @eval begin
        JacobiElliptic.$fun(m,t::TensorField) = TensorField(base(t), $fun.(m,fiber(t)))
        JacobiElliptic.$fun(m,t::LocalTensor) = LocalTensor(base(t), $fun(m,fiber(t)))
        JacobiElliptic.$fun(a::TensorField,z) = TensorField(base(a), $fun.(fiber(a),z))
        JacobiElliptic.$fun(a::LocalTensor,z) = LocalTensor(base(a), $fun(fiber(a),z))
        JacobiElliptic.$fun(x::TensorField,y::TensorField) = TensorField(base(x), $fun.(fiber(x),fiber(y)))
        JacobiElliptic.$fun(x::LocalTensor,y::LocalTensor) = LocalTensor(base(x), $fun(fiber(x),fiber(y)))
    end
end
for fun ∈ (:Pi,:J)
    @eval begin
        JacobiElliptic.$fun(n,m,t::TensorField) = TensorField(base(t), $fun.(n,m,fiber(t)))
        JacobiElliptic.$fun(n,m,t::LocalTensor) = LocalTensor(base(t), $fun(n,m,fiber(t)))
        JacobiElliptic.$fun(n,a::TensorField,z) = TensorField(base(a), $fun.(n,fiber(a),z))
        JacobiElliptic.$fun(n,a::LocalTensor,z) = LocalTensor(base(a), $fun(n,fiber(a),z))
        JacobiElliptic.$fun(n,x::TensorField,y::TensorField) = TensorField(base(x), $fun.(n,fiber(x),fiber(y)))
        JacobiElliptic.$fun(n,x::LocalTensor,y::LocalTensor) = LocalTensor(base(x), $fun(n,fiber(x),fiber(y)))
    end
end

end # module
