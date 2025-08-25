module EllipticExt

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
isdefined(Cartan, :Requires) ? (import Cartan: Elliptic) : (using Elliptic)

for fun ∈ (:K,:E,:ellipke)
    @eval begin
        Elliptic.$fun(t::TensorField) = TensorField(base(t), Elliptic.$fun.(fiber(t)))
        Elliptic.$fun(t::LocalTensor) = LocalTensor(base(t), Elliptic.$fun(fiber(t)))
    end
end
for fun ∈ (:F,:E,:ellipj)
    @eval begin
        Elliptic.$fun(m,t::TensorField) = TensorField(base(t), Elliptic.$fun.(m,fiber(t)))
        Elliptic.$fun(m,t::LocalTensor) = LocalTensor(base(t), Elliptic.$fun(m,fiber(t)))
        Elliptic.$fun(a::TensorField,z) = TensorField(base(a), Elliptic.$fun.(fiber(a),z))
        Elliptic.$fun(a::LocalTensor,z) = LocalTensor(base(a), Elliptic.$fun(fiber(a),z))
        Elliptic.$fun(x::TensorField,y::TensorField) = TensorField(base(x), Elliptic.$fun.(fiber(x),fiber(y)))
        Elliptic.$fun(x::LocalTensor,y::LocalTensor) = LocalTensor(base(x), Elliptic.$fun(fiber(x),fiber(y)))
    end
end
for fun ∈ (:Pi,)
    @eval begin
        Elliptic.$fun(n,m,t::TensorField) = TensorField(base(t), Elliptic.$fun.(n,m,fiber(t)))
        Elliptic.$fun(n,m,t::LocalTensor) = LocalTensor(base(t), Elliptic.$fun(n,m,fiber(t)))
        Elliptic.$fun(n,a::TensorField,z) = TensorField(base(a), Elliptic.$fun.(n,fiber(a),z))
        Elliptic.$fun(n,a::LocalTensor,z) = LocalTensor(base(a), Elliptic.$fun(n,fiber(a),z))
        Elliptic.$fun(n,x::TensorField,y::TensorField) = TensorField(base(x), Elliptic.$fun.(n,fiber(x),fiber(y)))
        Elliptic.$fun(n,x::LocalTensor,y::LocalTensor) = LocalTensor(base(x), Elliptic.$fun(n,fiber(x),fiber(y)))
    end
end
for fun ∈ (:am,:sn,:cn,:dn,:sd,:cd,:nd,:dc,:nc,:sc,:ns,:ds,:cs)
    @eval begin
        Elliptic.Jacobi.$fun(m,t::TensorField) = TensorField(base(t), Elliptic.Jacobi.$fun.(m,fiber(t)))
        Elliptic.Jacobi.$fun(m,t::LocalTensor) = LocalTensor(base(t), Elliptic.Jacobi.$fun(m,fiber(t)))
        Elliptic.Jacobi.$fun(a::TensorField,z) = TensorField(base(a), Elliptic.Jacobi.$fun.(fiber(a),z))
        Elliptic.Jacobi.$fun(a::LocalTensor,z) = LocalTensor(base(a), Elliptic.Jacobi.$fun(fiber(a),z))
        Elliptic.Jacobi.$fun(x::TensorField,y::TensorField) = TensorField(base(x), Elliptic.Jacobi.$fun.(fiber(x),fiber(y)))
        Elliptic.Jacobi.$fun(x::LocalTensor,y::LocalTensor) = LocalTensor(base(x), Elliptic.Jacobi.$fun(fiber(x),fiber(y)))
    end
end

end # module
