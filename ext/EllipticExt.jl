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
        Elliptic.$fun(x::TensorField) = TensorField(base(x), Elliptic.$fun.(fiber(x)))
        Elliptic.$fun(x::LocalTensor) = LocalTensor(base(x), Elliptic.$fun(fiber(x)))
    end
end
for fun ∈ (:F,:E,:ellipj)
    @eval begin
        Elliptic.$fun(x,y::TensorField) = TensorField(base(y), Elliptic.$fun.(x,fiber(y)))
        Elliptic.$fun(x,y::LocalTensor) = LocalTensor(base(y), Elliptic.$fun(x,fiber(y)))
        Elliptic.$fun(x::TensorField,y) = TensorField(base(x), Elliptic.$fun.(fiber(x),y))
        Elliptic.$fun(x::LocalTensor,y) = LocalTensor(base(x), Elliptic.$fun(fiber(x),y))
        Elliptic.$fun(x::TensorField,y::TensorField) = TensorField(base(x), Elliptic.$fun.(fiber(x),fiber(y)))
        Elliptic.$fun(x::LocalTensor,y::LocalTensor) = LocalTensor(base(x), Elliptic.$fun(fiber(x),fiber(y)))
    end
end
for fun ∈ (:Pi,)
    @eval begin
        Elliptic.$fun(x,y,z::TensorField) = TensorField(base(z), Elliptic.$fun.(x,y,fiber(z)))
        Elliptic.$fun(x,y,z::LocalTensor) = LocalTensor(base(z), Elliptic.$fun(x,y,fiber(z)))
        Elliptic.$fun(x,y::TensorField,z) = TensorField(base(y), Elliptic.$fun.(x,fiber(y),z))
        Elliptic.$fun(x,y::LocalTensor,z) = LocalTensor(base(y), Elliptic.$fun(x,fiber(y),z))
        Elliptic.$fun(x::TensorField,y,z) = TensorField(base(x), Elliptic.$fun.(fiber(x),y,z))
        Elliptic.$fun(x::LocalTensor,y,z) = LocalTensor(base(x), Elliptic.$fun(fiber(x),y,z))
        Elliptic.$fun(x,y::TensorField,z::TensorField) = TensorField(base(y), Elliptic.$fun.(x,fiber(y),fiber(z)))
        Elliptic.$fun(x,y::LocalTensor,z::LocalTensor) = LocalTensor(base(y), Elliptic.$fun(x,fiber(y),fiber(z)))
        Elliptic.$fun(x::TensorField,y,z::TensorField) = TensorField(base(x), Elliptic.$fun.(fiber(x),y,fiber(z)))
        Elliptic.$fun(x::LocalTensor,y,z::LocalTensor) = LocalTensor(base(x), Elliptic.$fun(fiber(x),y,fiber(z)))
        Elliptic.$fun(x::TensorField,y::TensorField,z) = TensorField(base(x), Elliptic.$fun.(fiber(x),fiber(y),z))
        Elliptic.$fun(x::LocalTensor,y::LocalTensor,z) = LocalTensor(base(x), Elliptic.$fun(fiber(x),fiber(y),z))
        Elliptic.$fun(x::TensorField,y::TensorField,z::TensorField) = TensorField(base(x), Elliptic.$fun.(fiber(x),fiber(y),fiber(z)))
        Elliptic.$fun(x::LocalTensor,y::LocalTensor,z::LocalTensor) = LocalTensor(base(x), Elliptic.$fun(fiber(x),fiber(y),fiber(z)))
    end
end
for fun ∈ (:am,:sn,:cn,:dn,:sd,:cd,:nd,:dc,:nc,:sc,:ns,:ds,:cs)
    @eval begin
        Elliptic.Jacobi.$fun(x,y::TensorField) = TensorField(base(y), Elliptic.Jacobi.$fun.(x,fiber(y)))
        Elliptic.Jacobi.$fun(x,y::LocalTensor) = LocalTensor(base(y), Elliptic.Jacobi.$fun(x,fiber(y)))
        Elliptic.Jacobi.$fun(x::TensorField,y) = TensorField(base(x), Elliptic.Jacobi.$fun.(fiber(x),y))
        Elliptic.Jacobi.$fun(x::LocalTensor,y) = LocalTensor(base(x), Elliptic.Jacobi.$fun(fiber(x),y))
        Elliptic.Jacobi.$fun(x::TensorField,y::TensorField) = TensorField(base(x), Elliptic.Jacobi.$fun.(fiber(x),fiber(y)))
        Elliptic.Jacobi.$fun(x::LocalTensor,y::LocalTensor) = LocalTensor(base(x), Elliptic.Jacobi.$fun(fiber(x),fiber(y)))
    end
end

end # module
