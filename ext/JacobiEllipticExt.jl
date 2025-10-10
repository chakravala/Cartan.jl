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

JacobiElliptic.CarlsonAlg.ellipke(x::TensorField) = TensorField(base(x), JacobiElliptic.CarlsonAlg.ellipke.(fiber(x)))
JacobiElliptic.CarlsonAlg.ellipke(x::LocalTensor) = LocalTensor(base(x), JacobiElliptic.CarlsonAlg.ellipke(fiber(x)))
for fun ∈ (:K,:E)
    @eval begin
        JacobiElliptic.$fun(x::TensorField) = TensorField(base(x), JacobiElliptic.$fun.(fiber(x)))
        JacobiElliptic.$fun(x::LocalTensor) = LocalTensor(base(x), JacobiElliptic.$fun(fiber(x)))
    end
end
JacobiElliptic.FukushimaAlg.J(x,y::TensorField) = TensorField(base(y), JacobiElliptic.FukushimaAlg.J.(x,fiber(y)))
JacobiElliptic.FukushimaAlg.J(x,y::LocalTensor) = LocalTensor(base(y), JacobiElliptic.FukushimaAlg.J(x,fiber(y)))
JacobiElliptic.FukushimaAlg.J(x::TensorField,y) = TensorField(base(x), JacobiElliptic.FukushimaAlg.J.(fiber(x),y))
JacobiElliptic.FukushimaAlg.J(x::LocalTensor,y) = LocalTensor(base(x), JacobiElliptic.FukushimaAlg.J(fiber(x),y))
JacobiElliptic.FukushimaAlg.J(x::TensorField,y::TensorField) = TensorField(base(x), JacobiElliptic.FukushimaAlg.J.(fiber(x),fiber(y)))
JacobiElliptic.FukushimaAlg.J(x::LocalTensor,y::LocalTensor) = LocalTensor(base(x), JacobiElliptic.FukushimaAlg.J(fiber(x),fiber(y)))
for fun ∈ (:F,:E,:Pi,:am,:sn,:cn,:dn,:sd,:cd,:nd,:dc,:sc,:nc,:ns,:ds,:cs,:ss,:dd,:cc,:nn,:asn) #:acn
    @eval begin
        JacobiElliptic.$fun(x,y::TensorField) = TensorField(base(y), JacobiElliptic.$fun.(x,fiber(y)))
        JacobiElliptic.$fun(x,y::LocalTensor) = LocalTensor(base(y), JacobiElliptic.$fun(x,fiber(y)))
        JacobiElliptic.$fun(x::TensorField,y) = TensorField(base(x), JacobiElliptic.$fun.(fiber(x),y))
        JacobiElliptic.$fun(x::LocalTensor,y) = LocalTensor(base(x), JacobiElliptic.$fun(fiber(x),y))
        JacobiElliptic.$fun(x::TensorField,y::TensorField) = TensorField(base(x), JacobiElliptic.$fun.(fiber(x),fiber(y)))
        JacobiElliptic.$fun(x::LocalTensor,y::LocalTensor) = LocalTensor(base(x), JacobiElliptic.$fun(fiber(x),fiber(y)))
    end
end
JacobiElliptic.FukushimaAlg.J(x,y,z::TensorField) = TensorField(base(z), JacobiElliptic.FukushimaAlg.J.(x,y,fiber(z)))
JacobiElliptic.FukushimaAlg.J(x,y,z::LocalTensor) = LocalTensor(base(z), JacobiElliptic.FukushimaAlg.J(x,y,fiber(z)))
JacobiElliptic.FukushimaAlg.J(x,y::TensorField,z) = TensorField(base(y), JacobiElliptic.FukushimaAlg.J.(x,fiber(y),z))
JacobiElliptic.FukushimaAlg.J(x,y::LocalTensor,z) = LocalTensor(base(y), JacobiElliptic.FukushimaAlg.J(x,fiber(y),z))
JacobiElliptic.FukushimaAlg.J(x::TensorField,y,z) = TensorField(base(x), JacobiElliptic.FukushimaAlg.J.(fiber(x),y,z))
JacobiElliptic.FukushimaAlg.J(x::LocalTensor,y,z) = LocalTensor(base(x), JacobiElliptic.FukushimaAlg.J(fiber(x),y,z))
JacobiElliptic.FukushimaAlg.J(x,y::TensorField,z::TensorField) = TensorField(base(y), JacobiElliptic.FukushimaAlg.J(x,fiber(y),fiber(z)))
JacobiElliptic.FukushimaAlg.J(x,y::LocalTensor,z::LocalTensor) = LocalTensor(base(y), JacobiElliptic.FukushimaAlg.J(x,fiber(y),fiber(z)))
JacobiElliptic.FukushimaAlg.J(x::TensorField,y,z::TensorField) = TensorField(base(x), JacobiElliptic.FukushimaAlg.J.(fiber(x),y,fiber(z)))
JacobiElliptic.FukushimaAlg.J(x::LocalTensor,y,z::LocalTensor) = LocalTensor(base(x), JacobiElliptic.FukushimaAlg.J(fiber(x),y,fiber(z)))
JacobiElliptic.FukushimaAlg.J(x::TensorField,y::TensorField,z) = TensorField(base(x), JacobiElliptic.FukushimaAlg.J.(fiber(x),fiber(y),z))
JacobiElliptic.FukushimaAlg.J(x::LocalTensor,y::LocalTensor,z) = LocalTensor(base(x), JacobiElliptic.FukushimaAlg.J(fiber(x),fiber(y),z))
JacobiElliptic.FukushimaAlg.J(x::TensorField,y::TensorField,z::TensorField) = TensorField(base(x), JacobiElliptic.FukushimaAlg.J.(fiber(x),fiber(y),fiber(z)))
JacobiElliptic.FukushimaAlg.J(x::LocalTensor,y::LocalTensor,z::LocalTensor) = LocalTensor(base(x), JacobiElliptic.FukushimaAlg.J(fiber(x),fiber(y),fiber(z)))
for fun ∈ (:Pi,)
    @eval begin
        JacobiElliptic.$fun(x,y,z::TensorField) = TensorField(base(z), JacobiElliptic.$fun.(x,y,fiber(z)))
        JacobiElliptic.$fun(x,y,z::LocalTensor) = LocalTensor(base(z), JacobiElliptic.$fun(x,y,fiber(z)))
        JacobiElliptic.$fun(x,y::TensorField,z) = TensorField(base(y), JacobiElliptic.$fun.(x,fiber(y),z))
        JacobiElliptic.$fun(x,y::LocalTensor,z) = LocalTensor(base(y), JacobiElliptic.$fun(x,fiber(y),z))
        JacobiElliptic.$fun(x::TensorField,y,z) = TensorField(base(x), JacobiElliptic.$fun.(fiber(x),y,z))
        JacobiElliptic.$fun(x::LocalTensor,y,z) = LocalTensor(base(x), JacobiElliptic.$fun(fiber(x),y,z))
        JacobiElliptic.$fun(x,y::TensorField,z::TensorField) = TensorField(base(y), JacobiElliptic.$fun.(x,fiber(y),fiber(z)))
        JacobiElliptic.$fun(x,y::LocalTensor,z::LocalTensor) = LocalTensor(base(y), JacobiElliptic.$fun(x,fiber(y),fiber(z)))
        JacobiElliptic.$fun(x::TensorField,y,z::TensorField) = TensorField(base(x), JacobiElliptic.$fun.(fiber(x),y,fiber(z)))
        JacobiElliptic.$fun(x::LocalTensor,y,z::LocalTensor) = LocalTensor(base(x), JacobiElliptic.$fun(fiber(x),y,fiber(z)))
        JacobiElliptic.$fun(x::TensorField,y::TensorField,z) = TensorField(base(x), JacobiElliptic.$fun.(fiber(x),fiber(y),z))
        JacobiElliptic.$fun(x::LocalTensor,y::LocalTensor,z) = LocalTensor(base(x), JacobiElliptic.$fun(fiber(x),fiber(y),z))
        JacobiElliptic.$fun(x::TensorField,y::TensorField,z::TensorField) = TensorField(base(x), JacobiElliptic.$fun.(fiber(x),fiber(y),fiber(z)))
        JacobiElliptic.$fun(x::LocalTensor,y::LocalTensor,z::LocalTensor) = LocalTensor(base(x), JacobiElliptic.$fun(fiber(x),fiber(y),fiber(z)))
    end
end

end # module
