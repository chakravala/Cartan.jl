module FewSpecialFunctionsExt

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
isdefined(Cartan, :Requires) ? (import Cartan: FewSpecialFunctions) : (using FewSpecialFunctions)

# U,V,MacrumQ,dQdb,debye_function

for fun ∈ (:η,:FresnelC,:FresnelS,:FresnelE,:Ci_complex)
    @eval begin
        FewSpecialFunctions.$fun(x::TensorField) = TensorField(base(x), FewSpecialFunctions.$fun.(fiber(x)))
        FewSpecialFunctions.$fun(x::LocalTensor) = LocalTensor(base(x), FewSpecialFunctions.$fun(fiber(x)))
    end
end
for fun ∈ (:C,:η,:U,:V,:FermiDiracIntegral,:FermiDiracIntegralNorm,:Clausen)
    @eval begin
        FewSpecialFunctions.$fun(x,y::TensorField) = TensorField(base(y), FewSpecialFunctions.$fun.(x,fiber(y)))
        FewSpecialFunctions.$fun(x,y::LocalTensor) = LocalTensor(base(y), FewSpecialFunctions.$fun(x,fiber(y)))
        FewSpecialFunctions.$fun(x::TensorField,y) = TensorField(base(x), FewSpecialFunctions.$fun.(fiber(x),y))
        FewSpecialFunctions.$fun(x::LocalTensor,y) = LocalTensor(base(x), FewSpecialFunctions.$fun(fiber(x),y))
        FewSpecialFunctions.$fun(x::TensorField,y::TensorField) = TensorField(base(x), FewSpecialFunctions.$fun.(fiber(x),fiber(y)))
        FewSpecialFunctions.$fun(x::LocalTensor,y::LocalTensor) = LocalTensor(base(x), FewSpecialFunctions.$fun(fiber(x),fiber(y)))
    end
end
for fun ∈ (:F,:G,:H⁺,:H⁻,:MarcumQ,:dQdb,:F_clausen,:f_n,:debye_function)
    @eval begin
        FewSpecialFunctions.$fun(x,y,z::TensorField) = TensorField(base(z), FewSpecialFunctions.$fun.(x,y,fiber(z)))
        FewSpecialFunctions.$fun(x,y,z::LocalTensor) = LocalTensor(base(z), FewSpecialFunctions.$fun(x,y,fiber(z)))
        FewSpecialFunctions.$fun(x,y::TensorField,z) = TensorField(base(y), FewSpecialFunctions.$fun.(x,fiber(y),z))
        FewSpecialFunctions.$fun(x,y::LocalTensor,z) = LocalTensor(base(y), FewSpecialFunctions.$fun(x,fiber(y),z))
        FewSpecialFunctions.$fun(x::TensorField,y,z::Number) = TensorField(base(x), FewSpecialFunctions.$fun.(fiber(x),y,z))
        FewSpecialFunctions.$fun(x::LocalTensor,y,z) = LocalTensor(base(x), FewSpecialFunctions.$fun(fiber(x),y,z))
        FewSpecialFunctions.$fun(x,y::TensorField,z::TensorField) = TensorField(base(y), FewSpecialFunctions.$fun.(x,fiber(y),fiber(z)))
        FewSpecialFunctions.$fun(x,y::LocalTensor,z::LocalTensor) = LocalTensor(base(y), FewSpecialFunctions.$fun(x,fiber(y),fiber(z)))
        FewSpecialFunctions.$fun(x::TensorField,y,z::TensorField) = TensorField(base(x), FewSpecialFunctions.$fun.(fiber(x),y,fiber(z)))
        FewSpecialFunctions.$fun(x::LocalTensor,y,z::LocalTensor) = LocalTensor(base(x), FewSpecialFunctions.$fun(fiber(x),y,fiber(z)))
        FewSpecialFunctions.$fun(x::TensorField,y::TensorField,z) = TensorField(base(x), FewSpecialFunctions.$fun.(fiber(x),fiber(y),z))
        FewSpecialFunctions.$fun(x::LocalTensor,y::LocalTensor,z) = LocalTensor(base(x), FewSpecialFunctions.$fun(fiber(x),fiber(y),z))
        FewSpecialFunctions.$fun(x::TensorField,y::TensorField,z::TensorField) = TensorField(base(x), FewSpecialFunctions.$fun.(fiber(x),fiber(y),fiber(z)))
        FewSpecialFunctions.$fun(x::LocalTensor,y::LocalTensor,z::LocalTensor) = LocalTensor(base(x), FewSpecialFunctions.$fun(fiber(x),fiber(y),fiber(z)))
    end
end

end # module
