module ColorTypesExt

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
isdefined(Cartan, :Requires) ? (import Cartan: ColorTypes) : (using ColorTypes)

_rectangle(x) = TensorField(ProductSpace{2}(range(-3,3,100),range(-3,3,100)))

function Cartan.raster(ga::Vector,R=_rectangle(3))
    nx,ny = size(R)
    out = zeros(ColorTypes.GrayA{Float64},nx,ny)
    δx,δy = step.(split(points(R)))
    δ = sqrt(δx^2+δy^2)/2
    M = Manifold(eltype(ga))
    for x ∈ 1:nx, y ∈ 1:ny
        P = Chain{M}(1.0,R[x,y][1],R[x,y][2])
        c = 0.0
        for g ∈ ga
            norm(P∧g)<δ && (c+=1)
        end
        out[1+ny-y,x] = ColorTypes.GrayA(c,c)
    end
    return out
end

end # module
