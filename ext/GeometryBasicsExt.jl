module GeometryBasicsExt

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
isdefined(Cartan, :Requires) ? (import Cartan: GeometryBasics) : (using GeometryBasics)

Base.convert(::Type{GeometryBasics.Point},t::T) where T<:LocalFiber = GeometryBasics.Point(base(t))
GeometryBasics.Point(t::T) where T<:LocalFiber = convert(GeometryBasics.Point,t)
function GeometryBasics.Mesh(m::TensorField{B,<:Chain,2,<:GridBundle} where B)
    if mdims(fibertype(m))≠2
        GeometryBasics.Mesh(GridBundle(fiber(m)),normal(m))
    else
        GeometryBasics.Mesh(GridBundle(fiber(m)))
    end
end
function GeometryBasics.Mesh(m::GridBundle{2},n)
    pts, dec, uv = _mesh(m)
    GeometryBasics.Mesh(pts, dec; uv=uv, normal=Point.(vec(fiber(n))))
end
function GeometryBasics.Mesh(m::GridBundle{2})
    pts, dec, uv = _mesh(m)
    GeometryBasics.Mesh(pts, dec; uv=uv)
end
function _mesh(m::GridBundle{2})
    nm = size(points(m))
    faces = GeometryBasics.Tesselation(GeometryBasics.Rect(0, 0, 1, 1), nm)
    uv = Chain(0.0,0.0):map(inv,Chain((nm.-1)...)):Chain(1.0,1.0)
    pts = GeometryBasics.Point.(vec(points(m)))
    dec = GeometryBasics.decompose(GeometryBasics.QuadFace{GeometryBasics.GLIndex}, faces)
    uv = GeometryBasics.Vec{2}.(value.(vec(uv)))
    return pts, dec, uv
end
function Cartan.SimplexBundle(m::GeometryBasics.Mesh)
    c,f = GeometryBasics.coordinates(m),GeometryBasics.faces(m)
    s = size(eltype(c))[1]+1; V = Cartan.varmanifold(s) # s
    n = size(eltype(f))[1]
    p = PointCloud([Chain{V,1}(Values{s,Float64}(1.0,k...)) for k ∈ c])
    p(SimplexTopology([Values{n,Int}(k) for k ∈ f],length(p)))
end

end # module
