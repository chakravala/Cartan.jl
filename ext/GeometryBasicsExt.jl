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

(m::GridBundle{1})(t::GeometryBasics.Point) = m(t[1])
(m::GridBundle{2})(t::GeometryBasics.Point) = m(t[1],t[2])
(m::GridBundle{3})(t::GeometryBasics.Point) = m(t[1],t[2],t[3])
(m::GridBundle{4})(t::GeometryBasics.Point) = m(t[1],t[2],t[3],t[4])
(m::GridBundle{5})(t::GeometryBasics.Point) = m(t[1],t[2],t[3],t[4],t[5])
(m::TensorField{B,F,1} where {B,F})(t::GeometryBasics.Point) = m(t[1])
(m::TensorField{B,F,2} where {B,F})(t::GeometryBasics.Point) = m(t[1],t[2])
(m::TensorField{B,F,3} where {B,F})(t::GeometryBasics.Point) = m(t[1],t[2],t[3])
(m::TensorField{B,F,4} where {B,F})(t::GeometryBasics.Point) = m(t[1],t[2],t[3],t[4])
(m::TensorField{B,F,5} where {B,F})(t::GeometryBasics.Point) = m(t[1],t[2],t[3],t[4],t[5])

Cartan.unorientedpoly(p,v1,v2) = GeometryBasics.Point.(Cartan.polytransform(Cartan._unorientedplane(p,v1,v2)))
Cartan.orientedpoly(p,v1,v2) = GeometryBasics.Point.(Cartan.polytransform(Cartan._orientedplane(p,v1,v2)))

Base.convert(::Type{GeometryBasics.Point},t::T) where T<:LocalFiber = GeometryBasics.Point(base(t))
GeometryBasics.Point(t::T) where T<:LocalFiber = convert(GeometryBasics.Point,t)
GeometryBasics.Mesh(m::TensorField{B,<:Couple,2,<:GridBundle} where B) = GeometryBasics.Mesh(vectorize(m))
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
