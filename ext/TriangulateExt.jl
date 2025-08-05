module TriangulateExt

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
isdefined(Cartan, :Requires) ? (import Cartan: Triangulate) : (using Triangulate)

const triangle_point_cache = (Array{T,2} where T)[]
const triangle_simplex_cache = (Array{T,2} where T)[]
function triangle_point(p::Array{T,2} where T,B)
    for k ∈ length(triangle_point_cache):B
        push!(triangle_point_cache,Array{Any,2}(undef,0,0))
    end
    triangle_point_cache[B] = p
end
function triangle_simplex(p::Array{T,2} where T,B)
    for k ∈ length(triangle_simplex_cache):B
        push!(triangle_simplex_cache,Array{Any,2}(undef,0,0))
    end
    triangle_simplex_cache[B] = p
end
function triangle(p::PointCloud)
    B = bundle(p)
    iszero(B) && (return array(p)'[2:end,:])
    if length(triangle_point_cache)<B || isempty(triangle_point_cache[B])
        triangle_point(Cartan.array(p)'[2:end,:],B)
    else
        return triangle_point_cache[B]
    end
end
function triangle(p::SimplexTopology)
    B = p.id
    if length(triangle_simplex_cache)<B || isempty(triangle_simplex_cache[B])
        triangle_simplex(Cint.(Cartan.array(p)'),B)
    else
        return triangle_simplex_cache[B]
    end
end
triangle(p::Vector{<:Chain{V,1,T} where V}) where T = Cartan.array(p)'[2:end,:]
triangle(p::Vector{<:Values}) = Cint.(Cartan.array(p)')
function Triangulate.TriangulateIO(e::SimplexBundle,h=nothing)
    triin=Triangulate.TriangulateIO()
    triin.pointlist=triangle(fullcoordinates(e))
    triin.segmentlist=triangle(immersion(e))
    !isnothing(h) && (triin.holelist=triangle(h))
    return triin
end
function Triangulate.triangulate(i,e::SimplexBundle;holes=nothing)
    initmesh(Triangulate.triangulate(i,Triangulate.TriangulateIO(e,holes))[1])
end
Cartan.initmesh(t::Triangulate.TriangulateIO) = Cartan.initmeshdata(t.pointlist,t.segmentlist,t.trianglelist,Val(2))
#aran(area=0.001,angle=20) = "pa$(Printf.@sprintf("%.15f",area))q$(Printf.@sprintf("%.15f",angle))Q"

end # module
