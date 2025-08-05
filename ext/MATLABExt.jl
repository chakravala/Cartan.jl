module MATLABExt

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
isdefined(Cartan, :Requires) ? (import Cartan: MATLAB) : (using MATLAB)

const matlab_cache = (Array{T,2} where T)[]
const matlab_top_cache = (Array{T,2} where T)[]
function matlab(p::Array{T,2} where T,B)
    for k ∈ length(matlab_cache):B
        push!(matlab_cache,Array{Any,2}(undef,0,0))
    end
    matlab_cache[B] = p
end
function matlab_top(p::Array{T,2} where T,B)
        for k ∈ length(matlab_top_cache):B
        push!(matlab_top_cache,Array{Any,2}(undef,0,0))
    end
    matlab_top_cache[B] = p
end
function matlab(p::PointCloud)
    B = Cartan.bundle(p)
    if length(matlab_cache)<B || isempty(matlab_cache[B])
        ap = Cartan.array(p)'
        matlab(ap[2:end,:],B)
    else
        return matlab_cache[B]
    end
end
function matlab(p::SimplexTopology)
    B = Cartan.bundle(p)
    if length(matlab_top_cache)<B || isempty(matlab_top_cache[B])
        ap = Cartan.array(p)'
        matlab_top(vcat(ap,ones(length(p))'),B)
    else
        return matlab_top_cache[B]
    end
end
Cartan.initmesh(g,args...) = initmeshall(g,args...)[Cartan.list(1,3)]
initmeshall(g::Matrix{Int},args...) = initmeshall(Matrix{Float64}(g),args...)
function initmeshall(g,args...)
    P,E,T = MATLAB.mxcall(:initmesh,3,g,args...)
    pt,pe = Cartan.initmeshdata(P,E,T,Val(2))
    return (pt,pe,T,E,P)
end
function initmeshes(g,args...)
    pt,pe,T = initmeshall(g,args...)
    pt,pe,TensorField(FaceBundle(pt),Int[T[end,k] for k ∈ 1:size(T,2)])
end
totalmesh(g,args...) = totalmeshall(g,args...)[Cartan.list(1,3)]
totalmeshall(g::Matrix{Int},args...) = totalmeshall(Matrix{Float64}(g),args...)
function totalmeshall(g,args...)
    P,E,T = MATLAB.mxcall(:initmesh,3,g,args...)
    pt,pe = totalmeshdata(P,E,T,Val(2))
    return (pt,pe,T,E,P)
end
function totalmeshes(g,args...)
    pt,pe,T = totalmeshall(g,args...)
    pt,pe,TensorField(FaceBundle(pt),Int[T[end,k] for k ∈ 1:size(T,2)])
end
export initmeshes, totalmeshes, totalmesh
function Cartan.refinemesh(g,args...)
    pt,pe,T,E,P = initmeshall(g,args...)
    matlab(P,Cartan.bundle(fullcoordinates(pt)))
    matlab_top(E,Cartan.bundle(immersion(pe)))
    matlab_top(T,Cartan.bundle(immersion(pt)))
    return (g,Cartan.refine(pt),Cartan.refine(pe))
end
refinemesh3(g,p,e,t,s...) = MATLAB.mxcall(:refinemesh,3,g,matlab(p),matlab(e),matlab(t),s...)
refinemesh4(g,p,e,t,s...) = MATLAB.mxcall(:refinemesh,4,g,matlab(p),matlab(e),matlab(t),s...)
refinemesh(g,p::PointCloud,e,t) = refinemesh3(g,p,e,t)
refinemesh(g,p::PointCloud,e,t,s::String) = refinemesh3(g,p,e,t,s)
refinemesh(g,p::PointCloud,e,t,η::Vector{Int}) = refinemesh3(g,p,e,t,float.(η))
refinemesh(g,p::PointCloud,e,t,η::Vector{Int},s::String) = refinemesh3(g,p,e,t,float.(η),s)
refinemesh(g,p::PointCloud,e,t,u) = refinemesh4(g,p,e,t,u)
refinemesh(g,p::PointCloud,e,t,u,s::String) = refinemesh4(g,p,e,t,u,s)
refinemesh(g,p::PointCloud,e,t,u,η) = refinemesh4(g,p,e,t,u,float.(η))
refinemesh(g,p::PointCloud,e,t,u,η,s) = refinemesh4(g,p,e,t,u,float.(η),s)
refinemesh!(g::Matrix{Int},e::SimplexBundle,args...) = refinemesh!(Matrix{Float64}(g),e,args...)
function refinemesh!(g,pt::SimplexBundle,pe,s...)
    p,e,t = Cartan.unbundle(pt,pe)
    V = Manifold(p)
    P,E,T = refinemesh(g,p,e,t,s...); l = size(P,1)+1
    matlab(P,Cartan.bundle(p))
    matlab(E,Cartan.bundle(e))
    matlab(T,Cartan.bundle(t))
    Cartan.submesh!(p); Cartan.array!(p); Cartan.array!(t)
    Cartan.deletepointcloud!(Cartan.bundle(p))
    el,tl = Cartan.list(1,l-1),Cartan.list(1,l)
    np,ne,nt = size(P,2),size(E,2),size(T,2)
    ip = length(p)+1:np
    it = length(t)+1:nt
    Cartan.totalnodes!(t,np)
    resize!(fullpoints(p),np)
    resize!(fulltopology(e),ne)
    resize!(subelements(e),ne)
    resize!(verticesinv(e),np)
    resize!(vertices(t),np)
    resize!(fulltopology(t),nt)
    resize!(subelements(t),nt)
    fullpoints(p)[:] = [Chain{V,1,Float64}(1.0,P[:,k]...) for k ∈ 1:np]
    fulltopology(e)[:] = [Values{2,Int}(E[el,k]) for k ∈ 1:ne]
    fulltopology(t)[:] = [Values{3,Int}(T[tl,k]) for k ∈ 1:nt]
    vertices(t)[ip] = ip
    subelements(t)[it] = it
    ve = collect(vertices(fulltopology(e)))
    resize!(vertices(e),length(ve))
    vertices(e)[:] = ve
    verticesinv(e)[:] = verticesinv(np,ve)
    return (pt,pe)
end

end # module
