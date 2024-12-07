
#   This file is part of Cartan.jl
#   It is licensed under the AGPL license
#   Cartan Copyright (C) 2019 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com

export assemble, assembleglobal, assemblestiffness, assembleconvection, assembleSD
export assemblemass, assemblefunction, assemblemassfunction, assembledivergence
export assemblemassincidence, asssemblemassnodes, assemblenodes
export assembleload, assemblemassload, assemblerobin, edges, edgesindices, neighbors
export solvepoisson, solvetransportdiffusion, solvetransport, solvedirichlet, adaptpoisson
export gradienthat, gradientCR, gradient, interp, nedelec, nedelecmean, jumps
export submesh, detsimplex, iterable, callable, value, edgelengths, laplacian
export boundary, interior, trilength, trinormals, incidence, degrees, edges
import Grassmann: norm, column, columns, points, pointset
using Base.Threads

@inline iterpts(t,f) = iterable(points(t),f)
@inline iterable(p,f) = range(f,f,length=length(p))
@inline iterable(p,f::F) where F<:Function = f.(value(p))
@inline iterable(p,f::ChainBundle) = value(f)
@inline iterable(p,f::F) where F<:AbstractVector = f
@inline callable(c::F) where F<:Function = c
@inline callable(c) = x->c

edgelength(v) = value(abs(base(v[2])-base(v[1])))
Grassmann.volumes(t::SimplexFrameBundle) = mdims(immersion(t))≠2 ? Grassmann.volumes(t,Grassmann.detsimplex(t)) : edgelength.(PointCloud(t)[immersion(t)])

initedges(n::Int) = SimplexTopology(Values{2,Int}.(1:n-1,2:n),Base.OneTo(n))
initedges(r::R) where R<:AbstractVector = SimplexFrameBundle(PointCloud(initpoints(r)),initedges(length(r)))
function initmesh(r::R) where R<:AbstractVector
    t = initedges(r); p = PointCloud(t); n = length(p)
    bound = Values{1,Int}.([1,n])
    p,SimplexTopology(bound,vertices(bound),n),ImmersedTopology(t)
end

initpoints(P::T) where T<:AbstractVector{<:Real} = Chain{varmanifold(2),1}.(1.0,P)
initpoints(P::T) where T<:AbstractRange = Chain{varmanifold(2),1}.(1.0,P)
initpoints(P::AbstractVector) = initpoint.(P)
initpoints(P::AbstractMatrix) = initpoint.(P[:])
@generated function initpoint(P::Chain{V,G,T,N} where {V,G,T}) where N
    Expr(:call,:(Chain{$(varmanifold(N+1)),1}),
         Expr(:tuple,1.0,[:(P[$k]) for k ∈ list(1,N)]...))
end

function initpointsdata(P,E,N::Val{n}=Val(size(P,1))) where n
    p = PointCloud(initpoints(P,N)); l = list(1,n)
    p,SimplexTopology([Int.(E[l,k]) for k ∈ 1:size(E,2)],length(p))
end

function initmeshdata(P,E,T,N::Val{n}=Val(size(P,1))) where n
    p,e = initpointsdata(P,E,N); l = list(1,n+1)
    t = SimplexTopology([Int.(T[l,k]) for k ∈ 1:size(T,2)],length(p))
    return p,e,t
end

select(η,ϵ=sqrt(norm(η)^2/length(η))) = sort!(findall(x->x>ϵ,η))
refinemesh(g::R,args...) where R<:AbstractRange = (g,initmesh(g,args...)...)
function refinemesh!(::R,p::ChainBundle{W},e,t,η,_=nothing) where {W,R<:AbstractRange}
    p = points(t)
    x,T,V = value(p),value(t),Manifold(p)
    for i ∈ η
        push!(x,Chain{V,1}(Values(1,(x[i+1][2]+x[i][2])/2)))
    end
    sort!(x,by=x->x[2]); submesh!(p)
    e[end] = Chain{p(2),1}(Values(length(x)))
    for i ∈ length(t)+2:length(x)
        push!(T,Chain{p,1}(Values{2,Int}(i-1,i)))
    end
end

Grassmann.columns(t::ImmersedTopology{N},i=1) where N = columns(topology(t))
Grassmann.columns(t::AbstractVector{<:Values{N}},i=1) where N = column.(Ref(t),list(i,N))

pointset(m::SimplexTopology) = vertices(m)
pointset(m::AbstractFrameBundle) = vertices(m)
pointset(e::Vector{Values{N,Int}}) where N = vertices(e)
vertices(e::Vector{Values{1,Int}}) = column(e)
function vertices(e::Vector{Values{N,Int}}) where N
    out = Int[]
    mx = 0
    for i ∈ e
        for k ∈ i
            if k ∉ out
                k > mx && (mx = k)
                push!(out,k)
            end
        end
    end
    n = length(out)
    mx≠n ? out : Base.OneTo(n)
end

antiadjacency(t::AbstractFrameBundle,cols=columns(topology(t))) = (A = sparse(t,cols); A-transpose(A))
adjacency(t,cols=columns(immersion(t))) = (A = sparse(t,cols); A+transpose(A))
function SparseArrays.sparse(t::AbstractFrameBundle,cols=columns(topology(t)))
    np,N = length(points(t)),mdims(Manifold(t))
    A = spzeros(Int,np,np)
    for c ∈ Grassmann.combo(N,2)
        A += @inbounds sparse(cols[c[1]],cols[c[2]],1,np,np)
    end
    return A
end

edges(t,cols::Values) = edges(t,adjacency(t,cols))
function edges(t,adj=adjacency(t))
    mdims(t) == 2 && (return t)
    N = mdims(Manifold(t))
    f = findall(x->!iszero(x),LinearAlgebra.triu(adj))
    SimplexTopology([Values{2,Int}(@inbounds f[n].I) for n ∈ 1:length(f)],length(points(t)))
end

#=function facetsinterior(t::Vector{<:Chain{V}}) where V
    N = mdims(Manifold(t))-1
    W = V(list(2,N+1))
    N == 0 && (return [Chain{W,1}(list(2,1))],Int[])
    out = Chain{W,1,Int,N}[]
    bnd = Int[]
    for i ∈ t
        for w ∈ Values{N,Int}.(Leibniz.combinations(sort(value(i)),N))
            j = findfirst(isequal(w),out)
            isnothing(j) ? push!(out,w) : push!(bnd,j)
        end
    end
    return out,bnd
end
facets(t) = faces(t,Val(mdims(Manifold(t))-1))
facets(t,h) = faces(t,h,Val(mdims(Manifold(t))-1))
faces(t,v::Val) = faces(value(t),v)
faces(t,h,v,g=identity) = faces(value(t),h,v,g)
faces(t::Tuple,v,g=identity) = faces(t[1],t[2],v,g)
function faces(t::Vector{<:Chain{V}},::Val{N}) where {V,N}
    N == mdims(V) && (return t)
    N == 2 && (return edges(t))
    W = V(list(2,N+1))
    N == 1 && (return Chain{W,1}.(pointset(t)))
    N == 0 && (return Chain{W,1}(list(2,1)))
    out = Chain{W,1,Int,N}[]
    for i ∈ value(t)
        for w ∈ Chain{W,1}.(DirectSum.combinations(sort(value(i)),N))
            w ∉ out && push!(out,w)
        end
    end
    return out
end
function faces(t::Vector{<:Chain{V}},h,::Val{N},g=identity) where {V,N}
    W = V(list(1,N))
    N == 0 && (return [Chain{W,1}(list(1,N))],Int[sum(h)])
    out = Chain{W,1,Int,N}[]
    bnd = Int[]
    vec = zeros(Variables{mdims(V),Int})
    val = N+1==mdims(V) ? ∂(Manifold(points(t))(list(1,N+1))(I)) : ones(Values{binomial(mdims(V),N)})
    for i ∈ 1:length(t)
        vec[:] = @inbounds value(t[i])
        par = DirectSum.indexparity!(vec)
        w = Chain{W,1}.(DirectSum.combinations(par[2],N))
        for k ∈ 1:binomial(mdims(V),N)
            j = findfirst(isequal(w[k]),out)
            v = h[i]*(par[1] ? -val[k] : val[k])
            if isnothing(j)
                push!(out,w[k])
                push!(bnd,g(v))
            else
                bnd[j] += g(v)
            end
        end
    end
    return out,bnd
end

∂(t::ChainBundle) = ∂(value(t))
∂(t::Values{N,<:Tuple}) where N = ∂.(t)
∂(t::Values{N,<:Vector}) where N = ∂.(t)
∂(t::Tuple{Vector{<:Chain},Vector{Int}}) = ∂(t[1],t[2])
∂(t::Vector{<:Chain},u::Vector{Int}) = (f=facets(t,u); f[1][findall(x->!iszero(x),f[2])])
∂(t::Vector{<:Chain}) = mdims(t)≠3 ? (f=facetsinterior(t); f[1][setdiff(1:length(f[1]),f[2])]) : edges(t,adjacency(t).%2)
#∂(t::Vector{<:Chain}) = (f=facets(t,ones(Int,length(t))); f[1][findall(x->!iszero(x),f[2])])

skeleton(t::ChainBundle,v) = skeleton(value(t),v)
@inline (::Leibniz.Derivation)(x::Vector{<:Chain},v=Val{true}()) = skeleton(x,v)
@generated skeleton(t::Vector{<:Chain{V}},v) where V = :(faces.(Ref(t),Ref(ones(Int,length(t))),$(Val.(list(1,mdims(V)))),abs))
#@generated skeleton(t::Vector{<:Chain{V}},v) where V = :(faces.(Ref(t),$(Val.(list(1,mdims(V))))))
=#

const array_cache = (Array{T,2} where T)[]
const array_top_cache = (Array{T,2} where T)[]
array(m::Vector{<:Chain}) = [m[i][j] for i∈1:length(m),j∈1:mdims(Manifold(m))]
array(m::Vector{<:Values{N}}) where N = [m[i][j] for i∈1:length(m),j∈1:N]
array(m::SimplexFrameBundle) = array(PointCloud(m))
array!(m::SimplexFrameBundle) = array!(PointCloud(m))
function array(m::SimplexTopology)
    B = bundle(m)
    for k ∈ length(array_top_cache):B
        push!(array_top_cache,Array{Any,2}(undef,0,0))
    end
    isempty(array_top_cache[B]) && (array_top_cache[B] = array(topology(m)))
    return array_top_cache[B]
end
function array!(m::SimplexTopology)
    B = bundle(m)
    length(array_top_cache) ≥ B && (array_top_cache[B] = Array{Any,2}(undef,0,0))
end
function array(m::PointCloud)
    B = bundle(m)
    for k ∈ length(array_cache):B
        push!(array_cache,Array{Any,2}(undef,0,0))
    end
    isempty(array_cache[B]) && (array_cache[B] = array(points(m)))
    return array_cache[B]
end
function array!(m::PointCloud)
    B = bundle(m)
    length(array_cache) ≥ B && (array_cache[B] = Array{Any,2}(undef,0,0))
end

const submesh_cache = (Array{T,2} where T)[]
#submesh(m) = [m[i][j] for i∈1:length(m),j∈2:mdims(Manifold(m))]
Grassmann.submesh(m::SimplexFrameBundle) = submesh(PointCloud(m))
Grassmann.submesh!(m::SimplexFrameBundle) = submesh!(PointCloud(m))
function Grassmann.submesh(m::PointCloud)
    B = bundle(m)
    for k ∈ length(submesh_cache):B
        push!(submesh_cache,Array{Any,2}(undef,0,0))
    end
    isempty(submesh_cache[B]) && (submesh_cache[B] = submesh(points(m)))
    return submesh_cache[B]
end
function Grassmann.submesh!(m::PointCloud)
    B = bundle(m)
    length(submesh_cache) ≥ B && (submesh_cache[B] = Array{Any,2}(undef,0,0))
end

function Base.findfirst(P::GradedVector{V},M::SimplexFrameBundle) where V
    p = points(M); t = immersion(M)
    for i ∈ 1:length(t)
        P ∈ Chain{V}(p[t[i]]) && (return i)
    end
    return 0
end
function Base.findlast(P::GradedVector{V},M::SimplexFrameBundle) where V
    p = points(M); t = immersion(M)
    for i ∈ length(t):-1:1
        P ∈ Chain{V}(p[t[i]]) && (return i)
    end
    return 0
end
Base.findall(P::GradedVector{V},t::SimplexFrameBundle) where V = findall(P .∈ Chain{V}.(points(t)[immersion(t)]))

Grassmann.detsimplex(m::SimplexFrameBundle) = ∧(m)/factorial(mdims(immersion(m))-1)
function Grassmann.:∧(m::SimplexFrameBundle)
    p = points(m); pm = p[m]; V = Manifold(p)
    if mdims(p)>mdims(immersion(m))
        .∧(Grassmann.vectors.(pm))
    else
        Chain{↓(V),mdims(V)-1}.(value.(.∧(pm)))
    end
end
for op ∈ (:mean,:barycenter,:curl)
    ops = Symbol(op,:s)
    @eval begin
        export $op, $ops
        Grassmann.$ops(m::SimplexFrameBundle) = Grassmann.$op.(points(m)[immersion(m)])
    end
end

revrot(hk::Chain{V,1},f=identity) where V = Chain{V,1}(-f(hk[2]),f(hk[1]))

function gradienthat(t,m=volumes(t))
    N = mdims(Manifold(t))
    if N == 2 #inv.(m)
        V = Manifold(points(t))
        c = Chain{↓(V),1}.(inv.(m))
        Chain{V,1}.(-c,c)
    elseif N == 3
        h = curls(t)./2m
        V = Manifold(h); V2 = ↓(V)
        [Chain{V,1}(revrot.(V2.(value(h[k])))) for k ∈ 1:length(h)]
    else
        Grassmann.grad.(points(t)[value(t)])
    end
end

laplacian_2(t,u,m=volumes(t),g=gradienthat(t,m)) = Real(abs(gradient_2(t,u,m,g)))
laplacian(t,m=volumes(domain(t)),g=gradienthat(domain(t),m)) = Real(abs(gradient(t,m,g)))
function gradient(f::ScalarMap,m=volumes(domain(f)),g=gradienthat(domain(f),m))
    TensorField(domain(f), interp(domain(f),gradient_2(domain(f),codomain(f),m,g)))
end
function gradient_2(t,u,m=volumes(t),g=gradienthat(t,m))
    T = immersion(t)
    [u[value(T[k])]⋅value(g[k]) for k ∈ 1:length(T)]
end
#=function gradient(t::SimplexFrameBundle,u::Vector,m=volumes(t),g=gradienthat(t,m))
    i = immersion(t)
    [u[value(i[k])]⋅value(g[k]) for k ∈ 1:length(t)]
end=#

for T ∈ (:Values,:Variables)
    @eval function assemblelocal!(M,mat,m,tk::$T{N}) where N
        for i ∈ 1:N, j∈ 1:N
            M[tk[i],tk[j]] += mat[i,j]*m
        end
    end
end

weights(t,d::Vector=degrees(t)) = inv.(d)
weights(t,B::SparseMatrixCSC) = inv.(degrees(t,f))
degrees(t,B::SparseMatrixCSC) = B*ones(Int,length(t)) # A = incidence(t)
function degrees(t,f=nothing)
    b = zeros(Int,length(points(t)))
    for tk ∈ immersion(t)
        b[value(tk)] .+= 1
    end
    return b
end

assembleincidence(t,f,B::SparseMatrixCSC) = Diagonal(iterpts(t,f))*B
assembleincidence(t,f,m=volumes(t)) = assembleincidence(t,iterpts(t,f),iterable(t,m))
function assembleincidence(t,f::F,m::V,::Val{T}=Val{false}()) where {F<:AbstractVector,V<:AbstractVector,T}
    b = zeros(eltype(T ? m : f),length(points(t)))
    for k ∈ 1:length(t)
        tk = value(t[k])
        b[tk] .+= f[tk].*m[k]
    end
    return b
end
function assembleincidence(X::SimplexFrameBundle,f::F,m::V,::Val{T}=Val{false}()) where {F<:AbstractVector,V<:AbstractVector,T}
    b,t = zeros(eltype(T ? m : f),length(points(X))),immersion(X)
    for k ∈ 1:length(t)
        tk = value(t[k])
        b[tk] .+= f[tk].*m[k]
    end
    return b
end
function incidence(t,cols=columns(topology(t)))
    np,nt = length(points(t)),length(immersion(t))
    A = spzeros(Int,np,nt)
    for i ∈ Grassmann.list(1,mdims(immersion(t)))
        A += sparse(cols[i],1:nt,1,np,nt)
    end
    return A
end # node-element incidence, A[i,j]=1 -> i∈t[j]

assembleload(t,m=volumes(t),d=degrees(t,m)) = assembleincidence(t,inv.(d),m,Val(true))

interp(t,B::SparseMatrixCSC=incidence(t)) = Diagonal(inv.(degrees(t,B)))*B
interp(t,b,d=degrees(t,b)) = assembleload(t,b,d)
pretni(t,B::SparseMatrixCSC=incidence(t)) = interp(t,sparse(B'))
pretni(t,ut,B=pretni(t)) = B*ut #interp(t,ut,B::SparseMatrixCSC) = B*ut

interior(e) = interior(length(points(e)),pointset(e))
interior(fixed,neq) = sort!(setdiff(1:neq,fixed))

facesindices(t,cols=columns(t)) = mdims(t) == 3 ? edgesindices(t,cols) : throw(error())

function edgesindices(t,cols=columns(t))
    np,nt = length(points(t)),length(t)
    e = edges(t,cols); i,j,k = cols
    A = sparse(getindex.(e,1),getindex.(e,2),1:length(e),np,np)
    V = ChainBundle(means(e,points(t))); A += A'
    e,[Chain{V,2}(A[j[n],k[n]],A[i[n],k[n]],A[i[n],j[n]]) for n ∈ 1:nt]
end

function neighbor(k::Int,ab...)::Int
    n = setdiff(intersect(ab...),k)
    isempty(n) ? 0 : n[1]
end

@generated function neighbors(A::SparseMatrixCSC,V,tk,k)
    N,F = mdims(Manifold(V)),(x->x>0)
    N1 = Grassmann.list(1,N)
    x = Values{N}([Symbol(:x,i) for i ∈ N1])
    f = Values{N}([:(findall($F,A[:,tk[$i]])) for i ∈ N1])
    b = Values{N}([Expr(:call,:neighbor,:k,x[setdiff(N1,i)]...) for i ∈ N1])
    Expr(:block,Expr(:(=),Expr(:tuple,x...),Expr(:tuple,f...)),
        Expr(:call,:(Chain{V,1}),b...))
end

function neighbors(t,n2e=incidence(t))
    V,A = Manifold(Manifold(t)),sparse(n2e')
    nt = length(t)
    n = Chain{V,1,Int,mdims(V)}[]; resize!(n,nt)
    @threads for k ∈ 1:nt
        n[k] = neighbors(A,V,t[k],k)
    end
    return n
end

function centroidvectors(t,m=means(t))
    p,nt = points(t),length(t)
    V = Manifold(p)(2,3)
    c = Vector{FixedVector{3,Chain{V,1,Float64,2}}}(undef,nt)
    δ = Vector{FixedVector{3,Float64}}(undef,nt)
    for k ∈ 1:nt
        c[k] = V.(m[k].-p[value(t[k])])
        δ[k] = value.(abs.(c[k]))
    end
    return c,δ
end

