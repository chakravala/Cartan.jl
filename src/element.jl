
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
export adjacency, antiadjacency, facetsigns
import Grassmann: norm, column, columns, points, pointset
using Base.Threads

@inline iterpts(t,f) = iterable(fullpoints(t),f)
@inline iterable(p,f) = range(f,f,length=length(p))
@inline iterable(p,f::F) where F<:Function = f.(value(p))
@inline iterable(p,f::ChainBundle) = value(f)
@inline iterable(p,f::F) where F<:AbstractVector = f
@inline callable(c::F) where F<:Function = c
@inline callable(c) = x->c

edgelength(v) = value(abs(v[2]-v[1]))
function Grassmann.volumes(t::SimplexFrameBundle)
    if mdims(immersion(t))≠2
        Real.(abs.(Grassmann.detsimplex(t)))
    else
        edgelength.(affinehull(t))
    end
end

initedges(n::Int) = SimplexTopology(Values{2,Int}.(OneTo(n-1),2:n),Base.OneTo(n))
initedges(r::R) where R<:AbstractVector = SimplexFrameBundle(PointCloud(initpoints(r)),initedges(length(r)))
function initmesh(r::R) where R<:AbstractVector
    t = initedges(r); p = coordinates(t); n = length(p)
    bound = Values{1,Int}.([1,n])
    p,SimplexTopology(bound,vertices(bound),n),immersion(t)
end

initpoints(P::AbstractArray) = initpoint.(vec(P))
initpoint(P::Real) = Chain{varmanifold(2),1}(1.0,P)
@generated function initpoints(P,::Val{n}) where n
    Expr(:.,:(Chain{$(varmanifold(n+1)),1}),
         Expr(:tuple,1.0,[:(P[$k,:]) for k ∈ list(1,n)]...))
end
@generated function initpoint(P::Chain{V,G,T,N} where {V,G,T}) where N
    Expr(:call,:(Chain{$(varmanifold(N+1)),1}),
         Expr(:tuple,1.0,[:(P[$k]) for k ∈ list(1,N)]...))
end

function initpointsdata(P,E,N::Val{n}=Val(size(P,1))) where n
    p = PointCloud(initpoints(P,N)); l = list(1,n)
    p,SimplexTopology([Int.(E[l,k]) for k ∈ 1:size(E,2)],length(p))
end

function initmeshdata(P,E,T,N::Val{n}=Val(size(P,1))) where n
    p,e = initpointsdata(P,E,N); l = list(1,n+1); np = length(p)
    p,e,SimplexTopology([Int.(T[l,k]) for k ∈ 1:size(T,2)],OneTo(np),np)
end
function totalmeshdata(P,E,T,N::Val{n}=Val(size(P,1))) where n
    p = PointCloud(initpoints(P,N))
    np,ln,ln1 = length(p),list(1,n),list(1,n+1)
    t = SimplexTopology([Int.(T[ln1,k]) for k ∈ 1:size(T,2)],OneTo(np),np)
    bd = [Int.(E[ln,k]) for k ∈ 1:size(E,2)]
    ed = edgetopology(p(t))
    ne = length(bd)
    ind = Vector{Int}(undef,ne)
    for i ∈ OneTo(ne)
        bdi = bd[i]
        j = findfirst(x->x==bdi,ed)
        if isnothing(j)
            bdir = reverse(bdi)
            k = findfirst(x->x==bdir,ed)
            ind[i] = k
            ed[k] = bdi
        else
            ind[i] = j
        end
    end
    p,SimplexTopology((global top_id+=1),ed,vertices(view(ed,ind)),length(p),ind),t
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

doubleget(x,i) = getindex.(Ref(x),i)
reducedcolumns(m::AbstractFrameBundle) = reducedcolumns(immersion(m))
function reducedcolumns(m::SimplexTopology)
    iscover(m) && (return columns(m))
    v = vertices(m)
    ind = Dict(v .=> OneTo(length(v)))
    doubleget.(Ref(ind),columns(m))
end

pointset(m::SimplexTopology) = vertices(m)
pointset(m::AbstractFrameBundle) = vertices(m)
pointset(e::ImmersedTopology{N,1}) where N = vertices(e)
vertices(e::ImmersedTopology{1,1}) = column(e)
function vertices(e::ImmersedTopology{N,1}) where N
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

antiadjacency(t,cols=reducedcolumns(t)) = (A = sparse(t,cols); A-transpose(A))
adjacency(t,cols=reducedcolumns(t)) = (A = sparse(t,cols); A+transpose(A))
SparseArrays.sparse(t::AbstractFrameBundle,cols=reducedcolumns(t)) = sparse(immersion(t),cols)
function SparseArrays.sparse(t::SimplexTopology{N},cols=reducedcolumns(t)) where N
    np = nodes(t)
    A = spzeros(Int,np,np)
    for c ∈ Grassmann.combo(N,2)
        A += @inbounds sparse(cols[c[1]],cols[c[2]],1,np,np)
    end
    return A
end

edges(t,cols::Values) = edges(t,adjacency(t,cols))
edges(t::AbstractFrameBundle) = t(edges(immersion(t)))
edges(t::AbstractFrameBundle,adj::AbstractMatrix) = t(edges(immersion(t),adj))
edges(t::SimplexTopology{2},adj::AbstractMatrix=nothing) = t
function edges(t::SimplexTopology,adj::AbstractMatrix=adjacency(t))
    SimplexTopology(0,edgetopology(adj),elements(t))
end
edgetopology(t) = edgetopology(adjacency(t))
function edgetopology(adj::AbstractMatrix=adjacency(t))
    f = findall((!)∘iszero,LinearAlgebra.triu(adj))
    Values{2,Int}[Values{2,Int}(@inbounds f[n].I) for n ∈ 1:length(f)]
end
edgefacets(t,cols::Values) = edgefacets(t,adjacency(t,cols))
edgefacets(t,adj=adjacency(t)) = FacetFrameBundle(edges(t,adj))

function facetsinterior(t::Vector{Values{M,Int}}) where M
    N = M-1
    N == 0 && (return [list(2,1)],Int[])
    out = Values{N,Int}[]
    bnd = Int[]
    for i ∈ t
        for w ∈ Values{N}.(Leibniz.combinations(sort(i),N))
            j = findfirst(isequal(w),out)
            isnothing(j) ? push!(out,w) : push!(bnd,j)
        end
    end
    return out,bnd
end
facets(t) = faces(t,Val(N-1))
facets(t,h) = faces(t,h,Val(N-1))
#faces(t,v::Val) = faces(value(t),v)
#faces(t,h,v,g=identity) = faces(value(t),h,v,g)
faces(t::Tuple,v,g=identity) = faces(t[1],t[2],v,g)
function faces(t::Vector{Values{M,Int}},::Val{N}) where {N,M}
    N == M && (return t)
    #N == 2 && (return edgefacets(t))
    N == 1 && (return Values.(vertices(t)))
    N == 0 && (return list(2,1))
    out = Values{N,Int}[]
    for i ∈ t
        for w ∈ Values{N}.(Leibniz.combinations(sort(i),N))
            w ∉ out && push!(out,w)
        end
    end
    return out
end
function faces(t::Vector{Values{M,Int}},h,::Val{N},g=identity) where {N,M}
    N == 0 && (return [list(1,N)],Int[sum(h)])
    out = Values{N,Int}[]
    bnd = Int[]
    vec = zeros(Variables{M,Int})
    val = N+1==M ? ∂(Manifold(points(t))(list(1,N+1))(I)) : ones(Values{binomial(M,N)})
    for i ∈ 1:length(t)
        vec[:] = @inbounds t[i]
        par = DirectSum.indexparity!(vec)
        w = Values{N,Int}.(Leibniz.combinations(par[2],N))
        for k ∈ 1:binomial(M,N)
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

∂(t::Values{N,<:Tuple}) where N = ∂.(t)
∂(t::Values{N,<:Vector}) where N = ∂.(t)
∂(t::Tuple{Vector{<:Values},Vector{Int}}) = ∂(t[1],t[2])
function ∂(t::Vector{<:Chain},u::Vector{Int})
    f = facets(t,u)
    f[1][findall(x->!iszero(x),f[2])]
end
function ∂(t::Vector{Values{N,Int}}) where N
    if N≠3
        f = facetsinterior(t)
        f[1][setdiff(1:length(f[1]),f[2])]
    else
        edgesfacets(t,adjacency(t).%2)
    end
end
#∂(t::Vector{<:Values}) = (f=facets(t,ones(Int,length(t))); f[1][findall(x->!iszero(x),f[2])])

import Grassmann: Leibniz
#=skeleton(t::ChainBundle,v) = skeleton(value(t),v)
@inline (::Leibniz.Derivation)(x::Vector{<:Chain},v=Val{true}()) = skeleton(x,v)
@generated skeleton(t::Vector{<:Chain{V}},v) where V = :(faces.(Ref(t),Ref(ones(Int,length(t))),$(Val.(list(1,mdims(V)))),abs))
#@generated skeleton(t::Vector{<:Chain{V}},v) where V = :(faces.(Ref(t),$(Val.(list(1,mdims(V))))))=#

const array_cache = (Array{T,2} where T)[]
const array_top_cache = (Array{T,2} where T)[]
array(m::Vector{<:Chain}) = [m[i][j] for i∈1:length(m),j∈1:mdims(Manifold(m))]
array(m::Vector{<:Values{N,Int}}) where N = Int[m[i][j] for i∈1:length(m),j∈1:N]
array(m::SimplexFrameBundle) = array(coordinates(m))
array!(m::SimplexFrameBundle) = array!(coordinates(m))
function array(m::SimplexTopology)
    B = bundle(m)
    if iszero(B)
        if isfull(m)
            return array(fulltopology(m))
        else
            return view(array(fulltopology(m)),subelements(m),:)
        end
    end
    for k ∈ length(array_top_cache):B
        push!(array_top_cache,Array{Any,2}(undef,0,0))
    end
    isempty(array_top_cache[B]) && (array_top_cache[B] = array(fulltopology(m)))
    return isfull(m) ? array_top_cache[B] : view(array_top_cache[B],subelements(m),:)
end
function array!(m::SimplexTopology)
    B = bundle(m)
    length(array_top_cache) ≥ B && (array_top_cache[B] = Array{Any,2}(undef,0,0))
end
function array(m::PointCloud)
    B = bundle(m)
    iszero(B) && (return array(points(m)))
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
Grassmann.submesh(m::SimplexFrameBundle) = submesh(coordinates(m))
Grassmann.submesh!(m::SimplexFrameBundle) = submesh!(coordinates(m))
function Grassmann.submesh(m::PointCloud)
    B = bundle(m)
    iszero(B) && (return submesh(points(m)))
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

@generated function Grassmann.vectors(t::SimplexFrameBundle,c=columns(topology(t)))
    sdims(t) == 1 && (return :(Grassmann.vectors.(fullpoints(t)[topology(t)])))
    v = Expr(:tuple,[:(M.(p[c[$i]]-A)) for i ∈ list(2,sdims(t))]...)
    V = :(Manifold(t)($(list(2,sdims(t))...)))
    quote
        p = points(t)
        V,M,A = $V,↓(Manifold(p)),p[c[1]]
        TensorOperator.(Chain{V,1}.($(Expr(:.,:Values,v))))
    end
end

Grassmann.detsimplex(m::SimplexFrameBundle) = ∧(m)/factorial(sdims(m)-1)
function Grassmann.:∧(m::SimplexFrameBundle)
    mdims(m)>sdims(m) ? .∧(Grassmann.vectors.(affinehull(m))) : .∧(affinehull(m))
end
for op ∈ (:mean,:barycenter,:curl)
    ops = Symbol(op,:s)
    @eval begin
        export $op, $ops
        Grassmann.$ops(m::SimplexFrameBundle) = Grassmann.$op.(affinehull(m))
    end
end

revrot(hk::TensorOperator,f=identity) = TensorOperator(revrot(value(hk)))
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
        l = list(1,N)
        for i ∈ l, j ∈ l
            M[tk[i],tk[j]] += mat[i,j]*m
        end
    end
    @eval function assemblelocal!(M,mat,tk::$T{N}) where N
        l = list(1,N)
        for i ∈ l, j ∈ l
            M[tk[i],tk[j]] += mat[i,j]
        end
    end
end

weights(t,d::Vector=degrees(t)) = inv.(d)
weights(t,B::SparseMatrixCSC) = inv.(degrees(t,B))
degrees(t::AbstractFrameBundle,B::SparseMatrixCSC) = degrees(immersion(t),B)
degrees(t::SimplexTopology,B::SparseMatrixCSC) = B*ones(Int,nodes(t)) # A = incidence(t)
degrees(t::AbstractFrameBundle,f=nothing) = degrees(immersion(t),f)
function degrees(t::SimplexTopology,f=nothing)
    b = zeros(Int,nodes(t))
    for tk ∈ topology(t)
        b[tk] .+= 1
    end
    return b
end

assembleincidence(t,f,B::SparseMatrixCSC) = Diagonal(iterpts(t,f))*B
assembleincidence(t,f,m=volumes(t),v::Val=Val(false)) = assembleincidence(t,iterpts(t,f),iterable(t,m))
function assembleincidence(X::AbstractFrameBundle,f,m,v::Val=Val(false))
    assembleincidence(immersion(X),f,m,v)
end
function assembleincidence(t::SimplexTopology,f::AbstractVector,m::AbstractVector,::Val{T}=Val{false}()) where T
    typ = eltype(T ? m : f)
    b = zeros(typ<:Int ? Float64 : typ,nodes(t))
    for k ∈ 1:elements(t)
        tk = t[k]
        b[tk] .+= f[tk].*m[k]
    end
    return b
end
function incidence(t::AbstractFrameBundle,cols::Values=columns(topology(t)))
    incidence(immersion(t),cols)
end
function incidence(t::SimplexTopology,cols::Values{N}=columns(topology(t))) where N
    np,nt = nodes(t),elements(t)
    A = spzeros(Int,np,nt)
    for i ∈ list(1,N)
        A += sparse(cols[i],1:nt,1,np,nt)
    end
    return A
end # node-element incidence, A[i,j]=1 -> i∈t[j]

assembleload(t,f=1,m=volumes(t)) = assembleincidence(t,iterpts(t,f)/sdims(t),m,Val(true))

interp(t::TensorField{B,F,N,<:FacetFrameBundle} where {B,F,N}) = interp(immersion(t),fiber(t))
interp(t,B::SparseMatrixCSC=incidence(t)) = Diagonal(inv.(degrees(t,B)))*B
interp(t,b,d=degrees(t,b)) = assembleincidence(t,inv.(d),b,Val(true))
pretni(t::TensorField{B,F,N,<:SimplexFrameBundle} where {B,F,N}) = pretni(immersion(t),fiber(t))
pretni(t,B::SparseMatrixCSC=incidence(t)) = interp(t,sparse(B'))
pretni(t,ut) = means(t,ut) #interp(t,ut,B::SparseMatrixCSC) = B*ut

interior(e) = interior(totalnodes(e),vertices(e))
interior(fixed,neq) = sort!(setdiff(1:neq,fixed))

facesindices(t,cols=columns(t)) = mdims(t) == 3 ? edgesindices(t,cols) : throw(error())

function edgesindices(t::SimplexFrameBundle)
    cols = columns(topology(t))
    edgesindices(t,edgefacets(t,cols),cols)
end
function edgesindices(t::SimplexFrameBundle,ed::SimplexFrameBundle,cols=columns(topology(t)))
    edgesindices(t,FacetFrameBundle(ed),cols)
end
function edgesindices(t::SimplexFrameBundle,e::FacetFrameBundle,cols=columns(immersion(t)))
    np,nt = nodes(t),elements(t)
    et = fulltopology(e); ne = totalelements(e); i,j,k = cols
    A = sparse(columns(et)...,OneTo(ne),np,np); A += A'
    ei = [Values(A[j[n],k[n]],A[i[n],k[n]],A[i[n],j[n]]) for n ∈ 1:nt]
    met = isinduced(e) ? metricextensor(e) : means(et,fullmetricextensor(e))
    PointCloud(0,means(et,fullpoints(e)),met)(SimplexTopology(0,ei,OneTo(ne),ne))
end

function neighbor(k::Int,ab...)::Int
    n = setdiff(intersect(ab...),k)
    isempty(n) ? 0 : n[1]
end

@generated function neighbors(A::SparseMatrixCSC,tk::Values{N},k) where N
    F = x->x>0
    N1 = Grassmann.list(1,N)
    x = Values{N}([Symbol(:x,i) for i ∈ N1])
    f = Values{N}([:(findall($F,A[:,tk[$i]])) for i ∈ N1])
    b = Values{N}([Expr(:call,:neighbor,:k,x[setdiff(N1,i)]...) for i ∈ N1])
    Expr(:block,Expr(:(=),Expr(:tuple,x...),Expr(:tuple,f...)),
        Expr(:call,:Values,b...))
end

function neighbors(t::SimplexTopology{N},n2e=incidence(t)) where N
    A = sparse(n2e')
    nt = elements(t)
    n = Vector{Values{N,Int}}(undef,nt)
    @threads for k ∈ 1:nt
        n[k] = neighbors(A,t[k],k)
    end
    return n
end

facetsign(i::Int,ni) = i<ni ? 1 : -1
facetsigns(i::Values,ni) = facetsign.(i,ni)
facetsigns(t::SimplexTopology) = facetsigns.(neighbors(t),OneTo(elements(t)))

function centroidvectors(t,m=means(t))
    p,nt = points(t),elements(t)
    V = Manifold(p)(2,3)
    c = Vector{FixedVector{3,Chain{V,1,Float64,2}}}(undef,nt)
    δ = Vector{FixedVector{3,Float64}}(undef,nt)
    for k ∈ 1:nt
        c[k] = V.(m[k].-p[value(t[k])])
        δ[k] = value.(abs.(c[k]))
    end
    return c,δ
end

