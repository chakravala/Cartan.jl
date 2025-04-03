
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
export boundary, interior, trilength, trinormals, incidence, degrees, edges, faces, facets
export adjacency, antiadjacency, facetsigns,refinemesh, refinemesh!, select, rms, unbundle
import Grassmann: norm, column, columns, points
using Base.Threads

@inline iterpts(t,f) = iterable(fullpoints(t),f)
@inline iterpts(t,f::Number) = iterable(totalnodes(t),f)
@inline iterable(p::Int,f::Number) = range(f,f,length=p)
@inline iterable(p,f::Number) = iterable(length(p),f)
@inline iterable(p,f::F) where F<:Function = f.(p) #f.(points(p))
@inline iterable(p,f::F) where F<:AbstractVector = f
@inline callable(c::F) where F<:Function = c
@inline callable(c) = x->c

(m::SimplexBundle)(t::Chain) = sinterp(m,t)
(m::TensorField{B,F,N,<:SimplexBundle} where {B,F,N})(t::Chain) = sinterp(m,t)
function sinterp(m,t::Chain)
    V = Manifold(pointtype(m))
    pt = Chain{V}(value(t))
    j = findfirst(pt,base(m))
    iszero(j) && (return zero(fibertype(m)))
    i = immersion(m)[j]
    Chain{V}(fiber(m)[i])⋅(Chain{V}(points(m)[i])\pt)
end

edgelength(v) = value(abs(v[2]-v[1]))
Grassmann.volumes(t::TensorField) = volumes(base(t))
Grassmann.volumes(t::SimplexBundle) = volumes(FaceBundle(t))
function Grassmann.volumes(t::FaceBundle)
    if sdims(immersion(t))≠2
        out = Grassmann.detsimplex(t)
        TensorField(t,Real.(abs.(fiber(out))))
    else
        TensorField(t,edgelength.(affinehull(t)))
    end
end

unbundle(te::Tuple) = unbundle(te...)
unbundle(t) = (fullcoordinates(t),immersion(t))
unbundle(t,e) = (fullcoordinates(t),immersion(e),immersion(t))
unbundle(g,t,e) = (g,unbundle(t,e)...)

initedges(n::Int) = SimplexTopology(Values{2,Int}.(OneTo(n-1),2:n),Base.OneTo(n))
initedges(r::R) where R<:AbstractVector = PointCloud(initpoints(r))(initedges(length(r)))
function initmesh(r::R) where R<:AbstractVector
    t = initedges(r); n = refnodes(t)
    bound = Values{1,Int}.([1,n.x])
    t,t(SimplexTopology(bound,vertices(bound),n))
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
    p(SimplexTopology([Int.(E[l,k]) for k ∈ 1:size(E,2)],length(p)))
end

function initmeshdata(P,E,T,N::Val{n}=Val(size(P,1))) where n
    e = initpointsdata(P,E,N); l = list(1,n+1); np = refnodes(e)
    e(SimplexTopology([Int.(T[l,k]) for k ∈ 1:size(T,2)],OneTo(np.x),np)),e
end
function totalmeshdata(P,E,T,N::Val{n}=Val(size(P,1))) where n
    p = PointCloud(initpoints(P,N))
    np,ln,ln1 = length(p),list(1,n),list(1,n+1)
    t = SimplexTopology([Int.(T[ln1,k]) for k ∈ 1:size(T,2)],OneTo(np),np)
    ed,ind = edgemeshdata(p(t),E,N)
    p(t),p(SimplexTopology((global top_id+=1),ed,vertices(view(ed,ind)),refnodes(t),ind,vertices(ed)))
end
function edgemeshdata(pt::SimplexBundle,E,::Val{n}) where n
    np,ln = totalnodes(pt),list(1,n)
    bd = [Int.(E[ln,k]) for k ∈ 1:size(E,2)]
    ed = edgetopology(pt)
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
    return ed,ind
end

for fun ∈ (:(Base.maximum),:(Base.minimum),:rms)
    @eval $fun(η::TensorField) = $fun(fiber(η))
end
rms(η) = norm(η)/sqrt(length(η))
select(η,ϵ=rms(η)) = sort!(findall(x->x>ϵ,fiber(η)))
function refinemesh(g::AbstractRange,args...)
    pt,pe = initmesh(g,args...)
    return (g,refine(pt),refine(pe))
end
function refinemesh!(::AbstractRange,pt::SimplexBundle,pe,η,_=nothing)
    p,e,t = unbundle(pt,pe)
    x,V = fullpoints(p),Manifold(p)
    for i ∈ η
        push!(x,Chain{V,1}(Values(1,(x[i+1][2]+x[i][2])/2)))
    end
    sort!(x,by=x->x[2])
    submesh!(p)
    np = length(x)
    totalnodes!(t,np)
    ind = length(t)+2:np
    fulltopology(e)[end] = Values(np)
    vertices(e)[end] = np
    verticesinv(e)[end] = 0
    resize!(verticesinv(e),np)
    verticesinv(e)[ind] .= 0
    verticesinv(e)[end] = np
    resize!(vertices(t),np)
    resize!(fulltopology(t),np-1)
    resize!(subelements(t),np-1)
    vertices(t)[ind] = ind
    for i ∈ ind
        fulltopology(t)[i-1] = Values(i-1,i)
        subelements(t)[i-1] = i-1
    end
    return (pt,pe)
end

Grassmann.columns(t::ImmersedTopology{N},i=1) where N = columns(topology(t))
Grassmann.columns(t::AbstractVector{<:Values{N}},i=1) where N = column.(Ref(t),list(i,N))

reducedcolumns(m::FrameBundle) = reducedcolumns(immersion(m))
reducedcolumns(m::SimplexTopology) = iscover(m) ? columns(m) : columns(subtopology(m))

vertices(e::ImmersedTopology{1,1}) = column(e)
const pointset = vertices
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

antiadjacency(t,cols=reducedcolumns(t),n=nodes(t)) = (A = sparse(t,cols,n); A-transpose(A))
adjacency(t,cols=reducedcolumns(t),n=nodes(t)) = (A = sparse(t,cols,n); A+transpose(A))
SparseArrays.sparse(t::FrameBundle,cols=reducedcolumns(t),np::Int=nodes(t)) = sparse(immersion(t),cols,np)
function SparseArrays.sparse(t::SimplexTopology,cols::Values{N}=reducedcolumns(t),np::Int=nodes(t)) where N
    A = spzeros(Int,np,np)
    for c ∈ Grassmann.combo(N,2)
        A += @inbounds sparse(cols[c[1]],cols[c[2]],1,np,np)
    end
    return A
end

edges(t,cols::Values,np=totalnodes(t)) = edges(t,adjacency(t,cols,np))
edges(t::ElementBundle) = t(edges(immersion(t)))
edges(t::ElementBundle,adj::AbstractMatrix) = t(edges(immersion(t),adj))
edges(t::SimplexTopology{2}) = t
edges(t::SimplexTopology{2},cols::Values) = t
edges(t::SimplexTopology{2},adj::AbstractMatrix) = t
function edges(t::SimplexTopology,adj::AbstractMatrix=adjacency(t,columns(t),totalnodes(t)))
    SimplexTopology(0,edgetopology(adj),refnodes(t))
end
edgetopology(t) = edgetopology(adjacency(t,columns(immersion(t)),totalnodes(t)))
function edgetopology(adj::AbstractMatrix=adjacency(t))
    f = findall((!)∘iszero,LinearAlgebra.triu(adj))
    Values{2,Int}[Values{2,Int}(@inbounds f[n].I) for n ∈ 1:length(f)]
end
function edges(t::DiscontinuousTopology{3},args...)
    nt = elements(t)
    out = Vector{Values{2,Int}}(undef,3nt)
    for i ∈ 1:nt
        ti = t[i]
        n = 3(i-1)
        out[n+1] = Values(ti[1],ti[2])
        out[n+2] = Values(ti[2],ti[3])
        out[n+3] = Values(ti[3],ti[1])
    end
    return SimplexTopology(0,out,OneTo(3nt),3nt)
end

function facetsinterior(t::SimplexTopology{M}) where M
    N = M-1
    #N == 0 && (return [list(2,1)],Int[])
    out = Values{N,Int}[]
    bnd = Int[]
    for i ∈ topology(t)
        for w ∈ Values{N}.(Leibniz.combinations(sort(i),N))
            j = findfirst(isequal(w),out)
            isnothing(j) ? push!(out,w) : push!(bnd,j)
        end
    end
    return SimplexTopology(0,out,vertices(t),refnodes(t)),bnd
end
facets(t) = faces(t,Val(sdims(t)-1))
facets(t,h) = faces(t,h,Val(sdims(t)-1))
faces(t::Tuple,v,g=identity) = faces(t[1],t[2],v,g)
faces(t,N::Int) = faces(t,Val(N))
faces(t::ElementBundle,v::Val) = t(faces(immersion(t),v))
faces(t::ElementBundle,h,v::Val,g=identity) = faces(immersion(t),h,v,g)
function faces(t::SimplexTopology{M},::Val{N}) where {N,M}
    ver = vertices(t)
    N == M && (return t)
    N == 2 && (return edges(t))
    N == 1 && (return SimplexTopology(Values.(ver),ver,refnodes(t)))
    #N == 0 && (return list(2,1))
    out = Values{N,Int}[]
    for i ∈ topology(t)
        for w ∈ Values{N}.(Leibniz.combinations(sort(i),N))
            w ∉ out && push!(out,w)
        end
    end
    return SimplexTopology(0,out,ver,refnodes(t))
end
function _facetsindices(t::SimplexTopology{M}) where M
    N = M-1
    ver = vertices(t)
    N == 2 && (e = edges(t); return (e,edgesindices(t,e)))
    #N == 1 && (return SimplexTopology(Values.(ver),ver,refnodes(t)))
    #N == 0 && (return list(2,1))
    out = Values{N,Int}[]
    outi = zeros(Values{M,Int},elements(t))
    bas = Values([Int.(isequal(j).(list(1,M))) for j ∈ list(1,M)])
    fw = list(1,M)
    bw = reverse(fw) # reverse order for opposing edge vertex
    for i ∈ OneTo(elements(t))
        c = Values{N}.(Leibniz.combinations(topology(t)[i],N))
        for j ∈ fw
            cj = @inbounds sort(c[j])
            k = findfirst(isequal(cj),out)
            if isnothing(k)
                push!(out,cj)
                outi[i] += length(out)*bas[bw[j]]
            else
                outi[i] += k*bas[bw[j]]
            end
        end
    end
    ne = length(out)
    return SimplexTopology(0,out,ver,refnodes(t)),SimplexTopology(0,outi,OneTo(ne),ne)
end
function faces(t::SimplexTopology{M},h,::Val{N},g=identity) where {N,M}
    #N == 0 && (return [list(1,N)],Int[sum(h)])
    out = Values{N,Int}[]
    bnd = Int[]
    vec = zeros(Variables{M,Int})
    val = N+1==M ? value(∂(Submanifold(N+1)(I))) : ones(Values{binomial(M,N)})
    bin = list(1,binomial(M,N))
    top = topology(t)
    for i ∈ 1:length(t)
        vec[:] = @inbounds top[i]
        par = Leibniz.indexparity!(vec)
        w = Values{N,Int}.(Leibniz.combinations(par[2],N))
        for k ∈ bin
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
    return SimplexTopology(0,out,refnodes(t)),bnd
end

#∂(t::Values{N,<:Tuple}) where N = ∂.(t)
#∂(t::Values{N,<:Vector}) where N = ∂.(t)
Grassmann.∂(t::Tuple{<:ElementBundle,Vector{Int}}) = ∂(t[1],t[2])
Grassmann.∂(t::Tuple{<:SimplexTopology,Vector{Int}}) = ∂(t[1],t[2])
Grassmann.∂(t::SimplexBundle,u::Vector{Int}) = t(∂(immersion(t),u))
function Grassmann.∂(t::SimplexTopology,u::Vector{Int})
    top,bnd = facets(t,u)
    top[findall((!)∘iszero,bnd)]
end
Grassmann.∂(t::ElementBundle) = t(∂(immersion(t)))
function Grassmann.∂(t::SimplexTopology{N}) where N
    if N≠3
        top,bnd = facetsinterior(t)
        top[setdiff(OneTo(length(top)),bnd)]
    else
        edges(t,adjacency(t).%2)
    end
end
#=function ∂(t::Vector{<:Values})
    f = facets(t,ones(Int,length(t)))
    f[1][findall((!)∘iszero,f[2])]
end=#

Grassmann.complement(t::ElementBundle) = t(complement(immersion(t)))
function Grassmann.complement(t::SimplexTopology)
    fullimmersion(t)[setdiff(1:totalelements(t),subelements(t))]
end

import Grassmann: Leibniz
skeleton(t::SimplexBundle) = skeleton(immersion(t))
@generated skeleton(t::SimplexTopology{N}) where N = :(faces.(Ref(t),Ref(ones(Int,elements(t))),$(Val.(list(1,N+1))),abs))
#@generated skeleton(t::SimplexTopology{N}) where N = :(faces.(Ref(t),$(Val.(list(1,N+1)))))

isedge(e) = t -> isedge(e,t)
isedge(e,t) = prod(e .∈ Ref(t))
function discontinuousboundary(dt,e)
    t = SimplexTopology(dt)
    out = copy(topology(e))
    for i ∈ OneTo(elements(e))
        ei = e[i]
        j = findfirst(isedge(ei),topology(t))
        out[i] = dt[j][invmap.(Ref(topology(t)[j]),ei)]
    end
    SimplexTopology(out,totalnodes(dt))
end

const array_cache = (Array{T,2} where T)[]
const array_top_cache = (Array{T,2} where T)[]
array(m::Vector{<:Chain}) = [m[i][j] for i∈1:length(m),j∈1:mdims(Manifold(m))]
array(m::Vector{<:Values{N,Int}}) where N = Int[m[i][j] for i∈1:length(m),j∈1:N]
array(m::SubArray) = array(Array(m))
array(m::SimplexBundle) = array(fullcoordinates(m))
array!(m::SimplexBundle) = array!(fullcoordinates(m))
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
function array(m::DiscontinuousTopology)
    B = bundle(m)
    if iszero(B)
        return array(topology(m))
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
function array!(m::DiscontinuousTopology)
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
submesh(m) = [m[i][j] for i∈1:length(m),j∈list(2,mdims(Manifold(m)))]
submesh!(m::SimplexBundle) = submesh!(fullcoordinates(m))
function submesh(m::SimplexBundle)
    out = submesh(fullcoordinates(m))
    (isdiscontinuous(m) ? isdisconnected(m) : iscover(m)) ? out : view(out,vertices(m),:)
end
function submesh(m::PointCloud)
    B = bundle(m)
    iszero(B) && (return submesh(points(m)))
    for k ∈ length(submesh_cache):B
        push!(submesh_cache,Array{Any,2}(undef,0,0))
    end
    isempty(submesh_cache[B]) && (submesh_cache[B] = submesh(points(m)))
    return submesh_cache[B]
end
function submesh!(m::PointCloud)
    B = bundle(m)
    length(submesh_cache) ≥ B && (submesh_cache[B] = Array{Any,2}(undef,0,0))
end

function Base.findfirst(P::GradedVector{V},M::SimplexBundle) where V
    p = fullpoints(M); t = immersion(M)
    for i ∈ 1:length(t)
        P ∈ Chain{V}(p[t[i]]) && (return i)
    end
    return 0
end
function Base.findlast(P::GradedVector{V},M::SimplexBundle) where V
    p = fullpoints(M); t = immersion(M)
    for i ∈ length(t):-1:1
        P ∈ Chain{V}(p[t[i]]) && (return i)
    end
    return 0
end

import Grassmann: affineframe
affineframe(t::SimplexBundle) = affineframe(FaceBundle(t))
@generated function affineframe(t::FaceBundle,c=columns(topology(t)))
    sdims(t) == 1 && (return :(TensorField(t,affineframe.(affinehull(t)))))
    v = Expr(:tuple,[:(M.(p[c[$i]]-A)) for i ∈ list(2,sdims(t))]...)
    V = :(Manifold(t)($(list(2,sdims(t))...)))
    quote
        p = fullpoints(t)
        V,M,A = $V,↓(Manifold(p)),p[c[1]]
        TensorField(t,TensorOperator.(Chain{V,1}.($(Expr(:.,:Values,v)))))
    end
end

Grassmann.detsimplex(m::ElementBundle) = ∧(m)/factorial(sdims(m)-1)
Grassmann.:∧(m::SimplexBundle) = ∧(FaceBundle(m))
function Grassmann.:∧(m::FaceBundle)
    TensorField(m,mdims(m)>sdims(m) ? .∧(Grassmann.vectors.(affinehull(m))) : .∧(affinehull(m)))
end
for op ∈ (:mean,:barycenter,:curl)
    ops = Symbol(op,:s)
    @eval begin
        export $op, $ops
        Grassmann.$ops(m::ElementBundle,u) = Grassmann.$ops(immersion(m),u)
        Grassmann.$ops(m::SimplexBundle) = Grassmann.$ops(FaceBundle(m))
        function Grassmann.$ops(m::FaceBundle)
            p,i = if isdisconnected(m)
                points(m),immersion(m)
            else
                fullpoints(m),SimplexTopology(immersion(m))
            end
            TensorField(m,Grassmann.$ops(i,p))
        end
        function Grassmann.$ops(m::SimplexMap)
            TensorField(FaceBundle(base(m)),Grassmann.$ops(subimmersion(m),fiber(m)))
        end
    end
end

revrot(hk::TensorOperator,f=identity) = TensorOperator(revrot(value(hk)))
revrot(hk::Chain{V,1},f=identity) where V = Chain{V,1}(-f(hk[2]),f(hk[1]))

gradienthat(t::TensorField,m=volumes(t)) = gradienthat(base(t),m)
gradienthat(t::SimplexBundle,m=volumes(t)) = gradienthat(FaceBundle(t),m)
function gradienthat(t::FaceBundle,m=volumes(t))
    N = mdims(Manifold(t))
    TensorField(t, if N == 2 #inv.(m)
        V = Manifold(points(t))
        c = Chain{↓(V),1}.(inv.(fiber(m)))
        TensorOperator.(Chain{V,1}.(-c,c))
    elseif N == 3
        h = fiber(curls(t))./2fiber(m)
        V = Manifold(h); V2 = ↓(V)
        [TensorOperator(Chain{V,1}(revrot.(V2.(value(h[k]))))) for k ∈ 1:length(h)]
    else
        TensorOperator.(Grassmann.grad.(affinehull(t)))
    end)
end

function laplacian(t::ElementMap,m=volumes(domain(t)),g=gradienthat(domain(t),m))
    out = gradient(t,m,g)
    TensorField(base(out),Real.(abs.(fiber(out))))
end
function gradient(t::SimplexMap,m=volumes(t),g=gradienthat(t,m))
    out = gradient_2(t,m,g)
    pt = continuous(base(out))
    TensorField(SimplexBundle(pt),interp(pt,fiber(out)))
end
function gradient(t::FaceMap,m=volumes(t),g=gradienthat(t,m))
    pt = continuous(base(t))
    gradient_2(pt,interp(pt,fiber(t)),m,g)
end
function gradient_2(t::SimplexMap,m=volumes(t),g=gradienthat(t,m))
    gradient_2(base(t),fiber(t),m,g)
end
function gradient_2(t::ElementBundle,u,m=volumes(t),g=gradienthat(t,m))
    T = immersion(t)
    pt = FaceBundle(t)
    TensorField(pt,[fiber(u)[T[k]]⋅value(value(fiber(g)[k])) for k ∈ 1:length(T)])
end

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

weights(t::FrameBundle) = inv(degrees(t))
weights(t::FrameBundle,B::SparseMatrixCSC) = inv(degrees(t,B))
weights(t::SimplexTopology) = inv.(degrees(t))
weights(t::SimplexTopology,B::SparseMatrixCSC) = inv.(degrees(t,B))
degrees(t::SimplexTopology,B::SparseMatrixCSC) = B*ones(Int,totalnodes(t)) #B=incidence(t)
degrees(t::FaceBundle,f=nothing) = degrees(SimplexBundle(t),f)
degrees(t::SimplexBundle,f=nothing) = TensorField(t,degrees(immersion(t),f))
function degrees(t::SimplexTopology,f=nothing)
    b = zeros(Int,totalnodes(t))
    for tk ∈ topology(t)
        b[tk] .+= 1
    end
    return b
end

assembleincidence(t,f,B::SparseMatrixCSC) = Diagonal(iterpts(t,f))*B
assembleincidence(t,f,m=volumes(t),v::Val=Val(false)) = assembleincidence(t,iterpts(t,f),iterable(t,m))
function assembleincidence(X::FrameBundle,f,m,v::Val=Val(false))
    assembleincidence(immersion(X),f,m,v)
end
function assembleincidence(t::ImmersedTopology,f::AbstractVector,m::AbstractVector,::Val{T}=Val{false}()) where T
    typ = fibertype(T ? m : f)
    b = zeros(typ<:Int ? Float64 : typ,totalnodes(t))
    for k ∈ 1:elements(t)
        tk = t[k]
        b[tk] .+= fiber(f)[tk].*fiber(m)[k]
    end
    return b
end
incidence(t::FrameBundle) = incidence(subimmersion(t),cols)
function incidence(t::ImmersedTopology,cols::Values{N}=columns(t)) where N
    np,nt = totalnodes(t),elements(t)
    A = spzeros(Int,np,nt)
    for i ∈ list(1,N)
        A += sparse(cols[i],1:nt,1,np,nt)
    end
    return A
end # node-element incidence, A[i,j]=1 -> i∈t[j]

assembleload(t,f=1,m=volumes(t)) = assembleincidence(t,iterpts(t,f)/sdims(t),m,Val(true))

interp(t::FaceMap) = TensorField(SimplexBundle(base(t)),interp(subimmersion(t),fiber(t)))
interp(t,B::SparseMatrixCSC=incidence(t)) = Diagonal(weights(t,B))*B
interp(t::FaceBundle,args...) = interp(subimmersion(t),args...)
interp(t::DiscontinuousTopology,b,w) = interp(t,b)
interp(t::DiscontinuousTopology,b) = view(b,discontinuousvertices(t))
interp(t::SimplexTopology,b,w=weights(t)) = assembleincidence(t,w,b,Val(true))
pretni(t::SimplexMap) = means(t)
pretni(t,B::SparseMatrixCSC=incidence(t)) = interp(t,sparse(B'))
pretni(t,ut) = means(t,ut) #interp(t,ut,B::SparseMatrixCSC) = B*ut

interpCR(pt,m) = interpCR(pt,edges(pt),m)
interpCR(pt,ed,m) = interpCR(pt,discontinuous(immersion(pt)),ed,m)
function interpCR(pt,dt::DiscontinuousTopology,ed,m::TensorField)
    ei = base(m)
    nt = elements(dt)
    b = zeros(totalnodes(dt))
    for k ∈ OneTo(nt)
        dk = dt[k]
        tk = immersion(pt)[k]
        nk = immersion(ei)[k]
        ek = immersion(ed)[nk]
        for j ∈ Values(1,2,3)
            ekj = invmap.(Ref(tk),ek[j])
            mj = fiber(m)[nk[j]]
            b[dk[ekj]] .+= ones(Values{2,Int}).*mj
            b[dk[findmissing(ekj)]] -= mj
        end
    end
    TensorField(SimplexBundle(fullcoordinates(pt),dt),b)
end

invmap(t::Values{3,Int},n::Int) = n == t[1] ? 1 : n == t[2] ? 2 : 3
findmissing(n::Values{2,Int}) = 1 ∉ n ? 1 : 2 ∉ n ? 2 : 3

interior(e) = interior(totalnodes(e),vertices(e))
interior(fixed,neq) = sort!(setdiff(1:neq,fixed))

facesindices(t) = sdims(t) == 3 ? edgesindices(t) : throw(error())

edgesindices(t::SimplexBundle) = edgesindices(t,edges(t))
function edgesindices(t::SimplexBundle,ed::SimplexBundle)
    edgesindices(t,FaceBundle(ed))
end
function edgesindices(t::SimplexBundle,e::FaceBundle)
    et = fullimmersion(e)
    met = isinduced(e) ? metricextensor(e) : means(et,fullmetricextensor(e))
    PointCloud(0,means(et,fullpoints(e)),met)(edgesindices(immersion(t),et))
end
function edgesindices(t::SimplexTopology,et::SimplexTopology{2}=edges(t))
    np,nt,ne = nodes(t),elements(t),totalelements(et)
    A = sparse(columns(et)...,OneTo(ne),np,np); A += A'
    ei = [localedge(A,t[n]) for n ∈ 1:nt]
    SimplexTopology(0,ei,OneTo(ne),ne)
end
function localedge(A,v::Values{2})
    v1,v2 = @inbounds (v[1],v[2])
    Values(A[v1,v2])
end
function localedge(A,v::Values{3})
    v1,v2,v3 = @inbounds (v[1],v[2],v[3])
    Values(A[v2,v3],A[v1,v3],A[v1,v2])
end
function localedge(A,v::Values{4})
    v1,v2,v3,v4 = @inbounds (v[1],v[2],v[3],v[4])
    Values(A[v1,v2],A[v1,v3],A[v1,v4],A[v2,v3],A[v2,v4],A[v3,v4])
end
function localedge(A,v::Values{5})
    v1,v2,v3,v4,v5 = @inbounds (v[1],v[2],v[3],v[4],v[5])
    Values(A[v1,v2],A[v1,v3],A[v1,v4],A[v1,v5],
        A[v2,v3],A[v2,v4],A[v2,v5],A[v3,v4],A[v3,v5],A[v4,v5])
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

neighbors(t::ElementBundle,args...) = neighbors(immersion(t),args...)
neighbors(t::DiscontinuousTopology) = neighbors(SimplexTopology(t))
neighbors(t::DiscontinuousTopology,n2e) = neighbors(SimplexTopology(t),n2e)
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
facetsigns(t::SimplexTopology,nbrs=neighbors(t)) = facetsigns.(nbrs,OneTo(elements(t)))
facetsigns(t::SimplexBundle,args...) = facetsigns(immersion(t),args...)

edgesigns(i::Values{2,Int}) = @inbounds i[1] < i[2] ? 1 : -1
edgesigns(i::Values{3,Int}) = @inbounds Values(i[2]<i[3] ? 1 : -1,i[3]<i[1] ? 1 : -1,i[1]<i[2] ? 1 : -1)
edgesigns(i::Values{4,Int}) = @inbounds Values(i[1]<i[2] ? 1 : -1,i[1]<i[3] ? 1 : -1,i[1]<i[4] ? 1 : -1,i[2]<i[3] ? 1 : -1,i[2]<i[4] ? 1 : -1,i[3]<i[4] ? 1 : -1)
edgesigns(i::Values{5,Int}) = @inbounds Values(i[1]<i[2] ? 1 : -1,i[1]<i[3] ? 1 : -1,i[1]<i[4] ? 1 : -1,i[1]<i[5] ? 1 : -1,i[2]<i[3] ? 1 : -1,i[2]<i[4] ? 1 : -1,i[2]<i[5] ? 1 : -1,i[3]<i[4] ? 1 : -1,i[3]<i[5] ? 1 : -1,i[4]<i[5] ? 1 : -1)

facets(i::Values{2,Int}) = @inbounds Values(Values(i[2]),Values(i[1]))
facets(i::Values{3,Int}) = @inbounds Values(Values(i[2],i[3]),Values(i[3],i[1]),Values(i[1],i[2]))
facets(i::Values{4,Int}) = @inbounds Values(Values(i[2],i[3],i[4]),Values(i[4],i[3],i[1]),Values(i[1],i[2],i[4]),Values(i[3],i[2],i[1]))
facets(i::Values{5,Int}) = @inbounds Values(Values(i[2],i[3],i[4],i[5]),Values(i[5],i[3],i[2],i[1]),Values(i[1],i[2],i[4],i[5]),Values(i[5],i[4],i[3],i[1]),Values(i[1],i[2],i[3],i[4],i[5]))

# LagrangeBundle

export LagrangeBundle, LagrangeBundle!

LagrangeBundle(pt) = LagrangeBundle(fullcoordinates(pt),immersion(pt))
LagrangeBundle(p,t) = LagrangeBundle!(PointCloud(0,copy(points(p))),t)
LagrangeBundle!(pt) = LagrangeBundle!(fullcoordinates(pt),immersion(pt))

function LagrangeBundle!(p::PointCloud,t::LagrangeEdges{2})
    ed = topology(t) # get element edges as nodes
    resize!(p,nodes(t))
    c = isinduced(p) ? fullpoints(p) : fullcoordinates(p)
    i,j = columns(cornertopology(pt))
    c[getindex.(ed,3)] = (c[i]+c[j])/2 # edge node coordinates
    return p(t)
end
function LagrangeBundle!(p::PointCloud,t::LagrangeEdges{M}) where M
    ed = topology(t) # get element edges as nodes
    resize!(p,nodes(t))
    c = isinduced(p) ? fullpoints(p) : fullcoordinates(p)
    i,j = columns(cornertopology(pt))
    ci,cj = c[i],c[j]
    cij = (cj-ci)/M
    c[getindex.(ed,3)] = cj+cij
    for x ∈ list(4,M+1)
        c[getindex.(ed,x)] = ci+(x-2)*cij
    end
    return p(t)
end

function LagrangeBundle!(p::PointCloud,t::LagrangeTriangles{2})
    ed = topology(t) # get element edges as nodes
    resize!(p,nodes(t))
    c = isinduced(p) ? fullpoints(p) : fullcoordinates(p)
    i,j,k = columns(cornertopology(t))
    ci,cj,ck = c[i],c[j],c[k]
    c[getindex.(ed,4)] = (cj+ck)/2 # edge node coordinates
    c[getindex.(ed,5)] = (ci+ck)/2
    c[getindex.(ed,6)] = (ci+cj)/2
    return p(t)
end
function LagrangeBundle!(p::PointCloud,t::LagrangeTriangles{M}) where M
    ed = topology(t) # get element edges as nodes
    resize!(p,nodes(t))
    c = isinduced(p) ? fullpoints(p) : fullcoordinates(p)
    x,y = columns(edges(t))
    cx = c[x]
    Δe = (c[y]-cx)/M
    i,j,k = columns(cornertopology(t))
    ci,cj,ck = c[i],c[j],c[k]
    cjk,cik,cij = (ck-cj)/M,(ck-ci)/M,(cj-ci)/M
    edg = getedge.(Ref(t),OneTo(totaledges(t)))
    for x ∈ list(1,M-1)
        c[getindex.(edg,x)] = cx+x*Δe
    end
    start = 3M+1
    #c[getindex.(ed,start)] = ci+cik+cij
    for x ∈ list(1,M-2)
        ls = lagrangesimplex(3,x-2)
        Y = start+ls:start+ls+x-1
        bw = reverse(OneTo(x))
        for y ∈ OneTo(x)
            c[getindex.(ed,Y[y])] = ci+bw[y]*cik+y*cij
        end
    end
    return p(t)
end

function LagrangeBundle!(p::PointCloud,t::LagrangeTetrahedra{M}) where M
    ed = topology(t) # get element edges as nodes
    resize!(p,nodes(t))
    c = isinduced(p) ? fullpoints(p) : fullcoordinates(p)
    i,j,k,l = columns(cornertopology(t))
    ci,cj,ck,cl = c[i],c[j],c[k],c[l]
    cij,cik,cil,cjk,cjl,ckl = (cj-ci)/M,(ck-ci)/M,(cl-ci)/M,(ck-cj)/M,(cl-cj)/M,(cl-ck)/M
    for x ∈ list(1,M-1)
        c[getindex.(ed,1+4+6(x-1))] = ci+x*cij
        c[getindex.(ed,2+4+6(x-1))] = ci+x*cik
        c[getindex.(ed,3+4+6(x-1))] = ci+x*cil
        c[getindex.(ed,4+4+6(x-1))] = cj+x*cjk
        c[getindex.(ed,5+4+6(x-1))] = cj+x*cjl
        c[getindex.(ed,6+4+6(x-1))] = ck+x*ckl
    end
    start = 4+6(M-1)+1
    for x ∈ list(1,M-2)
        ls = lagrangesimplex(3,x-2)
        Y = start+4ls:4:start+4ls+4x
        bw = reverse(OneTo(x))
        for y ∈ OneTo(x)
            c[getindex.(ed,Y[y])] = cj+bw[y]*cjk+y*cjl # jkl
            c[getindex.(ed,Y[y]+1)] = ci+bw[y]*cik+y*cil # ikl
            c[getindex.(ed,Y[y]+2)] = ci+bw[y]*cil+y*cij # ijl
            c[getindex.(ed,Y[y]+3)] = ci+bw[y]*cik+y*cij # ijk
        end
    end
    start += 4*facetsimplex(4,M)
    rz = reverse(OneTo(M-3))
    for z ∈ OneTo(M-3)
        ls1 = lagrangesimplex(4,z-2)
        for x ∈ OneTo(z)
            ls = ls1+lagrangesimplex(4,x-2)
            Y = start+ls:start+ls+x-1
            bw = reverse(OneTo(x))
            for y ∈ OneTo(x)
                c[getindex.(ed,Y[y])] = ci+rz[z]*cij+bw[y]*cik+y*cil
            end
        end
    end
    return p(t)
end

#=printlagrange(N::Int,M::Int) = printlagrange(Val(N),Val(M))
function printlagrange(::Val{2},::Val{M}) where M
    println("c[getindex.(ed,1)] = ci")
    println("c[getindex.(ed,2)] = cj")
    for x ∈ list(3,M+1)
        println("c[getindex.(ed,$x)] = ci+$(x-2)*cij")
    end
end
function printlagrange(::Val{3},::Val{M}) where M
    println("c[getindex.(ed,1)] = ci")
    println("c[getindex.(ed,2)] = cj")
    println("c[getindex.(ed,3)] = ck")
    for x ∈ list(1,M-1)
        println("c[getindex.(ed,$(3+x))] = cj+$x*cjk")
        println("c[getindex.(ed,$(3+(M-1)+x))] = ci+$x*cik")
        println("c[getindex.(ed,$(3+2(M-1)+x))] = ci+$x*cij")
    end
    start = 3M+1
    for x ∈ list(1,M-2)
        ls = lagrangesimplex(3,x-2)
        Y = start+ls:start+ls+x-1
        bw = reverse(OneTo(x))
        for y ∈ OneTo(x)
            println("c[getindex.(ed,$(Y[y]))] = ci+$(bw[y])*cik+$y*cij")
        end
    end
end
function printlagrange(::Val{4},::Val{M}) where M
    println("c[getindex.(ed,1)] = ci")
    println("c[getindex.(ed,2)] = cj")
    println("c[getindex.(ed,3)] = ck")
    println("c[getindex.(ed,4)] = cl")
    for x ∈ list(1,M-1)
        println("c[getindex.(ed,$(1+4+6(x-1)))] = ci+$x*cij")
        println("c[getindex.(ed,$(2+4+6(x-1)))] = ci+$x*cik")
        println("c[getindex.(ed,$(3+4+6(x-1)))] = ci+$x*cil")
        println("c[getindex.(ed,$(4+4+6(x-1)))] = cj+$x*cjk")
        println("c[getindex.(ed,$(5+4+6(x-1)))] = cj+$x*cjl")
        println("c[getindex.(ed,$(6+4+6(x-1)))] = ck+$x*ckl")
    end
    start = 4+6(M-1)+1
    for x ∈ list(1,M-2)
        ls = lagrangesimplex(3,x-2)
        Y = start+4ls:4:start+4ls+4x
        bw = reverse(OneTo(x))
        for y ∈ OneTo(x)
            println("c[getindex.(ed,$(Y[y]))] = cj+$(bw[y])*cik+$y*cil") # jkl
            println("c[getindex.(ed,$(Y[y]+1))] = ci+$(bw[y])*cik+$y*cil") # ikl
            println("c[getindex.(ed,$(Y[y]+2))] = ci+$(bw[y])*cil+$y*cij") # ijl
            println("c[getindex.(ed,$(Y[y]+3))] = ci+$(bw[y])*cik+$y*cij") # ijk
        end
    end
    start += 4*facetsimplex(4,M)
    rz = reverse(OneTo(M-3))
    for z ∈ OneTo(M-3)
        ls1 = lagrangesimplex(4,z-2)
        for x ∈ OneTo(z)
            ls = ls1+lagrangesimplex(4,x-2)
            Y = start+ls:start+ls+x-1
            bw = reverse(OneTo(x))
            for y ∈ OneTo(x)
                println("c[getindex.(ed,$(Y[y]))] = ci+$(rz[z])*cij+$(bw[y])*cik+$y*cil")
            end
        end
    end
end=#

# refinement

refinement(t::LagrangeTriangles{1}) = cornertopology(t)
function refinement(t::LagrangeTriangles)
    out = reduce(vcat,refinetriangle.(topology(t)))
    SimplexTopology(0,out,vertices(t),nodes(t))
end

refinetriangle(t::Values{3}) = [t]
function refinetriangle(t::Values{6})
    [Values(t[1],t[6],t[5]),
     Values(t[2],t[4],t[6]),
     Values(t[3],t[5],t[4]),
     Values(t[4],t[5],t[6])]
end

function refinetriangle(t::Values{10})
    [Values(t[1],t[8],t[7]),
     Values(t[2],t[4],t[9]),
     Values(t[3],t[6],t[5]),
     Values(t[8],t[9],t[10]),
     Values(t[9],t[4],t[10]),
     Values(t[4],t[5],t[10]),
     Values(t[5],t[6],t[10]),
     Values(t[6],t[7],t[10]),
     Values(t[7],t[8],t[10])]
end
function refinetriangle(t::Values{15})
    [Values(t[1],t[10],t[9]),
     Values(t[2],t[4],t[12]),
     Values(t[3],t[7],t[6]),
     Values(t[11],t[12],t[15]),
     Values(t[12],t[4],t[15]),
     Values(t[4],t[5],t[15]),
     Values(t[5],t[6],t[14]),
     Values(t[6],t[7],t[14]),
     Values(t[7],t[8],t[14]),
     Values(t[8],t[9],t[13]),
     Values(t[9],t[10],t[13]),
     Values(t[10],t[11],t[13]),
     Values(t[11],t[15],t[13]),
     Values(t[5],t[14],t[15]),
     Values(t[8],t[13],t[14]),
     Values(t[13],t[15],t[14])]
end

# refine tetrahedron

refinement(t::LagrangeTetrahedra{1}) = cornertopology(t)
function refinement(t::LagrangeTetrahedra)
    out = reduce(vcat,refinetetrahedron.(topology(t)))
    SimplexTopology(0,out,vertices(t),nodes(t))
end

refinetetrahedron(t::Values{4}) = [t]

