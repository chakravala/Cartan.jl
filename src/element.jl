
#   This file is part of TensorFields.jl
#   It is licensed under the AGPL license
#   TensorFields Copyright (C) 2019 Michael Reed
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
export solvepoisson, solveSD, solvetransport, solvedirichlet, adaptpoisson
export gradienthat, gradientCR, gradient, interp, nedelec, nedelecmean, jumps
export submesh, detsimplex, iterable, callable, value, edgelengths, laplacian
export boundary, interior, trilength, trinormals, incidence, degrees
import Grassmann: norm, column, columns, points, pointset, edges
using Base.Threads

@inline iterpts(t,f) = iterable(points(t),f)
@inline iterable(p,f) = range(f,f,length=length(p))
@inline iterable(p,f::F) where F<:Function = f.(value(p))
@inline iterable(p,f::ChainBundle) = value(f)
@inline iterable(p,f::F) where F<:AbstractVector = f
@inline callable(c::F) where F<:Function = c
@inline callable(c) = x->c

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

laplacian(t,u,m=volumes(t),g=gradienthat(t,m)) = value.(abs.(gradient(t,u,m,g)))
gradient(t::ChainBundle,u::Vector,m=volumes(t),g=gradienthat(t,m)) = [u[value(t[k])]⋅value(g[k]) for k ∈ 1:length(t)]
gradient(t::Vector,u::Vector,m=volumes(t),g=gradienthat(t,m)) = [u[value(t[k])]⋅value(g[k]) for k ∈ 1:length(t)]

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
    for tk ∈ value(t)
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
function incidence(t,cols=columns(t))
    np,nt = length(points(t)),length(t)
    A = spzeros(Int,np,nt)
    for i ∈ Grassmann.list(1,mdims(Manifold(t)))
        A += sparse(cols[i],1:nt,1,np,nt)
    end
    return A
end # node-element incidence, A[i,j]=1 -> i∈t[j]

assembleload(t,m=volumes(t),d=degrees(t,m)) = assembleincidence(t,inv.(d),m,Val(true))

interp(t) = assembleload(t,incidence(t))
interp(t,b,d=degrees(t,b)) = assembleload(t,b,d)
pretni(t,B::SparseMatrixCSC=incidence(t)) = assembleload(t,sparse(B'))
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

