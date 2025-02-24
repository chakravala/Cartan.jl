
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

export ProductSpace, RealRegion, NumberLine, Rectangle, Hyperrectangle, ⧺, ⊕, resample

resize_lastdim!(x::Vector,i) = resize!(x,i)

# ProductSpace

affmanifold(N::Int) = Submanifold(N+2)(list(2,N+1)...)
@generated function affinepoint(p::Chain{V,1,T}) where {V,T}
    :(Chain{$(V(list(1,mdims(V)+1)...))}(Values(one(T),$([:(@inbounds p[$i]) for i ∈ list(1,mdims(V))]...))))
end

struct ProductSpace{V,T,N,M,S} <: AbstractArray{Chain{V,1,T,N},N}
    v::Values{M,S} # how to deal with T???
    ProductSpace{V,T,N}(v::Values{M,S}) where {V,T,N,M,S} = new{V,T,N,M,S}(v)
    ProductSpace{V,T}(v::Values{M,S}) where {V,T,M,S} = new{V,T,mdims(V),M,S}(v)
end

const RealRegion{V,T<:Real,N,S<:AbstractVector{T}} = ProductSpace{V,T,N,N,S}
const NumberLine{V,T,S} = RealRegion{V,T,1,S}
const Rectangle{V,T,S} = RealRegion{V,T,2,S}
const Hyperrectangle{V,T,S} = RealRegion{V,T,3,S}

RealRegion{V}(v::Values{N,S}) where {V,T<:Real,N,S<:AbstractVector{T}} = ProductSpace{V,T,N}(v)
RealRegion(v::Values{N,S}) where {T<:Real,N,S<:AbstractVector{T}} = ProductSpace{affmanifold(N),T,N}(v)
ProductSpace{V}(v::Values{N,S}) where {V,T<:Real,N,S<:AbstractVector{T}} = ProductSpace{V,T,N}(v)
ProductSpace(v::Values{N,S}) where {T<:Real,N,S<:AbstractVector{T}} = ProductSpace{affmanifold(N),T,N}(v)

Base.show(io::IO,t::RealRegion{V,T,N,<:AbstractRange} where {V,T,N}) = print(io,'(',Chain(getindex.(t.v,1)),"):(",Chain(Number.(getproperty.(t.v,:step))),"):(",Chain(getindex.(t.v,length.(t.v))),')')

(::Base.Colon)(min::Chain{V,1,T},step::Chain{V,1,T},max::Chain{V,1,T}) where {V,T} = ProductSpace{V,T}(Colon().(value(min),value(step),value(max)))

Base.iterate(t::RealRegion) = (getindex(t,1),1)
Base.iterate(t::RealRegion,state) = (s=state+1; s≤length(t) ? (getindex(t,s),s) : nothing)

resize_lastdim!(m::ProductSpace,i) = (resize!(m.v[end],i); m)

resample(m::OneTo,i::Int) = LinRange(1,m.stop,i)
resample(m::UnitRange,i::Int) = LinRange(m.start,m.stop,i)
resample(m::StepRange,i::Int) = LinRange(m.start,m.stop,i)
resample(m::LinRange,i::Int) = LinRange(m.start,m.stop,i)
resample(m::StepRangeLen,i::Int) = StepRangeLen(m.ref,(m.step*(m.len-1))/(i-1),i)
resample(m::AbstractRange,i::NTuple{1,Int}) = resample(m,i...)
resample(m::ProductSpace,i::NTuple) = ProductSpace(resample.(m.v,i))

@generated Base.size(m::RealRegion{V}) where V = :(($([:(size(@inbounds m.v[$i])...) for i ∈ 1:mdims(V)]...),))
@generated Base.getindex(m::RealRegion{V,T,N},i::Vararg{Int}) where {V,T,N} = :(Chain{V,1,T}(Values{N,T}($([:((@inbounds m.v[$j])[@inbounds i[$j]]) for j ∈ 1:N]...))))
Base.getindex(m::NumberLine{V,T},i::Int) where {V,T} = Chain{V,1,T}(Values(((@inbounds m.v[1])[i],)))
@pure Base.getindex(t::RealRegion,i::CartesianIndex) = getindex(t,i.I...)
@pure Base.eltype(::Type{<:ProductSpace{V,T,N}}) where {V,T,N} = Chain{V,1,T,N}

Base.IndexStyle(::RealRegion) = IndexCartesian()
function Base.getindex(A::RealRegion, I::Int)
    Base.@_inline_meta
    @inbounds getindex(A, Base._to_subscript_indices(A, I)...)
end
function Base._to_subscript_indices(A::RealRegion, i::Int)
    Base.@_inline_meta
    Base._unsafe_ind2sub(A, i)
end
function Base._ind2sub(A::RealRegion, ind)
    Base.@_inline_meta
    Base._ind2sub(axes(A), ind)
end

⊕(a::AbstractVector{<:Real}...) = RealRegion(Values(a))
⊕(a::ProductSpace,b::AbstractVector{<:Real}) = RealRegion(Values(a.v...,b))
⊕(a::AbstractVector{<:Real},b::ProductSpace) = RealRegion(Values(a,b.v...))
⊕(a::ProductSpace,b::ProductSpace) = RealRegion(Values(a.v...,b.v...))
cross(a::ProductSpace,b::AbstractVector{<:Real}) = a⊕b
cross(a::ProductSpace,b::ProductSpace) = a⊕b

@generated ⧺(a::Real...) = :(Chain($([:(a[$i]) for i ∈ 1:length(a)]...)))
@generated ⧺(a::Complex...) = :(Chain($([:(a[$i]) for i ∈ 1:length(a)]...)))
⧺(a::Chain{A,G},b::Chain{B,G}) where {A,B,G} = Chain{A∪B,G}(vcat(a.v,b.v))

widths(t::AbstractVector) = t[end]-t[1]
widths(t::ProductSpace) = widths.(t.v)

remove(t::ProductSpace{V,T,2} where {V,T},::Val{1}) = (@inbounds t.v[2])
remove(t::ProductSpace{V,T,2} where {V,T},::Val{2}) = (@inbounds t.v[1])
@generated remove(t::ProductSpace{V,T,N} where {V,T},::Val{J}) where {N,J} = :(ProductSpace(domain(t).v[$(Values([i for i ∈ 1:N if i≠J]...))]))

# 1
(m::ProductSpace)(c::Colon,i::Int...) = (@inbounds m.v[1])
(m::ProductSpace)(i::Int,c::Colon,j::Int...) = m.v[2]
(m::ProductSpace)(i::Int,j::Int,c::Colon,k::Int...) = m.v[3]
(m::ProductSpace)(i::Int,j::Int,k::Int,c::Colon,l::Int...) = m.v[4]
(m::ProductSpace)(i::Int,j::Int,k::Int,l::Int,c::Colon,o::Int...) = m.v[5]

# 2 - 0
(m::ProductSpace)(c::Colon,::Colon,i::Int...) = ProductSpace(m.v[Values(1,2)])
(m::ProductSpace)(c::Colon,i::Int,::Colon,j::Int...) = ProductSpace(m.v[Values(1,3)])
(m::ProductSpace)(c::Colon,i::Int,j::Int,::Colon,k::Int...) = ProductSpace(m.v[Values(1,4)])
(m::ProductSpace)(c::Colon,i::Int,j::Int,k::Int,::Colon,l::Int...) = ProductSpace(m.v[Values(1,5)])
# 2 - 1
(m::ProductSpace)(i::Int,c::Colon,::Colon,j::Int...) = ProductSpace(m.v[Values(2,3)])
(m::ProductSpace)(i::Int,c::Colon,j::Int,::Colon,k::Int...) = ProductSpace(m.v[Values(2,4)])
(m::ProductSpace)(i::Int,c::Colon,j::Int,k::Int,::Colon,l::Int...) = ProductSpace(m.v[Values(2,5)])
# 2 - 2
(m::ProductSpace)(i::Int,j::Int,c::Colon,::Colon,k::Int...) = ProductSpace(m.v[Values(3,4)])
(m::ProductSpace)(i::Int,j::Int,c::Colon,k::Int,::Colon,l::Int...) = ProductSpace(m.v[Values(3,5)])
# 2 - 3
(m::ProductSpace)(i::Int,j::Int,k::Int,c::Colon,::Colon,l::Int...) = ProductSpace(m.v[Values(4,5)])

# 3 - 0 - 0
(m::ProductSpace)(c::Colon,::Colon,::Colon,i::Int...) = ProductSpace(m.v[Values(1,2,3)])
(m::ProductSpace)(c::Colon,::Colon,i::Int,::Colon,j::Int...) = ProductSpace(m.v[Values(1,2,4)])
(m::ProductSpace)(c::Colon,::Colon,i::Int,j::Int,::Colon,k::Int...) = ProductSpace(m.v[Values(1,2,5)])
# 3 - 0 - 1
(m::ProductSpace)(c::Colon,i::Int,::Colon,::Colon,j::Int...) = ProductSpace(m.v[Values(1,3,4)])
(m::ProductSpace)(c::Colon,i::Int,::Colon,j::Int,::Colon,k::Int...) = ProductSpace(m.v[Values(1,3,5)])
# 3 - 0 - 2
(m::ProductSpace)(c::Colon,i::Int,j::Int,::Colon,::Colon,k::Int...) = ProductSpace(m.v[Values(1,4,5)])
# 3 - 1
(m::ProductSpace)(i::Int,c::Colon,::Colon,::Colon,j::Int...) = ProductSpace(m.v[Values(2,3,4)])
(m::ProductSpace)(i::Int,c::Colon,j::Int,::Colon,::Colon,k::Int...) = ProductSpace(m.v[Values(2,4,5)])
(m::ProductSpace)(i::Int,c::Colon,::Colon,j::Int,::Colon,k::Int...) = ProductSpace(m.v[Values(2,3,5)])
# 3 - 2
(m::ProductSpace)(i::Int,j::Int,c::Colon,::Colon,::Colon,k::Int...) = ProductSpace(m.v[Values(3,4,5)])

# 4
(m::ProductSpace)(c::Colon,::Colon,::Colon,::Colon,i::Int...) = ProductSpace(m.v[Values(1,2,3,4)])
(m::ProductSpace)(c::Colon,::Colon,::Colon,i::Int,::Colon,j::Int...) = ProductSpace(m.v[Values(1,2,3,5)])
(m::ProductSpace)(c::Colon,::Colon,i::Int,::Colon,::Colon,j::Int...) = ProductSpace(m.v[Values(1,2,4,5)])
(m::ProductSpace)(c::Colon,i::Int,::Colon,::Colon,::Colon,j::Int...) = ProductSpace(m.v[Values(1,3,4,5)])
(m::ProductSpace)(i::Int,c::Colon,::Colon,::Colon,::Colon,j::Int...) = ProductSpace(m.v[Values(2,3,4,5)])

export CrossRange
struct CrossRange <: AbstractVector{Int}
    n::Int
    m::Int
    CrossRange(n::Int) = new(n,crossrange(n))
end

crossrange(n) = Int((isodd(n) ? n+1 : n)/2)-1

Base.iterate(t::CrossRange) = (getindex(t,1),1)
Base.iterate(t::CrossRange,state) = (s=state+1; s≤length(t) ? (getindex(t,s),s) : nothing)

#Base.resize!(m::CrossRange,i) = (m.n = i; m.m = crossrange(n))
Base.size(m::CrossRange) = (length(m),)
Base.length(m::CrossRange) = m.n
Base.getindex(m::CrossRange,i::Int) = i≤m.m ? i+m.m : i-m.m

# ImmersedTopology

export ImmersedTopology, ProductTopology, SimplexTopology, SimplexManifold
export QuotientTopology, OpenTopology, CompactTopology
export topology, immersion, vertices, iscover

const ImmersedTopology{N,M} = AbstractArray{Values{N,Int},M}
const immersion = ImmersedTopology

sdims(m::ImmersedTopology{N}) where N = N
sdims(m::Type{<:ImmersedTopology{N}}) where N = N
immersiontype(m::ImmersedTopology) = typeof(m)
fullimmersion(m::ImmersedTopology) = m
topology(m::ImmersedTopology{N,1}) where N = m
subelements(m::ImmersedTopology{N,1}) where N = OneTo(length(m))

@pure Base.eltype(::Type{ImmersedTopology{N}}) where N = Values{N,Int}

# ProductTopology

struct ProductTopology{N,S<:AbstractVector{Int}} <: ImmersedTopology{N,N}
    v::Values{N,S}
    ProductTopology(v::Values{N,S}) where {N,S<:AbstractVector{Int}} = new{N,S}(v)
end
ProductTopology() = ProductTopology(Values{0,Vector{Int}}())
ProductTopology(i::Int,jk::Int...) = ProductTopology(Values(OneTo(i),OneTo.(jk)...))
ProductTopology(i::Int) = ProductTopology(Values((OneTo(i),)))
ProductTopology(i::AbstractVector) = ProductTopology(Values((i,)))
ProductTopology(i::AbstractVector,jk::AbstractVector...) = ProductTopology(Values(i,jk...))

Base.show(io::IO,t::ProductTopology{N,<:AbstractRange} where N) = print(io,Values(getindex.(t.v,1)),':',Values(getindex.(t.v,length.(t.v))))

(::Base.Colon)(min::Values{N,Int},max::Values{N,Int}) where N = ProductTopology(Colon().(value(min),value(max)))
(::Base.Colon)(min::Values{N,Int},step::Values{N,Int},max::Values{N,Int}) where N = ProductTopology(Colon().(value(min),value(step),value(max)))

Base.iterate(t::ProductTopology) = (getindex(t,1),1)
Base.iterate(t::ProductTopology,state) = (s=state+1; s≤length(t) ? (getindex(t,s),s) : nothing)

resize(::OneTo,i) = OneTo(i)
resize(m::StepRange,i) = isone(m.start) ? (1:1:i) : (i:-1:1)
resize(m::CrossRange,i) = CrossRange(i)
resize(m::ProductTopology{1},i) = ProductTopology(@inbounds resize(m.v[1],i))
@generated function resize(m::ProductTopology{N},i) where N
    Expr(:call,:ProductTopology,Expr(:call,:Values,[j≠N ? :(@inbounds m.v[$j]) : :(@inbounds resize(m.v[$j],i)) for j ∈ list(1,N)]...))
end

resample(m,i::Int...) = resample(m,i)
resample(m::ProductTopology{1},i::NTuple{1}) = ProductTopology(@inbounds resize(m.v[1],i[1]))
@generated function resample(m::ProductTopology{N},i::NTuple{N}) where N
    Expr(:call,:ProductTopology,Expr(:call,:Values,[:(@inbounds resize(m.v[$j],i[$j])) for j ∈ list(1,N)]...))
end

@generated Base.size(m::ProductTopology{N}) where N = :(($([:(size(@inbounds m.v[$i])...) for i ∈ 1:N]...),))
@generated Base.getindex(m::ProductTopology{N},i::Vararg{Int}) where N = :(Values{N,Int}($([:((@inbounds m.v[$j])[@inbounds i[$j]]) for j ∈ 1:N]...)))
Base.getindex(m::ProductTopology{1},i::Int) = Values(((@inbounds m.v[1])[i],))
@pure Base.getindex(t::ProductTopology,i::CartesianIndex) = getindex(t,i.I...)
@pure Base.eltype(::Type{<:ProductTopology{N}}) where N = Values{N,Int}

Base.IndexStyle(::ProductTopology) = IndexCartesian()
function Base.getindex(A::ProductTopology, I::Int)
    Base.@_inline_meta
    @inbounds getindex(A, Base._to_subscript_indices(A, I)...)
end
function Base._to_subscript_indices(A::ProductTopology, i::Int)
    Base.@_inline_meta
    Base._unsafe_ind2sub(A, i)
end
function Base._ind2sub(A::ProductTopology, ind)
    Base.@_inline_meta
    Base._ind2sub(axes(A), ind)
end

#exclude(m::ProductTopology{N},n::Int) where N = ProductTopology(m.v[vcat([i≠n ? Values((i,)) : Values{0,Int}() for i ∈ list(1,N)]...)])
#exclude(m::ProductTopology{N},n::Int...) where N = ProductTopology(m.v[vcat([i∉n ? Values((i,)) : Values{0,Int}() for i ∈ list(1,N)]...)])
@inline exclude(m::ProductTopology) = m
@inline exclude(m::ProductTopology{2},::Val{1}) = ProductTopology(m.v[2])
@inline exclude(m::ProductTopology{2},::Val{2}) = ProductTopology(m.v[1])
@generated function exclude(m::ProductTopology{N},::Val{n}) where {N,n}
    N==2 && (return :(ProductTopology(m.v[$n])))
    vals = vcat([i≠n ? Values((i,)) : Values{0,Int}() for i ∈ list(1,N)]...)
    :(ProductTopology(m.v[$vals]))
end
@generated function exclude(m::ProductTopology{N},::Val{n1},::Val{n2}) where {N,n1,n2}
    vals = vcat([i∉(n1,n2) ? Values((i,)) : Values{0,Int}() for i ∈ list(1,N)]...)
    :(ProductTopology(m.v[$vals]))
end
@generated function exclude(m::ProductTopology{N},::Val{n1},::Val{n2},::Val{n3}) where {N,n1,n2,n3}
    vals = vcat([i∉(n1,n2,n3) ? Values((i,)) : Values{0,Int}() for i ∈ list(1,N)]...)
    :(ProductTopology(m.v[$vals]))
end
@generated function exclude(m::ProductTopology{N},::Val{n1},::Val{n2},::Val{n3},::Val{n4}) where {N,n1,n2,n3,n4}
    vals = vcat([i∉(n1,n2,n3,n4) ? Values((i,)) : Values{0,Int}() for i ∈ list(1,N)]...)
    :(ProductTopology(m.v[$vals]))
end

cross(a::ProductTopology,b::ProductTopology) = ProductTopology(Values(a.v...,b.v...))
cross(a::ProductTopology,b::AbstractVector{Int}) = ProductTopology(Values(a.v...,b))
cross(a::ProductTopology,b::Int) = a × OneTo(b)

# SimplexTopology

top_id = 0

refval(p) = p
refval(p::Base.RefValue) = p.x
refnodes(p::Int) = Ref(p)
refnodes(p::Base.RefValue) = p
const RefInt = Union{Base.RefValue{Int},Int}

struct SimplexTopology{N,P<:AbstractVector{Int},F<:AbstractVector{Int},T} <: ImmersedTopology{N,1}
    id::Int # bundle
    t::Vector{Values{N,Int}} # fulltopology
    i::P # vertices
    p::Base.RefValue{Int} # totalnodes
    f::F # subelements
    I::P # fullvertices
    v::P # verticesinv
    function SimplexTopology(id::Int,t::Vector{Values{N,Int}},i::P,p::RefInt,f::F=OneTo(length(t)),I::P=i,ist::Bool=length(i)==refval(p),isf::Bool=length(f)==length(t)) where {N,P,F}
        new{N,P,F,(ist,isf)}(id,t,i,refnodes(p),f,I,verticesinv(p,i,ist && isf))
    end
    function SimplexTopology(id::Int,t::Vector{Values{N,Int}},i::P,p::RefInt,f::F,I::OneTo,ist::Bool=length(i)==refval(p),isf::Bool=length(f)==length(t)) where {N,P<:DenseVector,F}
        new{N,P,F,(ist,isf)}(id,t,i,refnodes(p),f,collect(I),verticesinv(p,i,ist && isf))
    end
end

function SimplexTopology(id::Int,t::Vector,i=vertices(t),p::RefInt=maximum(i))
    ist = length(i)==refval(p)
    SimplexTopology(id,t,i,p,subelements(t),i,ist,true)
end
SimplexTopology(id::Int,t::Vector,p::RefInt) = SimplexTopology(id,t,vertices(t),p)
SimplexTopology(t::Vector,i=vertices(t),p::RefInt=maximum(i)) = SimplexTopology((global top_id+=1),t,i,p)
SimplexTopology(t::Vector,p::RefInt) = SimplexTopology(t,vertices(t),p)
SimplexTopology(t::SimplexTopology) = t

bundle(m::SimplexTopology) = m.id
fulltopology(m::SimplexTopology) = m.t
topology(m::SimplexTopology) = isfull(m) ? fulltopology(m) : view(fulltopology(m),subelements(m))
totalelements(m::SimplexTopology) = length(fulltopology(m))
elements(m::SimplexTopology) = length(subelements(m))
subelements(m::SimplexTopology) = m.f
refnodes(m::SimplexTopology) = m.p
totalnodes!(m::SimplexTopology,p) = (refnodes(m).x = p)
totalnodes(m::SimplexTopology) = refval(refnodes(m))
nodes(m::SimplexTopology) = length(vertices(m))
fullvertices(m::SimplexTopology) = m.I
vertices(m::SimplexTopology) = m.i
verticesinv(m::SimplexTopology) = m.v

Base.size(m::SimplexTopology) = size(subelements(m))
Base.length(m::SimplexTopology) = elements(m)
Base.axes(m::SimplexTopology) = axes(subelements(m))
Base.getindex(m::SimplexTopology,i::Int) = getindex(fulltopology(m),getfacet(m,i))
Grassmann.mdims(m::SimplexTopology{N}) where N = N

getimage(m::SimplexTopology{N,<:AbstractVector} where N,i) = iscover(m) ? i : vertices(m)[i]
getimage(m::SimplexTopology{N,<:OneTo} where N,i) = i
getfacet(m::SimplexTopology{N,P,<:AbstractVector} where {N,P},i) = isfull(m) ? i : subelements(m)[i]
getfacet(m::SimplexTopology{N,P,<:OneTo} where{N,P},i) = i

"""
    istotal(m::SimplexTopology) -> Bool

Return `true` if `fulltopology(m)` is covering `totalnodes(m)`, and `false` otherwise.
"""
istotal(m::SimplexTopology{N,P,F,T} where {N,P,F}) where T = T[1]

"""
    isfull(m::SimplexTopology) -> Bool

Return `true` if `m` is equal to `fulltopology(m)`, and `false` otherwise.
"""
isfull(m::SimplexTopology{N,P,F,T} where {N,P,F}) where T = T[2]

"""
    iscover(m::SimplexTopology) -> Bool

Return `true` if `isfull(m) && istotal(m)`, and `false` otherwise.
"""
iscover(m::SimplexTopology) = isfull(m) && istotal(m)

function fullimmersion(m::SimplexTopology)
    top = fulltopology(m)
    ist = istotal(m)
    ind = if ist
        OneTo(totalnodes(m))
    else
        out = fullvertices(m)
        n = length(out)
        maximum(out) == n ? OneTo(n) : out
    end
    SimplexTopology(bundle(m),top,ind,refnodes(m),OneTo(length(top)),ind,ist,true)
end

function Base.getindex(m::SimplexTopology,i::AbstractVector{Int})
    ind,top = getfacet(m,i),fulltopology(m)
    ver = vertices(view(top,ind))
    SimplexTopology(bundle(m),top,ver,refnodes(m),ind,vertices(m),istotal(m))
end

(m::SimplexTopology)(i::AbstractVector{Int}) = subtopology(m,i)
function subtopology(m::SimplexTopology{N},i::AbstractVector{Int}) where N
    top = fulltopology(m)
    ind = Vector{Int}()
    for j ∈ subelements(m)
        prod(top[j] .∈ Ref(i)) && push!(ind,j)
    end
    SimplexTopology(bundle(m),top,i,refnodes(m),ind,vertices(m),istotal(m))
end

getelement(m::SimplexTopology{N,<:OneTo} where N,i::Int) = m[i]
function getelement(m::SimplexTopology{N,<:AbstractVector} where N,i::Int)
    iscover(m) ? m[i] : getindex.(Ref(verticesinv(m)),m[i])
end

subtopology(m::SimplexTopology{N,<:OneTo} where N) = topology(m)
function subtopology(m::SimplexTopology{N,<:AbstractVector} where N)
    iscover(m) ? topology(m) : getelement.(Ref(m),OneTo(elements(m)))
end

function subimmersion(m::SimplexTopology{N,<:OneTo} where N)
    iscover(m) && (return m)
    top,ind = topology(m),vertices(m)
    SimplexTopology(0,top,ind,length(ind),OneTo(length(top)),ind,true,true)
end
function subimmersion(m::SimplexTopology{N,<:AbstractVector} where N)
    iscover(m) && (return m)
    top,ind = topology(m),vertices(m)
    p = length(ind)
    ver = OneTo(p)
    SimplexTopology(0,subtopology(m),ver,p,OneTo(length(top)),ver,true,true)
end

verticesinv(n,ind,isc) = isc ? ind : verticesinv(n,ind) # isc = iscover
verticesinv(n::Base.RefValue,ind) = verticesinv(refval(n),ind)
verticesinv(n::Int,ind::OneTo) = ind
function verticesinv(n::Int,ind)
    out = zeros(Int,n)
    out[ind] = OneTo(length(ind))
    return out
end

refine(m::SimplexTopology{N,<:AbstractVector,<:AbstractVector}) where N = m
function refine(m::SimplexTopology{N,<:OneTo,<:AbstractVector}) where N
    i = collect(vertices(m))
    fi = vertices(m)≠fullvertices(m) ? collect(fullvertices(m)) : i
    SimplexTopology(bundle(m),fulltopology(m),i,refnodes(m),subelements(m),fi,istotal(m),isfull(m))
end
function refine(m::SimplexTopology{N,<:AbstractVector,<:OneTo}) where N
    SimplexTopology(bundle(m),fulltopology(m),vertices(m),refnodes(m),collect(subelements(m)),fullvertices(m),istotal(m),isfull(m))
end
function refine(m::SimplexTopology{N,<:OneTo,<:OneTo}) where N
    i = collect(vertices(m))
    fi = vertices(m)≠fullvertices(m) ? collect(fullvertices(m)) : i
    SimplexTopology(bundle(m),fulltopology(m),i,refnodes(m),collect(subelements(m)),fi,istotal(m),isfull(m))
end

# DiscontinuousTopology

export DiscontinuousTopology, discontinuous, disconnect, continuous

struct DiscontinuousTopology{N,P<:AbstractVector{Int},T<:SimplexTopology{N}} <: ImmersedTopology{N,1}
    id::Int # bundle
    t::T
    i::P # vertices
    I::P # fullvertices
end

function DiscontinuousTopology(m::SimplexTopology)
    DiscontinuousTopology(iszero(bundle(m)) ? 0 : (global top_id+=1),m)
end
function DiscontinuousTopology(m::SimplexTopology,I)
    DiscontinuousTopology(iszero(bundle(m)) ? 0 : (global top_id+=1),m,I)
end
function DiscontinuousTopology(id::Int,m::SimplexTopology{N}) where N
    n = N*totalelements(m)
    I = zeros(Int,n)
    cols = columns(fulltopology(m))
    for j ∈ list(1,N)
        I[j:N:n] = cols[j]
    end
    DiscontinuousTopology(id,m,I)
end
function DiscontinuousTopology(id::Int,m::SimplexTopology{N},I) where N
    i = if isfull(m)
        I
    else
        n = N*elements(m)
        i = zeros(Int,n)
        cols = columns(m)
        for j ∈ list(1,N)
            i[j:N:n] = cols[j]
        end
    end
    DiscontinuousTopology(id,m,i,I)
end
SimplexTopology(m::DiscontinuousTopology) = m.t

function discontinuousvertices(m::DiscontinuousTopology{N}) where N
    n = totalnodes(m)
    I = zeros(Int,n)
    cols = OneTo(totalelements(m))
    for j ∈ list(1,N)
        I[j:N:n] = cols
    end
    I
end

for fun ∈ (:totalelements,:elements,:subelements,:istotal,:isfull,:iscover,:fullimmersion)
    @eval $fun(m::DiscontinuousTopology) = $fun(SimplexTopology(m))
end
bundle(m::DiscontinuousTopology) = m.id
fulltopology(m::DiscontinuousTopology) = getindex.(Ref(m),OneTo(totalelements(m)))
topology(m::DiscontinuousTopology) = collect(m)
totalnodes(m::DiscontinuousTopology{N}) where N = N*totalelements(m)
nodes(m::DiscontinuousTopology{N}) where N = N*elements(m)
fullvertices(m::DiscontinuousTopology) = m.I
vertices(m::DiscontinuousTopology) = m.i

isdiscontinuous(m::SimplexTopology) = false
isdiscontinuous(m::DiscontinuousTopology) = true
isdisconnected(m::SimplexTopology) = false
isdisconnected(m::DiscontinuousTopology{N,<:OneTo} where N) = true
isdisconnected(m::DiscontinuousTopology{N,<:AbstractVector} where N) = false

Base.size(m::DiscontinuousTopology) = size(subelements(m))
Base.length(m::DiscontinuousTopology) = elements(m)
Base.axes(m::DiscontinuousTopology) = axes(subelements(m))
Base.getindex(m::DiscontinuousTopology{N},i::Int) where N = list(1,N).+N*(getfacet(m,i)-1)
Grassmann.mdims(m::DiscontinuousTopology{N}) where N = N

getimage(m::DiscontinuousTopology{N,<:OneTo} where N,i) = i
getimage(m::DiscontinuousTopology{N,<:AbstractVector} where N,i) = vertices(m)[i]
getfacet(m::DiscontinuousTopology,i) = getfacet(SimplexTopology(m),i)

function Base.getindex(m::DiscontinuousTopology,i::AbstractVector{Int})
    DiscontinuousTopology(bundle(m),SimplexTopology(m)[i],fullvertices(m))
end

(m::DiscontinuousTopology)(i::AbstractVector{Int}) = subtopology(m,i)
function subtopology(m::DiscontinuousTopology{N},i::AbstractVector{Int}) where N
    DiscontinuousTopology(bundle(m),subtopology(SimplexTopology(m),i),fullvertices(m))
end

continuous(m::SimplexTopology) = m
continuous(m::DiscontinuousTopology) = SimplexTopology(m)
discontinuous(m::DiscontinuousTopology) = m
discontinuous(m::SimplexTopology) = DiscontinuousTopology(0,m)
disconnect(m::SimplexTopology) = disconnect(discontinuous(m))
disconnect(m::DiscontinuousTopology) = DiscontinuousTopology(bundle(m),SimplexTopology(m),OneTo(totalnodes(m)))

getelement(m::DiscontinuousTopology{N,P,<:SimplexTopology{N,<:OneTo}} where {N,P},i::Int) = m[i]
function getelement(m::DiscontinuousTopology{N,P,<:SimplexTopology{N,<:AbstractVector}} where {N,P},i::Int)
    iscover(m) ? m[i] : list(1,N).+N*(i-1)
end

subtopology(m::DiscontinuousTopology{N,P,<:SimplexTopology{N,<:OneTo}} where {N,P}) = topology(m)
function subtopology(m::DiscontinuousTopology{N,P,<:SimplexTopology{N,<:AbstractVector}} where {N,P})
    iscover(m) ? topology(m) : getelement.(Ref(m),OneTo(elements(m)))
end

function subimmersion(m::DiscontinuousTopology)
    iscover(m) ? m : DiscontinuousTopology(subimmersion(SimplexTopology(m)))
end

function refine(m::DiscontinuousTopology{N,<:OneTo}) where N
    DiscontinuousTopology(bundle(m),refine(SimplexTopology(m)),collect(vertices(m)),collect(fullvertices(m)))
end
function refine(m::DiscontinuousTopology{N,<:AbstractVector}) where N
    DiscontinuousTopology(bundle(m),refine(SimplexTopology(m)),vertices(m),fullvertices(m))
end

#= VectorTopology

export VectorTopology

struct VectorTopology{N,T<:SimplexTopology{N}} <: ImmersedTopology{N,1}
    id::Int # bundle
    t::T
end

function VectorTopology(m::SimplexTopology)
    VectorTopology(iszero(bundle(m)) ? 0 : (global top_id+=1),m)
end
SimplexTopology(m::VectorTopology) = m.t

for fun ∈ (:totalelements,:elements,:subelements,:istotal,:isfull,:iscover,:fullimmersion,:vertices,:fullvertices)
    @eval $fun(m::VectorTopology) = $fun(SimplexTopology(m))
end
bundle(m::VectorTopology) = m.id
fulltopology(m::VectorTopology) = getindex.(Ref(m),OneTo(totalelements(m)))
topology(m::VectorTopology) = collect(m)
totalnodes(m::VectorTopology{N}) where N = (N-1)*totalnodes(SimplexTopology(m))
nodes(m::VectorTopology{N}) where N = (N-1)*nodes(SimplexTopology(m))

isdiscontinuous(m::VectorTopology) = false
isdisconnected(m::VectorTopology) = false

Base.size(m::VectorTopology) = size(subelements(m))
Base.length(m::VectorTopology) = elements(m)
Base.axes(m::VectorTopology) = axes(subelements(m))
function Base.getindex(m::VectorTopology{3},i::Int)
    ti = 2SimplexTopology(m)[i]
    return vcat(ti.-1,ti)
end
Grassmann.mdims(m::VectorTopology{N}) where N = N

getimage(m::VectorTopology,i) = getimage(SimplexTopology(m),i)
getfacet(m::VectorTopology,i) = getfacet(SimplexTopology(m),i)

function Base.getindex(m::VectorTopology,i::AbstractVector{Int})
    VectorTopology(bundle(m),SimplexTopology(m)[i])#,fullvertices(m))
end

(m::VectorTopology)(i::AbstractVector{Int}) = subtopology(m,i)
function subtopology(m::VectorTopology{N},i::AbstractVector{Int}) where N
    VectorTopology(bundle(m),subtopology(SimplexTopology(m),i))
end

getelement(m::VectorTopology{N,<:SimplexTopology{N,<:OneTo}} where N,i::Int) = m[i]
function getelement(m::VectorTopology{N,<:SimplexTopology{N,<:AbstractVector}} where N,i::Int)
    if iscover(m)
        m[i]
    else
        ti = 2SimplexTopology(m)[i]
        vcat(ti.-1,ti)
    end
end

subtopology(m::VectorTopology{N,<:SimplexTopology{N,<:OneTo}} where N) = topology(m)
function subtopology(m::VectorTopology{N,<:SimplexTopology{N,<:AbstractVector}} where N)
    iscover(m) ? topology(m) : getelement.(Ref(m),OneTo(elements(m)))
end

function subimmersion(m::VectorTopology)
    iscover(m) ? m : VectorTopology(subimmersion(SimplexTopology(m)))
end

refine(m::VectorTopology) = VectorTopology(bundle(m),refine(SimplexTopology(m)))=#

# Common

_axes(t::ImmersedTopology{N}) where N = (Base.OneTo(length(t)),Base.OneTo(N))

for top ∈ (:SimplexTopology,:DiscontinuousTopology)#,:VectorTopology)
    @eval begin
        # anything array-like gets summarized e.g. 10-element Array{Int64,1}
        Base.summary(io::IO, a::$top) = Base.array_summary(io, a, _axes(a))
        function Base.array_summary(io::IO, a::$top, inds::Tuple{Vararg{OneTo}})
            print(io, Base.dims2string(length.(inds)))
            print(io, iscover(a) ? "⊆" : "⊂", totalnodes(a), " ")
            Base.showarg(io, a, true)
        end
    end
end

