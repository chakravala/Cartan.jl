
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

export FiberProduct, FiberProductBundle, HomotopyBundle, resample
export LocalSection, GlobalFiber, LocalFiber, localfiber, globalfiber
export base, fiber, domain, codomain, ↦, →, ←, ↤, basetype, fibertype, graph
export ProductSpace, RealRegion, NumberLine, Rectangle, Hyperrectangle, ⧺, ⊕

resize_lastdim!(x::Vector,i) = resize!(x,i)

# ProductSpace

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
RealRegion(v::Values{N,S}) where {T<:Real,N,S<:AbstractVector{T}} = ProductSpace{Submanifold(N),T,N}(v)
ProductSpace{V}(v::Values{N,S}) where {V,T<:Real,N,S<:AbstractVector{T}} = ProductSpace{V,T,N}(v)
ProductSpace(v::Values{N,S}) where {T<:Real,N,S<:AbstractVector{T}} = ProductSpace{Submanifold(N),T,N}(v)

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

# QuotientTopology

struct QuotientTopology{N,L,M,O,LA<:ImmersedTopology{L,L}} <: ImmersedTopology{N,N}
    p::Values{O,Int}
    q::Values{O,LA}
    r::Values{M,Int}
    s::Values{N,Int}
    #t::Values{O,Int}
    #QuotientTopology(p::Values{O,Int},q::Values{O,LA},r::Values{M,Int},n::Values{N,Int}) where {O,L,LA<:ImmersedTopology{L,L},M,N} = QuotientTopology{N,L,M,O,LA}(p,q,r,n)
end

#QuotientTopology(p::Values{O,Int},q::Values{O,LA},r::Values{M,Int},n::Values{N,Int}) where {O,L,LA<:ImmersedTopology{L,L},M,N} = QuotientTopology{N,L,M,O,LA}(p,q,r,n,invert_q(Val(O),r))

invert_q(s::Int,i::Int) = iszero(s) ? () : (i,)
invert_q(::Val{M},s::Values{M,Int}) where M = s
invert_q(::Val{0},s::Values{M,Int}) where M = Values{0,Int}()
@generated invert_q(::Val{O},s::Values{M,Int}) where {O,M} = Expr(:call,:(Values{O,Int}),[:(invert_q((@inbounds s[$i]),$i)...) for i ∈ list(1,M)]...)

const OpenTopology{N,L,M,LA} = QuotientTopology{N,L,M,0,LA}
const CompactTopology{N,L,M,LA} = QuotientTopology{N,L,M,M,LA}
QuotientTopology(n::ProductTopology) = OpenTopology(n.v)
OpenTopology(n::ProductTopology) = OpenTopology(n.v)
OpenTopology(n::QuotientTopology) = OpenTopology(size(n))
OpenTopology(n::Values{N,Int}) where N = QuotientTopology(Values{0,Int}(),Values{0,Array{Values{N-1,Int},N-1}}(),zeros(Values{2N,Int}),n)
RibbonTopology(n::Values{2,Int}) = QuotientTopology(Values(2,1),Values(ProductTopology(n[2]),ProductTopology(n[2])),Values(1,2,0,0),n)
MobiusTopology(n::Values{2,Int}) = QuotientTopology(Values(2,1),Values(ProductTopology(n[2]:-1:1),ProductTopology(n[2]:-1:1)),Values(1,2,0,0),n)
WingTopology(n::Values{2,Int}) = QuotientTopology(Values(1,2),Values(ProductTopology(n[2]:-1:1),ProductTopology(n[2]:-1:1)),Values(1,2,0,0),n)
MirrorTopology(n::Values{1,Int}) = QuotientTopology(Values((1,)),Array{Values{0,Int},0}.(Values((undef,))),Values(1,0),n)
MirrorTopology(n::Values{2,Int}) = QuotientTopology(Values((1,)),Values((ProductTopology(n[2]),)),Values(1,0,0,0),n)
MirrorTopology(n::Values{3,Int}) = QuotientTopology(Values((1,)),Values((ProductTopology(n[2],n[3]),)),Values(1,0,0,0,0,0),n)
MirrorTopology(n::Values{4,Int}) = QuotientTopology(Values((1,)),Values((ProductTopology(n[2],n[3],n[4]),)),Values(1,0,0,0,0,0,0,0),n)
MirrorTopology(n::Values{5,Int}) = QuotientTopology(Values((1,)),Values((ProductTopology(n[2],n[3],n[4],n[5]),)),Values(1,0,0,0,0,0,0,0,0,0),n)
ClampedTopology(n::Values{1,Int}) = QuotientTopology(Values(1,2),Array{Values{0,Int},0}.(Values(undef,undef)),Values(1,2),n)
ClampedTopology(n::Values{2,Int}) = QuotientTopology(Values(1,2,3,4),Values(ProductTopology(n[2]),ProductTopology(n[2]),ProductTopology(n[1]),ProductTopology(n[1])),Values(1,2,3,4),n)
ClampedTopology(n::Values{3,Int}) = QuotientTopology(Values(1,2,3,4,5,6),Values(ProductTopology(n[2],n[3]),ProductTopology(n[2],n[3]),ProductTopology(n[1],n[3]),ProductTopology(n[1],n[3]),ProductTopology(n[1],n[2]),ProductTopology(n[1],n[2])),Values(1,2,3,4,5,6),n)
ClampedTopology(n::Values{4,Int}) = QuotientTopology(Values(1,2,3,4,5,6,7,8),Values(ProductTopology(n[2],n[3],n[4]),ProductTopology(n[2],n[3],n[4]),ProductTopology(n[1],n[3],n[4]),ProductTopology(n[1],n[3],n[4]),ProductTopology(n[1],n[2],n[4]),ProductTopology(n[1],n[2],n[4]),ProductTopology(n[1],n[2],n[3]),ProductTopology(n[1],n[2],n[3])),Values(1,2,3,4,5,6,7,8),n)
ClampedTopology(n::Values{5,Int}) = QuotientTopology(Values(1,2,3,4,5,6,7,8,9,10),Values(ProductTopology(n[2],n[3],n[4],n[5]),ProductTopology(n[2],n[3],n[4],n[5]),ProductTopology(n[1],n[3],n[4],n[5]),ProductTopology(n[1],n[3],n[4],n[5]),ProductTopology(n[1],n[2],n[4],n[5]),ProductTopology(n[1],n[2],n[4],n[5]),ProductTopology(n[1],n[2],n[3],n[5]),ProductTopology(n[1],n[2],n[3],n[5]),ProductTopology(n[1],n[2],n[3],n[4]),ProductTopology(n[1],n[2],n[3],n[4])),Values(1,2,3,4,5,6,7,8,9,10),n)
TorusTopology(n::Values{1,Int}) = QuotientTopology(Values(2,1),Array{Values{0,Int},0}.(Values(undef,undef)),Values(1,2),n)
TorusTopology(n::Values{2,Int}) = QuotientTopology(Values(2,1,4,3),Values(ProductTopology(n[2]),ProductTopology(n[2]),ProductTopology(n[1]),ProductTopology(n[1])),Values(1,2,3,4),n)
TorusTopology(n::Values{3,Int}) = QuotientTopology(Values(2,1,4,3,6,5),Values(ProductTopology(n[2],n[3]),ProductTopology(n[2],n[3]),ProductTopology(n[1],n[3]),ProductTopology(n[1],n[3]),ProductTopology(n[1],n[2]),ProductTopology(n[1],n[2])),Values(1,2,3,4,5,6),n)
TorusTopology(n::Values{4,Int}) = QuotientTopology(Values(2,1,4,3,6,5,8,7),Values(ProductTopology(n[2],n[3],n[4]),ProductTopology(n[2],n[3],n[4]),ProductTopology(n[1],n[3],n[4]),ProductTopology(n[1],n[3],n[4]),ProductTopology(n[1],n[2],n[4]),ProductTopology(n[1],n[2],n[4]),ProductTopology(n[1],n[2],n[3]),ProductTopology(n[1],n[2],n[3])),Values(1,2,3,4,5,6,7,8),n)
TorusTopology(n::Values{5,Int}) = QuotientTopology(Values(2,1,4,3,6,5,8,7,10,9),Values(ProductTopology(n[2],n[3],n[4],n[5]),ProductTopology(n[2],n[3],n[4],n[5]),ProductTopology(n[1],n[3],n[4],n[5]),ProductTopology(n[1],n[3],n[4],n[5]),ProductTopology(n[1],n[2],n[4],n[5]),ProductTopology(n[1],n[2],n[4],n[5]),ProductTopology(n[1],n[2],n[3],n[5]),ProductTopology(n[1],n[2],n[3],n[5]),ProductTopology(n[1],n[2],n[3],n[4]),ProductTopology(n[1],n[2],n[3],n[4])),Values(1,2,3,4,5,6,7,8,9,10),n)
HopfTopology(n::Values{2,Int}) = QuotientTopology(Values(2,1,4,3),Values(ProductTopology(CrossRange(n[2])),ProductTopology(CrossRange(n[2])),ProductTopology(n[1]),ProductTopology(n[1])),Values(1,2,3,4),n)
HopfTopology(n::Values{3,Int}) = QuotientTopology(Values(2,1,4,3,6,5),Values(ProductTopology(OneTo(n[2]),CrossRange(n[3])),ProductTopology(OneTo(n[2]),CrossRange(n[3])),ProductTopology(OneTo(n[1]),CrossRange(n[3])),ProductTopology(OneTo(n[1]),CrossRange(n[3])),ProductTopology(n[1],n[2]),ProductTopology(n[1],n[2])),Values(1,2,3,4,5,6),n)
KleinTopology(n::Values{2,Int}) = QuotientTopology(Values(2,1,4,3),Values(ProductTopology(n[2]:-1:1),ProductTopology(n[2]:-1:1),ProductTopology(1:1:n[1]),ProductTopology(1:1:n[1])),Values(1,2,3,4),n)
ConeTopology(n::Values{2,Int}) = QuotientTopology(Values(1,4,3),Values(ProductTopology(CrossRange(n[2])),ProductTopology(n[1]),ProductTopology(n[1])),Values(1,0,2,3),n)
PolarTopology(n::Values{2,Int}) = QuotientTopology(Values(1,2,4,3),Values(ProductTopology(CrossRange(n[2])),ProductTopology(n[2]),ProductTopology(n[1]),ProductTopology(n[1])),Values(1,2,3,4),n)
SphereTopology(n::Values{1,Int}) = TorusTopology(n)
SphereTopology(n::Values{2,Int}) = QuotientTopology(Values(1,2,4,3),Values(ProductTopology(CrossRange(n[2])),ProductTopology(CrossRange(n[2])),ProductTopology(n[1]),ProductTopology(n[1])),Values(1,2,3,4),n)
SphereTopology(n::Values{3,Int}) = QuotientTopology(Values(1,2,3,4,6,5),Values(ProductTopology(OneTo(n[2]),CrossRange(n[3])),ProductTopology(OneTo(n[2]),CrossRange(n[3])),ProductTopology(OneTo(n[1]),CrossRange(n[3])),ProductTopology(OneTo(n[1]),CrossRange(n[3])),ProductTopology(n[1],n[2]),ProductTopology(n[1],n[2])),Values(1,2,3,4,5,6),n)
SphereTopology(n::Values{4,Int}) = QuotientTopology(Values(1,2,3,4,5,6,8,7),Values(ProductTopology(OneTo(n[2]),OneTo(n[3]),CrossRange(n[4])),ProductTopology(OneTo(n[2]),OneTo(n[3]),CrossRange(n[4])),ProductTopology(OneTo(n[1]),OneTo(n[3]),CrossRange(n[4])),ProductTopology(OneTo(n[1]),OneTo(n[3]),CrossRange(n[4])),ProductTopology(OneTo(n[1]),OneTo(n[2]),CrossRange(n[4])),ProductTopology(OneTo(n[1]),OneTo(n[2]),CrossRange(n[4])),ProductTopology(n[1],n[2],n[3]),ProductTopology(n[1],n[2],n[3])),Values(1,2,3,4,5,6,7,8),n)
SphereTopology(n::Values{5,Int}) = QuotientTopology(Values(1,2,3,4,5,6,7,8,10,9),Values(ProductTopology(OneTo(n[2]),OneTo(n[3]),OneTo(n[4]),CrossRange(n[5])),ProductTopology(OneTo(n[2]),OneTo(n[3]),OneTo(n[4]),CrossRange(n[5])),ProductTopology(OneTo(n[1]),OneTo(n[3]),OneTo(n[4]),CrossRange(n[5])),ProductTopology(OneTo(n[1]),OneTo(n[3]),OneTo(n[4]),CrossRange(n[5])),ProductTopology(OneTo(n[1]),OneTo(n[2]),OneTo(n[4]),CrossRange(n[5])),ProductTopology(OneTo(n[1]),OneTo(n[2]),OneTo(n[4]),CrossRange(n[5])),ProductTopology(OneTo(n[1]),OneTo(n[2]),OneTo(n[3]),CrossRange(n[5])),ProductTopology(OneTo(n[1]),OneTo(n[2]),OneTo(n[3]),CrossRange(n[5])),ProductTopology(n[1],n[2],n[3],n[4]),ProductTopology(n[1],n[2],n[3],n[4])),Values(1,2,3,4,5,6,7,8,9,10),n)
GeographicTopology(n::Values{2,Int}) = QuotientTopology(Values(2,1,3,4),Values(ProductTopology(n[2]),ProductTopology(n[2]),ProductTopology(CrossRange(n[1])),ProductTopology(CrossRange(n[1]))),Values(1,2,3,4),n)

OpenParameter(n::ProductTopology) = OpenParameter(n.v)
OpenParameter(n::Values{1,Int}) = OpenTopology(PointArray(0,LinRange(0,1,n[1])))
OpenParameter(n::Values{2,Int}) = OpenTopology(LinRange(0,1,n[1])⊕LinRange(0,1,n[2]))
OpenParameter(n::Values{3,Int}) = OpenTopology(LinRange(0,1,n[1])⊕LinRange(0,1,n[2])⊕LinRange(0,1,n[3]))
OpenParameter(n::Values{4,Int}) = OpenTopology(LinRange(0,1,n[1])⊕LinRange(0,1,n[2])⊕LinRange(0,1,n[3])⊕LinRange(0,1,n[4]))
OpenParameter(n::Values{5,Int}) = OpenTopology(LinRange(0,1,n[1])⊕LinRange(0,1,n[2])⊕LinRange(0,1,n[3])⊕LinRange(0,1,n[4])⊕LinRange(0,1,n[5]))
RibbonParameter(n::Values{2,Int}) = RibbonTopology(LinRange(0,2π,n[1])⊕LinRange(-1,1,n[2]))
MobiusParameter(n::Values{2,Int}) = MobiusTopology(LinRange(0,2π,n[1])⊕LinRange(-1,1,n[2]))
WingParameter(n::Values{2,Int}) = WingParameter(LinRange(0,1,n[1])⊕LinRange(-1,1,n[2]))
MirrorParameter(n::Values{1,Int}) = MirrorTopology(PointArray(0,LinRange(0,2π,n[1])))
MirrorParameter(n::Values{2,Int}) = MirrorTopology(LinRange(0,2π,n[1])⊕LinRange(0,1,n[2]))
MirrorParameter(n::Values{3,Int}) = MirrorTopology(LinRange(0,2π,n[1])⊕LinRange(0,1,n[2])⊕LinRange(0,1,n[3]))
MirrorParameter(n::Values{4,Int}) = MirrorTopology(LinRange(0,2π,n[1])⊕LinRange(0,1,n[2])⊕LinRange(0,1,n[3])⊕LinRange(0,1,n[4]))
MirrorParameter(n::Values{5,Int}) = MirrorTopology(LinRange(0,2π,n[1])⊕LinRange(0,1,n[2])⊕LinRange(0,1,n[3])⊕LinRange(0,1,n[4])⊕LinRange(0,1,n[5]))
ClampedParameter(n::Values{1,Int}) = ClampedTopology(PointArray(0,LinRange(0,2π,n[1])))
ClampedParameter(n::Values{2,Int}) = ClampedTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2]))
ClampedParameter(n::Values{3,Int}) = ClampedTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2])⊕LinRange(0,2π,n[3]))
ClampedParameter(n::Values{4,Int}) = ClampedTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2])⊕LinRange(0,2π,n[3])⊕LinRange(0,2π,n[4]))
ClampedParameter(n::Values{5,Int}) = ClampedTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2])⊕LinRange(0,2π,n[3])⊕LinRange(0,2π,n[4])⊕LinRange(0,2π,n[5]))
TorusParameter(n::Values{1,Int}) = TorusTopology(PointArray(0,LinRange(0,2π,n[1])))
TorusParameter(n::Values{2,Int}) = TorusTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2]))
TorusParameter(n::Values{3,Int}) = TorusTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2])⊕LinRange(0,2π,n[3]))
TorusParameter(n::Values{4,Int}) = TorusTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2])⊕LinRange(0,2π,n[3])⊕LinRange(0,2π,n[4]))
TorusParameter(n::Values{5,Int}) = TorusTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2])⊕LinRange(0,2π,n[3])⊕LinRange(0,2π,n[4])⊕LinRange(0,2π,n[5]))
HopfParameter(n::Values{2,Int}) = HopfTopology(LinRange(0,2π,n[2])⊕LinRange(0,4π,n[3]))
HopfParameter(n::Values{3,Int}) = HopfTopology(LinRange(7π/16/n[1],7π/16,n[1])⊕LinRange(0,2π,n[2])⊕LinRange(0,4π,n[3]))
KleinParameter(n::Values{2,Int}) = KleinTopology(LinRange(0,2π,n[1])⊕LinRange(0,2π,n[2]))
ConeParameter(n::Values{2,Int}) = ConeTopology(LinRange(0,1,n[1])⊕LinRange(0,2π,n[2]))
PolarParameter(n::Values{2,Int}) = PolarTopology(LinRange(0,1,n[1])⊕LinRange(0,2π,n[2]))
SphereParameter(n::Values{1,Int}) = TorusParameter(n)
SphereParameter(n::Values{2,Int}) = SphereTopology(LinRange(0,π,n[1])⊕LinRange(0,2π,n[2]))
SphereParameter(n::Values{3,Int}) = SphereTopology(LinRange(0,π,n[1])⊕LinRange(0,π,n[2])⊕LinRange(0,2π,n[3]))
SphereParameter(n::Values{4,Int}) = SphereTopology(LinRange(0,π,n[1])⊕LinRange(0,π,n[2])⊕LinRange(0,π,n[3])⊕LinRange(0,2π,n[4]))
SphereParameter(n::Values{5,Int}) = SphereTopology(LinRange(0,π,n[1])⊕LinRange(0,π,n[2])⊕LinRange(0,π,n[3])⊕LinRange(0,π,n[4])⊕LinRange(0,2π,n[5]))
GeographicParameter(n::Values{2,Int}) = GeographicTopology(LinRange(-π,π,n[1])⊕LinRange(-π/2,π/2,n[2]))

for fun ∈ (:Open,:Ribbon,:Mobius,:Wing,:Mirror,:Clamped,:Torus,:Hopf,:Klein,:Cone,:Polar,:Sphere,:Geographic)
    for typ ∈ (Symbol(fun,:Topology),Symbol(fun,:Parameter))
        @eval begin
            export $typ
            $typ(p::ProductSpace) = $typ(PointArray(p))
            $typ(p::Values{N,<:AbstractVector} where N) = $typ(ProductSpace(p))
            $typ(p::T...) where T<:AbstractVector = $typ(ProductSpace(Values(p)))
            $typ(n::NTuple) = $typ(Values(n))
        end
    end
end
for mod ∈ (:Topology,:Parameter)
    for fun ∈ (:Hopf,)
        for typ ∈ (Symbol(fun,mod),)
            @eval begin
                $typ(n::Int...) = $typ(Values(n...))
                $typ() = $typ(Values(7,60,61))
            end
        end
    end
    for fun ∈ (:Open,:Mirror,:Clamped,:Torus)
        for typ ∈ (Symbol(fun,mod),)
            @eval begin
                $typ() = $typ(60,60)
                $typ(n::Int...) = $typ(Values(n...))
            end
        end
    end
    for (fun,n,m) ∈ ((:Ribbon,60,20),(:Wing,60,20),(:Mobius,60,20),(:Klein,60,60),(:Cone,30,:(2n+1)),(:Polar,30,:(2n)),(:Sphere,30,:(2n+1)),(:Geographic,61,:(n÷2)))
        for typ ∈ (Symbol(fun,mod),)
            @eval begin
                $typ(n=$n,m=$m) = $typ(Values(n,m))
            end
        end
    end
end

isopen(t::QuotientTopology) = false
isopen(t::OpenTopology) = true
iscompact(t::QuotientTopology) = false
iscompact(t::CompactTopology) = true
_to_axis(f::Int) = (iseven(f) ? f : f+1)÷2

zeroprodtop(r,n) = iszero(r) ? () : (ProductTopology(n),)
LinearAlgebra.cross(m::OpenTopology,n::OpenTopology) = OpenTopology(m.s...,n.s...)
LinearAlgebra.cross(m::OpenTopology{1},n::OpenTopology{1}) = OpenTopology(m.s...,n.s...)
function LinearAlgebra.cross(m::QuotientTopology{1},n::QuotientTopology{1})
    M,N = m.s[1],n.s[1]
    QuotientTopology(Values(m.p...,(n.p.+2)...),
        Values((zeroprodtop(m.r[1],N)...,zeroprodtop(m.r[2],N)...,zeroprodtop(n.r[1],M)...,zeroprodtop(n.r[2],M)...)),
        Values(m.r...,iszero(n.r[1]) ? 0 : n.r[1]+length(m.p),iszero(n.r[2]) ? 0 : n.r[2]+length(m.p)),Values(M,N))
end
LinearAlgebra.cross(m::OpenTopology,n::Int) = OpenTopology(m.s...,n)
LinearAlgebra.cross(m::OpenTopology{1},n::Int) = OpenTopology(m.s...,n)
function LinearAlgebra.cross(m::QuotientTopology{1},n::Int)
    QuotientTopology(m.p,
        Values((zeroprodtop(m.r[1],n)...,zeroprodtop(m.r[2],n)...)),
        Values(m.r...,0,0),Values(m.s...,n))
end
function LinearAlgebra.cross(m::QuotientTopology,n::Int)
    QuotientTopology(m.p,m.q .× n,Values(m.r...,0,0),Values(m.s...,n))
end

getlocate(i) = Values((i,))
getlocate(a,i) = Values((i,))
getlocate(a,i,j) = isone(a) ? Values(i,j) : Values(j,i)
getlocate(a,i,j,k) = isone(a) ? Values(i,j,k) : (a==2) ? Values(j,i,k) : Values(j,k,i)
getlocate(a,i,j,k,l) = isone(a) ? Values(i,j,k,l) : (a==2) ? Values(j,i,k,l) : (a==3) ? Values(j,k,i,l) : Values(j,k,l,i)
getlocate(a,i,j,k,l,o) = isone(a) ? Values(i,j,k,l,o) : (a==2) ? Values(j,i,k,l,o) : (a==3) ? Values(j,k,i,l,o) : (a==4) ? Values(j,k,l,i,o) : Values(j,k,l,o,i)

function locate_fast(pr1::Int,s,i::Int)
    if isodd(pr1)
        return abs(i-1)+1
    else
        return @inbounds s[_to_axis(pr1)]-abs(i-1)
    end
end
function locate_fast(pr2::Int,n::Int,s,i::Int)
    if iseven(pr2)
        return @inbounds s[_to_axis(pr2)]+n-i
    else
        return (i+1)-n
    end
end
function locate(pr1::Int,a::Int,s,i::Int)
    if isodd(pr1)
        return abs(i-1)+1
    else
        return @inbounds s[a]-abs(i-1)
    end
end
function locate(pr2::Int,a::Int,n::Int,s,i::Int)
    if iseven(pr2)
        return @inbounds s[a]+n-i
    else
        return (i+1)-n
    end
end

function location(p,q,r::Int,s::Tuple,i::Int,jk::Int...)
    pr = @inbounds p[r]
    a = _to_axis(pr)
    getlocate(a,locate(pr,a,s,i),(@inbounds q[r])[jk...]...)
end
function location(p,q,r::Int,n::Int,s::Tuple,i::Int,jk::Int...)
    pr = @inbounds p[r]
    a = _to_axis(pr)
    getlocate(a,locate(pr,a,n,s,i),(@inbounds q[r])[jk...]...)
end

@generated function resize(m::OpenTopology{N},i) where N
    Expr(:call,:QuotientTopology,:(m.p),:(m.q),:(m.r),
         Expr(:call,:Values,[j≠N ? :(@inbounds m.s[$j]) : :i for j ∈ list(1,N)]...))
end
@generated function resize(m::QuotientTopology{N,L,M,O},i) where {N,L,M,O}
    Expr(:call,:QuotientTopology,:(m.p),
         Expr(:call,:Values,Expr(:tuple,[:((@inbounds m.r[$j])∉(@inbounds m.r[2N-1],@inbounds m.r[2N]) ? (@inbounds resize(m.q[$j],i)) : (@inbounds m.q[$j])) for j ∈ list(1,O)]...)),
         :(m.r),
         Expr(:call,:Values,[j≠N ? :(@inbounds m.s[$j]) : :i for j ∈ list(1,N)]...))
end
@generated function resize(m::QuotientTopology{N,L,O,O},i) where {N,L,O}
    Expr(:call,:QuotientTopology,:(m.p),
         Expr(:call,:Values,[j∉(O-1,O) ? :(@inbounds resize(m.q[$j],i)) : :(@inbounds m.q[$j]) for j ∈ list(1,O)]...),
         :(m.r),
         Expr(:call,:Values,[j≠N ? :(@inbounds m.s[$j]) : :i for j ∈ list(1,N)]...))
end

resample(m::QuotientTopology{N,L,M,0},i::NTuple{N}) where {N,L,M} = QuotientTopology(m.p,m.q,m.r,Values(i))
@generated function resample(m::QuotientTopology{N,L,O,O},i::NTuple{N}) where {N,L,O}
    Expr(:block,:(perms = $(reverse(Values{L}.(Values{N}(Grassmann.combo(N,L)))))),
        Expr(:call,:QuotientTopology,:(m.p),
            Expr(:call,:Values,Expr(:tuple,[:(@inbounds resample(m.q[$j],i[perms[$((j+1)÷2)]])) for j ∈ list(1,O)]...)),
            :(m.r),Expr(:call,:Values,:i)))
end
@generated function resample(m::QuotientTopology{N,L,M,O},i::NTuple{N}) where {N,L,M,O}
    Expr(:block,:(t = invert_q($(Val(O)),m.r)),
        :(perms = $(reverse(Values{L}.(Values{N}(Grassmann.combo(N,L)))))),
        Expr(:call,:QuotientTopology,:(m.p),
            Expr(:call,:Values,Expr(:tuple,[:(@inbounds resample(m.q[$j],i[perms[t[$((j+1)÷2)]]])) for j ∈ list(1,O)]...)),
            :(m.r),Expr(:call,:Values,:i)))
end

@pure Base.eltype(::Type{<:QuotientTopology{N}}) where N = Values{N,Int}
Base.size(m::QuotientTopology) = m.s.v
Base.iterate(t::QuotientTopology) = (getindex(t,1),1)
Base.iterate(t::QuotientTopology,state) = (s=state+1; s≤length(t) ? (getindex(t,s),s) : nothing)

function Base.getindex(m::QuotientTopology{N},i::Vararg{Int,N}) where N
    N > 5 ? Values{N,Int}(i) : getindex(m,Val(0),i...)
end
function Base.getindex(m::QuotientTopology{1},i::Int)
    s = size(m)
    n = @inbounds s[1]
    ii = if (i > 1 && i < n)
        i
    elseif i < 2
        r = @inbounds m.r[1]
        iszero(r) ? i : locate_fast((@inbounds m.p[r]),s,i)
    else
        r = @inbounds m.r[2]
        iszero(r) ? i : locate_fast((@inbounds m.p[r]),n,s,i)
    end
    return Values{1,Int}((ii,))
end

bounds(i,n,::Val{N},::Val{M}) where {N,M} = (i > (N∈(0,M) ? 1 : 0) && (N∈(0,M) ? (<) : (≤))(i,n))

Base.getindex(m::QuotientTopology{N},::Val,i::Vararg{Int,N}) where N = getindex(m,i...)
Base.getindex(m::QuotientTopology{1},::Val,i::Int) = getindex(m,i)
function Base.getindex(m::QuotientTopology{2},N::Val,i::Int,j::Int)
    s = size(m)
    n1,n2 = @inbounds (s[1],s[2])
    isi,isj = (bounds(i,n1,N,Val(1)),bounds(j,n2,N,Val(2)))
    if isj && !isi
        if i < 2
            r = @inbounds m.r[1]
            !iszero(r) && (return location(m.p,m.q,r,s,i,j))
        else
            r = @inbounds m.r[2]
            !iszero(r) && (return location(m.p,m.q,r,n1,s,i,j))
        end
    elseif isi && !isj
        if j < 2
            r = @inbounds m.r[3]
            !iszero(r) && (return location(m.p,m.q,r,s,j,i))
        else
            r = @inbounds m.r[4]
            !iszero(r) && (return location(m.p,m.q,r,n2,s,j,i))
        end
    end
    return Values(i,j)
end
function Base.getindex(m::QuotientTopology{3},N::Val,i::Int,j::Int,k::Int)
    s = size(m)
    n1,n2,n3 = @inbounds (s[1],s[2],s[3])
    isi,isj,isk = (bounds(i,n1,N,Val(1)),bounds(j,n2,N,Val(2)),bounds(k,n3,N,Val(3)))
    if isj && isk && !isi
        if i < 2
            r = @inbounds m.r[1]
            !iszero(r) && (return location(m.p,m.q,r,s,i,j,k))
        else
            r = @inbounds m.r[2]
            !iszero(r) && (return location(m.p,m.q,r,n1,s,i,j,k))
        end
    elseif isi && isk && !isj
        if j < 2
            r = @inbounds m.r[3]
            !iszero(r) && (return location(m.p,m.q,r,s,j,i,k))
        else
            r = @inbounds m.r[4]
            !iszero(r) && (return location(m.p,m.q,r,n2,s,j,i,k))
        end
    elseif isi && isj && !isk
        if k < 2
            r = @inbounds m.r[5]
            !iszero(r) && (return location(m.p,m.q,r,s,k,i,j))
        else
            r = @inbounds m.r[6]
            !iszero(r) && (return location(m.p,m.q,r,n3,s,k,i,j))
        end
    end
    return Values(i,j,k)
end
function Base.getindex(m::QuotientTopology{4},N::Val,i::Int,j::Int,k::Int,l::Int)
    s = size(m)
    n1,n2,n3,n4 = @inbounds (s[1],s[2],s[3],s[4])
    isi,isj,isk,isl = (bounds(i,n1,N,Val(1)),bounds(j,n2,N,Val(2)),bounds(k,n3,N,Val(3)),bounds(l,n4,N,Val(4)))
    if isj && isk && isl && !isi
        if i < 2
            r = @inbounds m.r[1]
            !iszero(r) && (return location(m.p,m.q,r,s,i,j,k,l))
        else
            r = @inbounds m.r[2]
            !iszero(r) && (return location(m.p,m.q,r,n1,s,i,j,k,l))
        end
    elseif isi && isk && isl && !isj
        if j < 2
            r = @inbounds m.r[3]
            !iszero(r) && (return location(m.p,m.q,r,s,j,i,k,l))
        else
            r = @inbounds m.r[4]
            !iszero(r) && (return location(m.p,m.q,r,n2,s,j,i,k,l))
        end
    elseif isi && isj && isl && !isk
        if k < 2
            r = @inbounds m.r[5]
            !iszero(r) && (return location(m.p,m.q,r,s,k,i,j,l))
        else
            r = @inbounds m.r[6]
            !iszero(r) && (return location(m.p,m.q,r,n3,s,k,i,j,l))
        end
    elseif isi && isj && isk && !isl
        if l < 2
            r = @inbounds m.r[7]
            !iszero(r) && (return location(m.p,m.q,r,s,l,i,j,k))
        else
            r = @inbounds m.r[8]
            !iszero(r) && (return location(m.p,m.q,r,n4,s,l,i,j,k))
        end
    end
    return Values(i,j,k,l)
end
function Base.getindex(m::QuotientTopology{5},N::Val,i::Int,j::Int,k::Int,l::Int,o::Int)
    s = size(m)
    n1,n2,n3,n4,n5 = @inbounds (s[1],s[2],s[3],s[4],s[5])
    isi,isj,isk,isl,iso = (bounds(i,n1,N,Val(1)),bounds(j,n2,N,Val(2)),bounds(k,n3,N,Val(3)),bounds(l,n4,N,Val(4)),bounds(o,n5,N,Val(5)))
    if isj && isk && isl && iso && !isi
        if i < 2
            r = @inbounds m.r[1]
            !iszero(r) && (return location(m.p,m.q,r,s,i,j,k,l,o))
        else
            r = @inbounds m.r[2]
            !iszero(r) && (return location(m.p,m.q,r,n1,s,i,j,k,l,o))
        end
    elseif isi && isk && isl && iso && !isj
        if j < 2
            r = @inbounds m.r[3]
            !iszero(r) && (return location(m.p,m.q,r,s,j,i,k,l,o))
        else
            r = @inbounds m.r[4]
            !iszero(r) && (return location(m.p,m.q,r,n2,s,j,i,k,l,o))
        end
    elseif isi && isj && isl && iso && !isk
        if k < 2
            r = @inbounds m.r[5]
            !iszero(r) && (return location(m.p,m.q,r,s,k,i,j,l,o))
        else
            r = @inbounds m.r[6]
            !iszero(r) && (return location(m.p,m.q,r,n3,s,k,i,j,l,o))
        end
    elseif isi && isj && isk && iso && !isl
        if l < 2
            r = @inbounds m.r[7]
            !iszero(r) && (return location(m.p,m.q,r,s,l,i,j,k,o))
        else
            r = @inbounds m.r[8]
            !iszero(r) && (return location(m.p,m.q,r,n4,s,l,i,j,k,o))
        end
    elseif isi && isj && isk && isl && !iso
        if o < 2
            r = @inbounds m.r[9]
            !iszero(r) && (return location(m.p,m.q,r,s,o,i,j,k,l))
        else
            r = @inbounds m.r[10]
            !iszero(r) && (return location(m.p,m.q,r,n4,s,o,i,j,k,l))
        end
    end
    return Values(i,j,k,l,o)
end

function findface(m,r,i,vals)
    ri = r[i]
    if iszero(ri) || m.p[ri] ∉ r || m.q[ri][vals...] ≠ vals
        0
    else
        m.p[ri]≠(@inbounds r[1]) ? 2 : 1
    end
end
function findface(m,r,i,vals,ex...)
    ri = r[i]
    if iszero(ri) || m.p[ri] ∉ r || exclude(m.q[ri],ex...)[vals...] ≠ vals
        0
    else
        pri = m.p[ri]
        findfirst(x->x==pri,r)
    end
end

subtopology(m::OpenTopology,i::NTuple{N,Int},::Colon,j::NTuple{M,Int} where M) where N = OpenTopology(size(m)[N+1])
subtopology(m::OpenTopology,i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int} where L) where {N,M} = OpenTopology(size(m)[N+1],size(m)[N+M+2])
subtopology(m::OpenTopology,i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int},::Colon,l::NTuple{O,Int} where O) where {N,M,L} = OpenTopology(size(m)[N+1],size(m)[N+M+2],size(m)[N+M+L+3])
subtopology(m::QuotientTopology{1},i::NTuple{N,Int},::Colon,j::NTuple{M,Int} where M) where N = m
subtopology(m::QuotientTopology{2},i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int} where L) where {N,M} = m
subtopology(m::QuotientTopology{3},i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int},::Colon,l::NTuple{O,Int} where O) where {N,M,L} = m
subtopology(m::QuotientTopology{4},i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int},::Colon,l::NTuple{O,Int},::Colon,o::NTuple{Y,Int} where Y) where {N,M,L,O} = m
subtopology(m::QuotientTopology{5},i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int},::Colon,l::NTuple{O,Int},::Colon,o::NTuple{Y,Int},::Colon,z::NTuple{Z,Int} where Z) where {N,M,L,O,Y} = m
function subtopology(m::QuotientTopology{M},::Val{N}) where {M,N}
    r1,r2 = m.r[2N-1],m.r[2N]
    r1z,r2z,n = iszero(r1),iszero(r2),size(m)[N]
    if r1z
        if r2z
            OpenTopology(n)
        else
            QuotientTopology(Values((isodd(m.p[r2]) ? 1 : 2,)),Array{Values{0,Int},0}.(Values((undef,))),Values(0,1),Values((n,)))
        end
    elseif r2z && !r1z
        QuotientTopology(Values((isodd(m.p[r1]) ? 1 : 2,)),Array{Values{0,Int},0}.(Values((undef,))),Values(1,0),Values((n,)))
    else
        QuotientTopology(Values(isodd(m.p[r1]) ? 1 : 2,isodd(m.p[r2]) ? 1 : 2),Array{Values{0,Int},0}.(Values((undef,undef))),Values(1,2),Values((n,)))
    end
end
function subtopology(m::QuotientTopology,i::NTuple{N,Int},::Colon,j::NTuple{M,Int} where M) where N
    N1 = N+1
    r,vals = m.r[Values(2N1-1,2N1)],Values(i...,j...)
    p1,p2 = findface(m,r,1,vals),findface(m,r,2,vals)
    p1z,p2z,n = iszero(p1),iszero(p2),size(m)[N1]
    if p1z
        if p2z
            OpenTopology(n)
        else
            QuotientTopology(Values((2,)),Array{Values{0,Int},0}.(Values((undef,))),Values(0,1),Values((n,)))
        end
    elseif p2z && !p1z
        MirrorTopology(n)
    else
        QuotientTopology(Values(p1,p2),Array{Values{0,Int},0}.(Values((undef,undef))),Values(1,2),Values((n,)))
    end
end
function subtopology(m::QuotientTopology,i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int} where L) where {N,M}
    N1,M1,s = N+1,N+M+2,size(m)
    r,vals = m.r[Values(2N1-1,2N1,2M1-1,2M1)],Values(i...,j...,k...)
    (p1,p2,p3,p4) = (
        findface(m,r,1,vals,Val(M1-1)),findface(m,r,2,vals,Val(M1-1)),
        findface(m,r,3,vals,Val(N1)),findface(m,r,4,vals,Val(N1)))
    p1z,p2z,p3z,p4z = iszero.((p1,p2,p3,p4))
    pz = !(p1z)+!(p2z)+!(p3z)+!(p4z)
    vpz = iszero(pz) ? Values{0} : isone(pz) ? Values{1} : pz==2 ? Values{2} : pz==3 ? Values{3} : Values{4}
    a = iszero(pz) ? Values{0,Int}() : vpz((p1z ? () : (p1,))...,(p2z ? () : (p2,))...,(p3z ? () : (p3,))...,(p4z ? () : (p4,))...)
    b = if iszero(pz)
        Values{0,Array{Values{1,Int},1}}()
    else; vpz(
        (p1z ? () : (ProductTopology(m.q[r[1]].v[M1-1]),))...,
        (p2z ? () : (ProductTopology(m.q[r[2]].v[M1-1]),))...,
        (p3z ? () : (ProductTopology(m.q[r[3]].v[N1]),))...,
        (p4z ? () : (ProductTopology(m.q[r[4]].v[N1]),))...)
    end
    c = Values{4,Int}(p1z ? 0 : 1,p2z ? 0 : !(p1z)+!(p2z),p3z ? 0 : !(p1z)+!(p2z)+!(p3z),p4z ? 0 : pz)
    #e = iszero(pz) ? Values{0,Int} : vpz((p1z ? () : (1,))...,(p2z ? () : (!(p1z)+!(p2z),))...,(p3z ? () : (!(p1z)+!(p2z)+!(p3z),))...,(p4z ? () : (pz,))...)
    QuotientTopology(a,b,c,Values(s[N1],s[M1]))
end
function subtopology(m::QuotientTopology,i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int},::Colon,l::NTuple{O,Int} where O) where {N,M,L}
    N1,M1,L1,s = N+1,N+M+2,N+M+L+3,size(m)
    r,vals = m.r[Values(2N1-1,2N1,2M1-1,2M1,2L1-1,2L1)],Values(i...,j...,k...,l...)
    (p1,p2,p3,p4,p5,p6) = (
        findface(m,r,1,vals,Val(M1-1),Val(L1-1)),findface(m,r,2,vals,Val(M1-1),Val(L1-1)),
        findface(m,r,3,vals,Val(N1),Val(L1-1)),findface(m,r,4,vals,Val(N1),Val(L1-1)),
        findface(m,r,5,vals,Val(N1),Val(M1)),findface(m,r,6,vals,Val(N1),Val(M1)))
    p1z,p2z,p3z,p4z,p5z,p6z = iszero.((p1,p2,p3,p4,p5,p6))
    pz = !(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z)+!(p6z)
    vpz = iszero(pz) ? Values{0} : isone(pz) ? Values{1} : pz==2 ? Values{2} : pz==3 ? Values{3} : pz==4 ? Values{4} : pz==5 ? Values{5} : Values{6}
    a = iszero(pz) ? Values{0,Int}() : vpz((p1z ? () : (p1,))...,(p2z ? () : (p2,))...,(p3z ? () : (p3,))...,(p4z ? () : (p4,))...,(p5z ? () : (p5,))...,(p6z ? () : (p6,))...)
    b = if iszero(pz)
        Values{0,Array{Values{2,Int},2}}()
    else; vpz(
        (p1z ? () : (ProductTopology(m.q[r[1]].v[Values(M1-1,L1-1)]),))...,
        (p2z ? () : (ProductTopology(m.q[r[2]].v[Values(M1-1,L1-1)]),))...,
        (p3z ? () : (ProductTopology(m.q[r[3]].v[Values(N1,L1-1)]),))...,
        (p4z ? () : (ProductTopology(m.q[r[4]].v[Values(N1,L1-1)]),))...,
        (p5z ? () : (ProductTopology(m.q[r[5]].v[Values(N1,M1)]),))...,
        (p6z ? () : (ProductTopology(m.q[r[6]].v[Values(N1,M1)]),))...)
    end
    c = Values(p1z ? 0 : 1,p2z ? 0 : !(p1z)+!(p2z),p3z ? 0 : !(p1z)+!(p2z)+!(p3z),p4z ? 0 : !(p1z)+!(p2z)+!(p3z)+!(p4z),p5z ? 0 : !(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z),p6z ? 0 : pz)
    #e = iszero(pz) ? Values{0,Int} : vpz((p1z ? () : (1,))...,(p2z ? () : (!(p1z)+!(p2z),))...,(p3z ? () : (!(p1z)+!(p2z)+!(p3z),))...,(p4z ? () : (!(p1z)+!(p2z)+!(p3z)+!(p4z),))...,(p5z ? () : (!(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z),))...,(p6z ? () : (pz,))...)
    QuotientTopology(a,b,c,Values(s[N1],s[M1],s[L1]))
end
function subtopology(m::QuotientTopology,i::NTuple{N,Int},::Colon,j::NTuple{M,Int},::Colon,k::NTuple{L,Int},::Colon,l::NTuple{O,Int},::Colon,h::NTuple{H,Int} where H) where {N,M,L,O}
    N1,M1,L1,O1,s = N+1,N+M+2,N+M+L+3,N+M+L+O+4,size(m)
    r,vals = m.r[Values(2N1-1,2N1,2M1-1,2M1,2L1-1,2L1,2O1-1,2O1)],Values(i...,j...,k...,l...,h...)
    (p1,p2,p3,p4,p5,p6,p7,p8) = (
        findface(m,r,1,vals,Val(M1-1),Val(L1-1),Val(O1-1)),findface(m,r,2,vals,Val(M1-1),Val(L1-1),Val(O1-1)),
        findface(m,r,3,vals,Val(N1),Val(L1-1),Val(O1-1)),findface(m,r,4,vals,Val(N1),Val(L1-1),Val(O1-1)),
        findface(m,r,5,vals,Val(N1),Val(M1),Val(O1-1)),findface(m,r,6,vals,Val(N1),Val(M1),Val(O1-1)),
        findface(m,r,7,vals,Val(N1),Val(M1),Val(L1)),findface(m,r,8,vals,Val(N1),Val(M1),Val(L1)))
    p1z,p2z,p3z,p4z,p5z,p6z,p7z,p8z = iszero.((p1,p2,p3,p4,p5,p6,p7,p8))
    pz = !(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z)+!(p6z)+!(p7z)+!(p8z)
    vpz = iszero(pz) ? Values{0} : isone(pz) ? Values{1} : pz==2 ? Values{2} : pz==3 ? Values{3} : pz==4 ? Values{4} : pz==5 ? Values{5} : pz==6 ? Values{6} : pz==7 ? Values{7} : Values{8}
    a = iszero(pz) ? Values{0,Int}() : vpz((p1z ? () : (p1,))...,(p2z ? () : (p2,))...,(p3z ? () : (p3,))...,(p4z ? () : (p4,))...,(p5z ? () : (p5,))...,(p6z ? () : (p6,))...,(p7z ? () : (p7,))...,(p8z ? () : (p8,))...)
    b = if iszero(pz)
        Values{0,Array{Values{3,Int},3}}()
    else; vpz(
        (p1z ? () : (ProductTopology(m.q[r[1]].v[Values(M1-1,L1-1,O1-1)]),))...,
        (p2z ? () : (ProductTopology(m.q[r[2]].v[Values(M1-1,L1-1,O1-1)]),))...,
        (p3z ? () : (ProductTopology(m.q[r[3]].v[Values(N1,L1-1,O1-1)]),))...,
        (p4z ? () : (ProductTopology(m.q[r[4]].v[Values(N1,L1-1,O1-1)]),))...,
        (p5z ? () : (ProductTopology(m.q[r[5]].v[Values(N1,M1,O1-1)]),))...,
        (p6z ? () : (ProductTopology(m.q[r[6]].v[Values(N1,M1,O1-1)]),))...,
        (p7z ? () : (ProductTopology(m.q[r[7]].v[Values(N1,M1,L1)]),))...,
        (p8z ? () : (ProductTopology(m.q[r[8]].v[Values(N1,M1,L1)]),))...)
    end
    c = Values(p1z ? 0 : 1,p2z ? 0 : !(p1z)+!(p2z),p3z ? 0 : !(p1z)+!(p2z)+!(p3z),p4z ? 0 : !(p1z)+!(p2z)+!(p3z)+!(p4z),p5z ? 0 : !(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z),p6z ? 0 : !(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z)+!(p6z),p7z ? 0 : !(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z)+!(p6z)+!(p7z),p8z ? 0 : pz)
    #e = iszero(pz) ? Values{0,Int} : vpz((p1z ? () : (1,))...,(p2z ? () : (!(p1z)+!(p2z),))...,(p3z ? () : (!(p1z)+!(p2z)+!(p3z),))...,(p4z ? () : (!(p1z)+!(p2z)+!(p3z)+!(p4z),))...,(p5z ? () : (!(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z),))...,(p6z ? () : (!(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z)+!(p6z),))...,(p7z ? () : (!(p1z)+!(p2z)+!(p3z)+!(p4z)+!(p5z)+!(p6z)+!(p7z),))...,(p8z ? () : (pz,))...)
    QuotientTopology(a,b,c,Values(s[N1],s[M1],s[L1],s[O1]))
end

# 1
(m::QuotientTopology)(c::Colon,i::Int...) = subtopology(m,(),c,i)
(m::QuotientTopology)(i::Int,c::Colon,j::Int...) = subtopology(m,(i,),c,j)
(m::QuotientTopology)(i::Int,j::Int,c::Colon,k::Int...) = subtopology(m,(i,j),c,k)
(m::QuotientTopology)(i::Int,j::Int,k::Int,c::Colon,l::Int...) = subtopology(m,(i,j,k),c,l)
(m::QuotientTopology)(i::Int,j::Int,k::Int,l::Int,c::Colon,h::Int...) = subtopology(m,(i,j,k,l),c,h)

# 2 - 0
(m::QuotientTopology)(c::Colon,::Colon,i::Int...) = subtopology(m,(),c,(),c,i)
(m::QuotientTopology)(c::Colon,i::Int,::Colon,j::Int...) = subtopology(m,(),c,(i,),c,j)
(m::QuotientTopology)(c::Colon,i::Int,j::Int,::Colon,k::Int...) = subtopology(m,(),c,(i,j),c,k)
(m::QuotientTopology)(c::Colon,i::Int,j::Int,k::Int,::Colon,l::Int...) = subtopology(m,(),c,(i,j,k),c,l)
# 2 - 1
(m::QuotientTopology)(i::Int,c::Colon,::Colon,j::Int...) = subtopology(m,(i,),c,(),c,j)
(m::QuotientTopology)(i::Int,c::Colon,j::Int,::Colon,k::Int...) = subtopology(m,(i,),c,(j,),c,k)
(m::QuotientTopology)(i::Int,c::Colon,j::Int,k::Int,::Colon,l::Int...) = subtopology(m,(i,),c,(j,k),c,l)
# 2 - 2
(m::QuotientTopology)(i::Int,j::Int,c::Colon,::Colon,k::Int...) = subtopology(m,(i,j),c,(),c,k)
(m::QuotientTopology)(i::Int,j::Int,c::Colon,k::Int,::Colon,l::Int...) = subtopology(m,(i,j),c,(k,),c,l)
# 2 - 3
(m::QuotientTopology)(i::Int,j::Int,k::Int,c::Colon,::Colon,l::Int...) = subtopology(m,(i,j,k),c,(),c,l)

# 3 - 0 - 0
(m::QuotientTopology)(c::Colon,::Colon,::Colon,i::Int...) = subtopology(m,(),c,(),c,(),c,i)
(m::QuotientTopology)(c::Colon,::Colon,i::Int,::Colon,j::Int...) = subtopology(m,(),c,(),c,(i,),c,j)
(m::QuotientTopology)(c::Colon,::Colon,i::Int,j::Int,::Colon,k::Int...) = subtopology(m,(),c,(),c,(i,j),c,k)
# 3 - 0 - 1
(m::QuotientTopology)(c::Colon,i::Int,::Colon,::Colon,j::Int...) = subtopology(m,(),c,(i,),c,(),c,j)
(m::QuotientTopology)(c::Colon,i::Int,::Colon,j::Int,::Colon,k::Int...) = subtopology(m,(),c,(i,),c,(j,),c,k)
# 3 - 0 - 2
(m::QuotientTopology)(c::Colon,i::Int,j::Int,::Colon,::Colon,k::Int...) = subtopology(m,(),c,(i,j),c,(),c,k)
# 3 - 1
(m::QuotientTopology)(i::Int,c::Colon,::Colon,::Colon,j::Int...) = subtopology(m,(i,),c,(),c,(),c,j)
(m::QuotientTopology)(i::Int,c::Colon,j::Int,::Colon,::Colon,k::Int...) = subtopology(m,(i,),c,(j,),c,(),c,k)
(m::QuotientTopology)(i::Int,c::Colon,::Colon,j::Int,::Colon,k::Int...) = subtopology(m,(i,),c,(),c,(j,),c,k)
# 3 - 2
(m::QuotientTopology)(i::Int,j::Int,c::Colon,::Colon,::Colon,k::Int...) = subtopology(m,(i,j),c,(),c,(),c,k)

# 4
(m::QuotientTopology)(c::Colon,::Colon,::Colon,::Colon,i::Int...) = subtopology(m,(),c,(),c,(),c,(),c,i)
(m::QuotientTopology)(c::Colon,::Colon,::Colon,i::Int,::Colon,j::Int...) = subtopology(m,(),c,(),c,(),c,(i,),c,j)
(m::QuotientTopology)(c::Colon,::Colon,i::Int,::Colon,::Colon,j::Int...) = subtopology(m,(),c,(),c,(i,),c,(),c,j)
(m::QuotientTopology)(c::Colon,i::Int,::Colon,::Colon,::Colon,j::Int...) = subtopology(m,(),c,(i,),c,(),c,(),c,j)
(m::QuotientTopology)(i::Int,c::Colon,::Colon,::Colon,::Colon,j::Int...) = subtopology(m,(i,),c,(),c,(),c,(),c,j)

# SimplexTopology

top_id = 0

struct SimplexTopology{N,P<:AbstractVector{Int}} <: ImmersedTopology{N,1}
    id::Int
    t::Vector{Values{N,Int}}
    i::P
    p::Int
    SimplexTopology(t::Vector{Values{N,Int}},i::P=vertices(t),p::Int=length(i)) where {N,P} = new{N,P}((global top_id+=1),t,i,p)
end
const SimplexManifold = SimplexTopology

SimplexTopology(t::Vector{Values{N,Int}},p::Int) where N = SimplexTopology(t,vertices(t),p)

bundle(m::SimplexTopology) = m.id
topology(m::SimplexTopology) = m.t
vertices(m::SimplexTopology) = m.i

Base.size(m::SimplexTopology) = size(topology(m))
Base.length(m::SimplexTopology) = length(topology(m))
Base.axes(m::SimplexTopology) = axes(topology(m))
Base.getindex(m::SimplexTopology,i::Int) = getindex(topology(m),i)
@pure Base.eltype(::Type{ImmersedTopology{N}}) where N = Values{N,Int}
Grassmann.mdims(m::SimplexTopology{N}) where N = N

_axes(t::SimplexTopology{N}) where N = (Base.OneTo(length(t)),Base.OneTo(N))

# anything array-like gets summarized e.g. 10-element Array{Int64,1}
Base.summary(io::IO, a::SimplexTopology) = Base.array_summary(io, a, _axes(a))

iscover(x::ImmersedTopology) = length(vertices(x)) == x.p
subsym(x) = iscover(x) ? "⊆" : "⊂"

function Base.array_summary(io::IO, a::SimplexTopology, inds::Tuple{Vararg{OneTo}})
    print(io, Base.dims2string(length.(inds)))
    print(io, subsym(a), length(vertices(a)), " ")
    Base.showarg(io, a, true)
end

# Global

export Global

struct Global{N,T} <: AbstractArray{T,N}
    v::T
    #n::NTuple{N,Int}
    #Global{N}(v::T,n=(1,)) where {T,N} = new{N,T}(v,n)
    Global{N}(v::T) where {T,N} = new{N,T}(v)
end

#Base.size(m::Global) = m.n
Base.getindex(m::Global,i::Vararg{Int}) = m.v
Base.getindex(t::Global,i::CartesianIndex) = m.v
Base.resize!(t::Global,i) = t
@pure Base.eltype(::Type{<:Global{T}}) where T = T

Base.IndexStyle(::Global) = IndexCartesian()
function Base.getindex(A::Global, I::Int)
    Base.@_inline_meta
    A.v
end

Base.show(io::IO,t::Global{N}) where N = print(io,"Global{$N}($(t.v))")
Base.show(io::IO, ::MIME"text/plain", t::Global) = show(io,t)

#metricextensor(c::AbstractArray{T,N} where T) where N = Global{N}(InducedMetric(),size(c))
ref(itr::InducedMetric) = Ref(itr)
ref(itr::Global) = Ref(itr.v)
ref(itr) = itr
refmetric(x) = ref(metricextensor(x))

# LocalFiber

abstract type LocalFiber{B,F} <: Number end
Base.@pure isfiber(::LocalFiber) = true
Base.@pure isfiber(::Any) = false

fiber(s) = s
fibertype(s) = typeof(s)
fibertype(::Type{T}) where T = T
base(s::LocalFiber) = s.v.first
fiber(s::LocalFiber) = s.v.second
basepoint(s::LocalFiber) = point(base(s))
basetype(::LocalFiber{B}) where B = B
basepointtype(::LocalFiber{B}) where B = pointtype(B)
fibertype(::LocalFiber{B,F} where B) where F = F
basetype(::Type{<:LocalFiber{B}}) where B = B
fibertype(::Type{<:LocalFiber{B,F} where B}) where F = F

Base.getindex(s::LocalFiber) = s.v.first
Base.getindex(s::LocalFiber,i::Int...) = getindex(s.v.second,i...)
Base.getindex(s::LocalFiber,i::Integer...) = getindex(s.v.second,i...)

function Base.show(io::IO, s::LocalFiber)
    fibertype(s) <: InducedMetric && (return show(io, base(s)))
    p = s.v
    Base.isdelimited(io, p) && return show_pairtyped(io, s)
    typeinfos = Base.gettypeinfos(io, p)
    for i = (1, 2)
        io_i = IOContext(io, :typeinfo => typeinfos[i])
        Base.isdelimited(io_i, p[i]) || print(io, "(")
        show(io_i, p[i])
        Base.isdelimited(io_i, p[i]) || print(io, ")")
        i == 1 && print(io, get(io, :compact, false)::Bool ? "↦" : " ↦ ")
    end
end

function show_pairtyped(io::IO, s::LocalFiber{B,F}) where {B,F}
    show(io, typeof(s))
    show(io, (base(s), fiber(s)))
end

# Coordinate

export Coordinate, point

struct Coordinate{P,G} <: LocalFiber{P,G}
    v::Pair{P,G}
    Coordinate(v::Pair{P,G}) where {P,G} = new{P,G}(v)
    Coordinate(p::P,g::G) where {P,G} = new{P,G}(p=>g)
end

point(c) = c
point(c::Coordinate) = base(c)
point(c::LocalFiber) = point(base(c))
metricextensor(c) = InducedMetric()
metricextensor(c::Coordinate) = fiber(c)
metrictensor(c) = InducedMetric()
metrictensor(c::Coordinate) = TensorOperator(fiber(c)[1])

Base.getindex(s::Coordinate,i::Int...) = getindex(s.v.first,i...)
Base.getindex(s::Coordinate,i::Integer...) = getindex(s.v.first,i...)

graph(s::LocalFiber{<:AbstractReal,<:AbstractReal}) = Chain(Real(base(s)),Real(fiber(s)))
graph(s::LocalFiber{<:Chain,<:AbstractReal}) = Chain(value(base(s))...,Real(fiber(s)))
graph(s::LocalFiber{<:Coordinate{<:AbstractReal},<:AbstractReal}) = Chain(Real(basepoint(s)),Real(fiber(s)))
graph(s::LocalFiber{<:Coordinate{<:Chain},<:AbstractReal}) = Chain(value(basepoint(s))...,Real(fiber(s)))

export Positions, Interval, RealSpace, ComplexSpace
const Positions{P<:Chain,G} = AbstractVector{<:Coordinate{P,G}}
const Interval{P<:AbstractReal,G} = AbstractVector{<:Coordinate{P,G}}
#const RectanglePatch{P,G} = AbstractMatrix{<:Coordinate{P,G}}
#const HyperrectanglePatch{P,G} = AbstractArray{<:Coordinate{P,G},3}
const RealSpace{N,P<:Chain{V,1,<:Real} where V,G} = AbstractArray{<:Coordinate{P,G},N}
const ComplexSpace{N,P<:Chain{V,1,<:Complex} where V,G} = AbstractArray{<:Coordinate{P,G},N}
#const RectanglePatch{P,G} = RealSpace{2,P,G}
#const HyperrectanglePatch{P,G} = RealSpace{3,P,G}

# LocalSection

struct LocalSection{B,F} <: LocalFiber{B,F}
    v::Pair{B,F}
    LocalSection(v::Pair{B,F}) where {B,F} = new{B,F}(v)
    LocalSection(b::B,f::F) where {B,F} = new{B,F}(b=>f)
    LocalSection(b::B,f::LocalSection{R,F} where R) where {B,F} = new{B,F}(b=>f.v.second)
    LocalSection(b::LocalSection{B,R} where R,f::F) where {B,F} = new{B,F}(base(b)=>f)
end

# LocalTensor

struct LocalTensor{B,F} <: LocalFiber{B,F}
    v::Pair{B,F}
    LocalTensor(v::Pair{B,F}) where {B,F} = new{B,F}(v)
    LocalTensor(b::B,f::F) where {B,F} = new{B,F}(b=>f)
    LocalTensor(b::B,f::LocalTensor{R,F} where R) where {B,F} = new{B,F}(b=>f.v.second)
    LocalTensor(b::LocalTensor{B,R} where R,f::F) where {B,F} = new{B,F}(base(b)=>f)
end

export Section
const Section = LocalTensor
const ↦, domain, codomain = LocalTensor, base, fiber
↤(F,B) = B ↦ F

localfiber(x) = x
localfiber(x::LocalTensor) = fiber(x)
localfiber(x::LocalSection) = fiber(x)

(m::TensorNested)(x::LocalTensor) = LocalTensor(base(x),m(fiber(x)))
@inline Base.:<<(a::LocalFiber,b::LocalFiber) = contraction(b,~a)
@inline Base.:>>(a::LocalFiber,b::LocalFiber) = contraction(~a,b)
@inline Base.:<(a::LocalFiber,b::LocalFiber) = contraction(b,a)
Base.sign(s::LocalTensor) = LocalTensor(base(s),sign(Real(fiber(s))))
Base.inv(a::LocalTensor{B,<:Real} where B) = LocalTensor(base(a), inv(fiber(a)))
Base.inv(a::LocalTensor{B,<:Complex} where B) = LocalTensor(base(a), inv(fiber(a)))
Base.:/(a::LocalTensor,b::LocalTensor{B,<:Real} where B) = LocalTensor(base(a), fiber(a)/fiber(b))
Base.:/(a::LocalTensor,b::LocalTensor{B,<:Complex} where B) = LocalTensor(base(a), fiber(a)/fiber(b))
LinearAlgebra.:×(a::LocalTensor{R},b::LocalTensor{R}) where R = TensorField(base(a), ⋆(fiber(a)∧fiber(b),metricextensor(base(a))))
Grassmann.compound(t::LocalTensor,i::Val) = LocalTensor(base(t), compound(fiber(t),i))
Grassmann.compound(t::LocalTensor,i::Int) = LocalTensor(base(t), compound(fiber(t),i))
Grassmann.eigen(t::LocalTensor,i::Val) = LocalTensor(base(t), eigen(fiber(t),i))
Grassmann.eigen(t::LocalTensor,i::Int) = LocalTensor(base(t), eigen(fiber(t),i))
Grassmann.eigvals(t::LocalTensor,i::Val) = LocalTensor(base(t), eigvals(fiber(t),i))
Grassmann.eigvals(t::LocalTensor,i::Int) = LocalTensor(base(t), eigvals(fiber(t),i))
Grassmann.eigvecs(t::LocalTensor,i::Val) = LocalTensor(base(t), eigvecs(fiber(t),i))
Grassmann.eigvecs(t::LocalTensor,i::Int) = LocalTensor(base(t), eigvecs(fiber(t),i))
Grassmann.eigpolys(t::LocalTensor,G::Val) = LocalTensor(base(t), eigpolys(fiber(t),G))
for type ∈ (:Coordinate,:LocalSection,:LocalTensor)
    for tensor ∈ (:Single,:Couple,:PseudoCouple,:Chain,:Spinor,:AntiSpinor,:Multivector,:DiagonalOperator,:TensorOperator,:Outermorphism)
        @eval (T::Type{<:$tensor})(s::$type) = $type(base(s), T(fiber(s)))
    end
    for fun ∈ (:-,:!,:~,:real,:imag,:conj,:deg2rad,:transpose)
        @eval Base.$fun(s::$type) = $type(base(s), $fun(fiber(s)))
    end
    for fun ∈ (:inv,:exp,:exp2,:exp10,:log,:log2,:log10,:sinh,:cosh,:abs,:sqrt,:cbrt,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:asech,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:cis,:abs2)
        @eval Base.$fun(s::$type) = $type(base(s), $fun(fiber(s),metricextensor(base(s))))
    end
    for fun ∈ (:reverse,:involute,:clifford,:even,:odd,:scalar,:vector,:bivector,:volume,:value,:curl,:∂,:d,:complementleft,:realvalue,:imagvalue,:outermorphism,:Outermorphism,:DiagonalOperator,:TensorOperator,:eigen,:eigvecs,:eigvals,:eigvalsreal,:eigvalscomplex,:eigvecsreal,:eigvecscomplex,:eigpolys,:∧)
        @eval Grassmann.$fun(s::$type) = $type(base(s), $fun(fiber(s)))
    end
    for fun ∈ (:⋆,:angle,:radius,:complementlefthodge,:pseudoabs,:pseudoabs2,:pseudoexp,:pseudolog,:pseudoinv,:pseudosqrt,:pseudocbrt,:pseudocos,:pseudosin,:pseudotan,:pseudocosh,:pseudosinh,:pseudotanh,:metric,:unit)
        @eval Grassmann.$fun(s::$type) = $type(base(s), $fun(fiber(s),metricextensor(base(s))))
    end
    for op ∈ (:+,:-,:&,:∧,:∨)
        let bop = op ∈ (:∧,:∨) ? :(Grassmann.$op) : :(Base.$op)
        @eval begin
            $bop(a::$type{R},b::$type{R}) where R = $type(base(a),$op(fiber(a),fiber(b)))
            $bop(a::Number,b::$type) = $type(base(b), $op(a,fiber(b)))
            $bop(a::$type,b::Number) = $type(base(a), $op(fiber(a),b))
        end end
    end
    for (op,mop) ∈ ((:*,:wedgedot_metric),(:wedgedot,:wedgedot_metric),(:veedot,:veedot_metric),(:⋅,:contraction_metric),(:>,:contraction_metric),(:⊘,:⊘),(:>>>,:>>>),(:/,:/),(:^,:^))
        let bop = op ∈ (:*,:>,:>>>,:/,:^) ? :(Base.$op) : :(Grassmann.$op)
        @eval begin
            $bop(a::$type{R},b::$type{R}) where R = $type(base(a),Grassmann.$mop(fiber(a),fiber(b),metricextensor(base(a))))
            $bop(a::Number,b::$type) = $type(base(b), Grassmann.$op(a,fiber(b)))
            $bop(a::$type,b::Number) = $type(base(a), Grassmann.$op(fiber(a),b,$((op≠:^ ? () : (:(metricextensor(base(a))),))...)))
        end end
    end
    @eval begin
        $type(b,f::Function) = $type(b,f(b))
        Grassmann.contraction(a::$type{R},b::$type{R}) where R = $type(base(a),Grassmann.contraction(fiber(a),fiber(b)))
        LinearAlgebra.norm(s::$type) = $type(base(s), norm(fiber(s)))
        LinearAlgebra.det(s::$type) = $type(base(s), det(fiber(s)))
        LinearAlgebra.tr(s::$type) = $type(base(s), tr(fiber(s)))
        (V::Submanifold)(s::$type) = $type(base(a), V(fiber(s)))
        (::Type{T})(s::$type) where T<:Real = $type(base(s), T(fiber(s)))
        (::Type{Complex})(s::$type) = $type(base(s), Complex(fiber(s)))
        (::Type{Complex{T}})(s::$type) where T = $type(base(s), Complex{T}(fiber(s)))
        Grassmann.Phasor(s::$type) = $type(base(s), Phasor(fiber(s)))
        Grassmann.Couple(s::$type) = $type(base(s), Couple(fiber(s)))
        (::Type{T})(s::$type...) where T<:Chain = @inbounds $type(base(s[1]), Chain(fiber.(s)...))
    end
end

# FiberBundle

abstract type FiberBundle{T,N} <: AbstractArray{T,N} end

base(t::FiberBundle) = t.dom
fiber(t::FiberBundle) = t.cod
base(t::Array) = ProductSpace(Values(axes(t)))
fiber(t::Array) = t
basetype(::Array{T}) where T = T
fibertype(::Array{T}) where T = T

# PointCloud

export PointArray, PointVector, PointMatrix, PointCloud

point_id = 0

struct PointArray{P,G,N,PA<:AbstractArray{P,N},GA<:AbstractArray{G,N}} <: FiberBundle{Coordinate{P,G},N}
    id::Int
    dom::PA
    cod::GA
    PointArray(id::Int,p::PA,g::GA) where {P,G,N,PA<:AbstractArray{P,N},GA<:AbstractArray{G,N}} = new{P,G,N,PA,GA}(id,p,g)
end

const PointVector{P,G,PA,GA} = PointArray{P,G,1,PA,GA}
const PointMatrix{P,G,PA,GA} = PointArray{P,G,2,PA,GA}
const PointCloud = PointVector

PointArray(id::Int,dom::AbstractArray{T,N} where T) where N = PointArray(id,dom,Global{N}(InducedMetric()))
PointArray(dom::AbstractArray{T,N} where T) where N = PointArray(0,dom,Global{N}(InducedMetric()))
PointArray(dom::AbstractVector) = PointCloud(dom)
PointArray(p::P,g::G) where {P<:AbstractArray,G<:AbstractArray} = PointArray(0,p,g)
#PointArray(p::P,g::G) where {P<:AbstractVector,G<:AbstractVector} = PointCloud(p,g)

PointArray(dom::PointArray,fun) = PointArray(base(dom), fun)
PointArray(dom::PointArray,fun::Array) = PointArray(base(dom), fun)
PointArray(dom::PointArray,fun::Function) = PointArray(base(dom), fun)
PointArray(dom::AbstractArray,fun::Function) = PointArray(dom, fun.(dom))

submetric(x::DiagonalOperator) = DiagonalOperator(getindex(x,1))
submetric(x::Outermorphism) = TensorOperator(getindex(x,1))

points(m::PointArray) = m.dom
metricextensor(m::PointArray) = m.cod
metrictensor(m::PointArray) = submetric.(metricextensor(m))
pointtype(m::PointArray) = basetype(m)
pointtype(m::Type{<:PointArray}) = basetype(m)
metrictype(m::PointArray) = fibertype(m)
metrictype(m::Type{<:PointArray}) = fibertype(m)
basetype(::PointArray{B}) where B = B
basetype(::Type{<:PointArray{B}}) where B = B
fibertype(::PointArray{B,F} where B) where F = F
fibertype(::Type{<:PointArray{B,F} where B}) where F = F
isinduced(::Array) = false
isinduced(::Global) = false
isinduced(::Global{N,<:InducedMetric} where N) = true
isinduced(p::PointArray) = isinduced(metricextensor(p))

@pure Grassmann.Manifold(m::PointArray) = Manifold(points(m))
@pure LinearAlgebra.rank(m::PointArray) = rank(points(m))
@pure Grassmann.grade(m::PointArray) = grade(points(m))
@pure Grassmann.antigrade(m::PointArray) = antigrade(points(m))
@pure Grassmann.mdims(m::PointArray) = mdims(points(m))

PointCloud(id::Int,p::PA,g::GA) where {P,G,PA<:AbstractVector{P},GA<:AbstractVector{G}} = PointArray(id,p,g)
PointCloud(id::Int,dom) = PointArray(id,dom)
PointCloud(dom::AbstractVector) = PointCloud(dom,Global{1}(InducedMetric()))

const point_cache = (Vector{Chain{V,G,T,X}} where {V,G,T,X})[]
const point_metric_cache = (AbstractVector{T} where T)[]

PointCloud(m::Int) = PointCloud(m,point_cache[m],point_metric_cache[m])
function PointCloud(p::P,g::G) where {P<:AbstractVector,G<:AbstractVector}
    push!(point_cache,p)
    push!(point_metric_cache,g)
    PointCloud(length(point_cache),p,g)
end
function clearpointcache!()
    for P ∈ 1:length(point_cache)
        deletebundle!(P)
    end
end
bundle(m::PointCloud) = m.id
deletebundle!(m::PointCloud) = deletepointcloud!(bundle(m))
function deletepointcloud!(P::Int)
    point_cache[P] = [Chain{Submanifold(0),0,Int}(Values(0))]
    point_metric_cache[P] = [Chain{Submanifold(0),0,Int}(Values(0))]
    nothing
end

⊕(a::PointArray,b::PointArray) = PointArray(points(a)⊕points(b))
⊕(a::PointArray,b::AbstractVector{<:Real}) = PointArray(points(a)⊕b)
cross(a::PointArray,b::PointArray) = a⊕b
cross(a::PointArray,b::AbstractVector{<:Real}) = a⊕b

Base.size(m::PointArray) = size(m.dom)
Base.firstindex(m::PointCloud) = 1
Base.lastindex(m::PointCloud) = length(points(m))
Base.length(m::PointCloud) = length(points(m))
Base.resize!(m::PointCloud,i::Int) = ((resize!(points(m),i),resize!(metricextensor(m),i)); m)
Base.broadcast(f,t::PointArray) = PointArray(f.(points(t)),f.(metricextensor(t)))
Base.broadcast(f,t::PointCloud) = PointCloud(f.(points(t)),f.(metricextensor(t)))
resize_lastdim!(m::Global,i) = m
resize_lastdim!(m::PointArray,i) = ((resize_lastdim!(m.dom,i),resize_lastdim!(m.cod,i)); m)

function (m::PointArray)(i::Vararg{Union{Int,Colon}})
    pa = points(m)(i...)
    ga = if isinduced(m)
        Global{ndims(pa)}(InducedMetric())
    else
        error("missing functionality")
    end
    return PointArray(0,pa,ga)
end

@pure Base.eltype(::Type{<:PointArray{P,G}}) where {P,G} = Coordinate{P,G}
function Base.getindex(m::PointArray,i::Vararg{Int})
    Coordinate(getindex(points(m),i...), getindex(metricextensor(m),i...))
end
function Base.getindex(m::PointArray,i::Vararg{Union{Int,Colon}})
    pa = getindex(points(m),i...)
    ga = if isinduced(m)
        Global{ndims(pa)}(InducedMetric())
    else
        getindex(metricextensor(m),i...)
    end
    return PointArray(0,pa,ga)
end
Base.setindex!(m::PointArray{P},s::P,i::Vararg{Int}) where P = setindex!(points(m),s,i...)
Base.setindex!(m::PointArray{P,G} where P,s::G,i::Vararg{Int}) where G = setindex!(metricextensor(m),s,i...)
function Base.setindex!(m::PointArray,s::Coordinate,i::Vararg{Int})
    setindex!(points(m),point(s),i...)
    setindex!(metricextensor(m),metricextensor(s),i...)
    return s
end

Base.BroadcastStyle(::Type{<:PointArray{P,G,N,PA,GA}}) where {P,G,N,PA,GA} = Broadcast.ArrayStyle{PointArray{P,G,N,PA,GA}}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{PointArray{P,G,N,PA,GA}}}, ::Type{ElType}) where {P,G,N,PA,GA,ElType<:Coordinate}
    ax = axes(bc)
    # Use the data type to create the output
    PointArray(similar(Array{pointtype(ElType),N}, ax), similar(Array{metrictype(ElType),N}, ax))
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{PointArray{P,G,N,PA,GA}}}, ::Type{ElType}) where {P,G,N,PA,GA,ElType}
    t = find_pa(bc)
    # Use the data type to create the output
    PointArray(similar(Array{ElType,N}, axes(bc)), metricextensor(t))
end

"`A = find_pa(As)` returns the first PointArray among the arguments."
find_pa(bc::Base.Broadcast.Broadcasted) = find_pa(bc.args)
find_pa(bc::Base.Broadcast.Extruded) = find_pa(bc.x)
find_pa(args::Tuple) = find_pa(find_pa(args[1]), Base.tail(args))
find_pa(x) = x
find_pa(::Tuple{}) = nothing
find_pa(a::PointArray, rest) = a
find_pa(::Any, rest) = find_pa(rest)

# FiberProduct

export FiberProduct

struct FiberProduct{P,N,PA<:AbstractArray{F,N} where F,FA<:AbstractArray} <: FiberBundle{Coordinate{P,InducedMetric},N}
    dom::PA
    cod::FA
    FiberProduct{P}(p::PA,f::FA) where {P,N,PA<:AbstractArray{F,N} where F,FA<:AbstractArray} = new{P,N,PA,FA}(p,f)
end

points(m::FiberProduct) = m.dom
fiberspace(m::FiberProduct) = m.cod
metricextensor(m::FiberProduct{P,N} where P) where N = Gloabl{N}(InducedMetric())
metrictensor(m::FiberProduct{P,N} where P) where N = Gloabl{N}(InducedMetric())
pointtype(m::FiberProduct) = basetype(m)
pointtype(m::Type{<:FiberProduct}) = basetype(m)
metrictype(m::FiberProduct) = fibertype(m)
metrictype(m::Type{<:FiberProduct}) = fibertype(m)
basetype(::FiberProduct{B}) where B = B
basetype(::Type{<:FiberProduct{B}}) where B = B
fibertype(::FiberProduct) = InducedMetric
fibertype(::Type{<:FiberProduct}) = InducedMetric

Base.size(m::FiberProduct) = size(m.dom)
#Base.broadcast(f,t::FiberProduct{P}) where P = FiberProduct{P}(f.(points(t)),f.(fiberspace(t)))

@pure Grassmann.Manifold(m::FiberProduct) = Manifold(points(m))
@pure LinearAlgebra.rank(m::FiberProduct) = rank(points(m))
@pure Grassmann.grade(m::FiberProduct) = grade(points(m))
@pure Grassmann.antigrade(m::FiberProduct) = antigrade(points(m))
@pure Grassmann.mdims(m::FiberProduct) = mdims(points(m))

@pure Base.eltype(::Type{<:FiberProduct{P}}) where P = Coordinate{P,InducedMetric}
function Base.getindex(m::FiberProduct,i::Int,j::Vararg{Int})
    Coordinate(getindex(points(m),i,j...) ⧺ getindex(fiberspace(m),j...), InducedMetric())
end
Base.setindex!(m::FiberProduct{P},s::P,i::Int,j::Vararg{Int}) where P = setindex!(points(m),s,i,j...)
function Base.setindex!(m::FiberProduct,s::Coordinate,i::Int,j::Vararg{Int})
    setindex!(points(m),point(s),i,j...)
    return s
end

# GlobalFiber

abstract type GlobalFiber{E,N} <: FiberBundle{E,N} end
Base.@pure isfiberbundle(::GlobalFiber) = true
Base.@pure isfiberbundle(::Any) = false

globalfiber(x) = x
globalfiber(x::GlobalFiber) = fiber(x)

topology(m::GlobalFiber) = topology(immersion(m))
vertices(m::GlobalFiber) = vertices(immersion(m))
iscover(m::GlobalFiber) = iscover(immersion(m))
imagepoints(m::GlobalFiber) = iscover(m) ? points(m) : points(m)[vertices(m)]

unitdomain(t::GlobalFiber) = base(t)*inv(base(t)[end])
arcdomain(t::GlobalFiber) = unitdomain(t)*arclength(codomain(t))
graph(t::GlobalFiber) = graph.(t)

Base.size(m::GlobalFiber) = size(m.cod)
Base.resize!(m::GlobalFiber,i) = ((resize!(domain(m),i),resize!(codomain(m),i)); m)
resize_lastdim!(m::GlobalFiber,i) = ((resize_lastdim!(domain(m),i),resize_lastdim!(codomain(m),i)); m)

# AbstractFrameBundle

export AbstractFrameBundle, GridFrameBundle, SimplexFrameBundle, FacetFrameBundle
export IntervalRange, AlignedRegion, AlignedSpace

abstract type AbstractFrameBundle{M,N} <: GlobalFiber{M,N} end

base(m::AbstractFrameBundle) = m.p
immersion(m::AbstractFrameBundle) = m.t
coordinates(m::AbstractFrameBundle) = m
points(m::AbstractFrameBundle) = points(base(m))
fiber(m::AbstractFrameBundle) = fiber(base(m))
metricextensor(m::AbstractFrameBundle) = metricextensor(base(m))
metrictensor(m::AbstractFrameBundle) = metrictensor(base(m))
pointtype(m::AbstractFrameBundle) = basetype(m)
pointtype(m::Type{<:AbstractFrameBundle}) = basetype(m)
metrictype(m::AbstractFrameBundle) = fibertype(m)
metrictype(m::Type{<:AbstractFrameBundle}) = fibertype(m)
Base.size(m::AbstractFrameBundle) = size(points(m))

@pure isbundle(::AbstractFrameBundle) = true
@pure isbundle(t) = false

@pure Grassmann.Manifold(m::AbstractFrameBundle) = Manifold(points(m))
@pure LinearAlgebra.rank(m::AbstractFrameBundle) = rank(points(m))
@pure Grassmann.grade(m::AbstractFrameBundle) = grade(points(m))
@pure Grassmann.antigrade(m::AbstractFrameBundle) = antigrade(points(m))
@pure Grassmann.mdims(m::AbstractFrameBundle) = mdims(points(m))

# GridFrameBundle

grid_id = 0

struct GridFrameBundle{N,C<:Coordinate,PA<:FiberBundle{C,N},TA<:ImmersedTopology} <: AbstractFrameBundle{C,N}
    id::Int
    p::PA
    t::TA
    GridFrameBundle(id::Int,p::PA,t::TA=OpenTopology(size(p))) where {N,C<:Coordinate,PA<:FiberBundle{C,N},TA<:ImmersedTopology} = new{N,C,PA,TA}(id,p,t)
    GridFrameBundle(p::PA,t::TA=OpenTopology(size(p))) where {N,C<:Coordinate,PA<:FiberBundle{C,N},TA<:ImmersedTopology} = new{N,C,PA,TA}((global grid_id+=1),p,t)
end

GridFrameBundle(id::Int,p::PA,g::GA) where {N,P,G,PA<:AbstractArray{P,N},GA<:AbstractArray{G,N}} = GridFrameBundle(id,PointArray(0,p,g))
GridFrameBundle(p::PA,g::GA) where {N,P,G,PA<:AbstractArray{P,N},GA<:AbstractArray{G,N}} = GridFrameBundle((global grid_id+=1),p,g)

const IntervalRange{P<:Real,G,PA<:AbstractRange,GA} = GridFrameBundle{1,Coordinate{P,G},<:PointVector{P,G,PA,GA}}
const AlignedRegion{N,P<:Chain,G<:InducedMetric,PA<:RealRegion{V,<:Real,N,<:AbstractRange} where V,GA<:Global} = GridFrameBundle{N,Coordinate{P,G},PointArray{P,G,N,PA,GA}}
const AlignedSpace{N,P<:Chain,G<:InducedMetric,PA<:RealRegion{V,<:Real,N,<:AbstractRange} where V,GA} = GridFrameBundle{N,Coordinate{P,G},PointArray{P,G,N,PA,GA}}

GridFrameBundle(id::Int,p::PA) where {N,P,PA<:AbstractArray{P,N}} = GridFrameBundle(id,PointArray(0,p,Global{N}(InducedMetric())))
GridFrameBundle(p::PA) where {N,P,PA<:AbstractArray{P,N}} = GridFrameBundle(p,Global{N}(InducedMetric()))
GridFrameBundle(dom::GridFrameBundle,fun) = GridFrameBundle(base(dom), fun)
GridFrameBundle(dom::GridFrameBundle,fun::Array) = GridFrameBundle(base(dom), fun)
GridFrameBundle(dom::GridFrameBundle,fun::Function) = GridFrameBundle(base(dom), fun)
GridFrameBundle(dom::AbstractArray,fun::Function) = GridFrameBundle(dom, fun.(dom))

isopen(t::GridFrameBundle) = isopen(immersion(t))
iscompact(t::GridFrameBundle) = iscompact(immersion(t))
isinduced(t::GridFrameBundle) = isinduced(base(t))
bundle(m::GridFrameBundle) = m.id
basetype(m::GridFrameBundle) = basetype(base(m))
basetype(::Type{<:GridFrameBundle{N,C} where N}) where C = basetype(C)
fibertype(m::GridFrameBundle) = fibertype(base(m))
fibertype(::Type{<:GridFrameBundle{N,C} where N}) where C = fibertype(C)

⊕(a::GridFrameBundle{1},b::GridFrameBundle{1}) = GridFrameBundle(base(a)⊕base(b),immersion(a) × immersion(b))
⊕(a::GridFrameBundle,b::AbstractVector{<:Real}) = GridFrameBundle(base(a)⊕b,immersion(a) × length(b))
cross(a::GridFrameBundle,b::GridFrameBundle) = a⊕b
cross(a::GridFrameBundle,b::AbstractVector{<:Real}) = a⊕b

function resample(m::GridFrameBundle,i::NTuple)
    rp,rq = resample(points(m),i),resample(immersion(m),i)
    gid = iszero(m.id) ? 0 : (global grid_id+=1)
    pid = iszero(base(m).id) ? 0 : (global point_id+=1)
    if isinduced(m)
        GridFrameBundle(gid,PointArray(pid,rp),rq)
    else
        GridFrameBundle(gid,PointArray(pid,rp,m.(rp)),rq)
    end
end

resize_lastdim!(m::GridFrameBundle,i) = (resize_lastdim!(base(m),i); m)
resize(m::GridFrameBundle) = GridFrameBundle(m.id,base(m),resize(immersion(m),size(base(m))[end]))
Base.resize!(m::GridFrameBundle,i) = (resize!(base(m),i); m)
Base.broadcast(f,t::GridFrameBundle) = GridFrameBundle(f.(base(t)))

(m::GridFrameBundle)(i::ImmersedTopology) = GridFrameBundle(bundle(m),base(m),i)
(m::GridFrameBundle)(i::Vararg{Union{Int,Colon}}) = GridFrameBundle(0,base(m)(i...),immersion(m)(i...))
@pure Base.eltype(::Type{<:GridFrameBundle{N,C} where N}) where C = C
Base.getindex(m::GridFrameBundle,i::Vararg{Int}) = getindex(base(m),i...)
Base.getindex(m::GridFrameBundle,i::Vararg{Union{Int,Colon}}) = GridFrameBundle(0,getindex(base(m),i...),immersion(m)(i...))
Base.setindex!(m::GridFrameBundle,s,i::Vararg{Int}) = setindex!(base(m),s,i...)

export Grid
const Grid = GridFrameBundle

#Grid(v::A,t::I=OpenTopology(size(v))) where {N,T,A<:AbstractArray{T,N},I} = GridFrameBundle(0,PointArray(0,v),t)

Base.getindex(g::GridFrameBundle{M,C,<:FiberBundle,<:OpenTopology} where C,j::Int,n::Val,i::Vararg{Int}) where M = getpoint(g,j,n,i...)
@generated function getpoint(g::GridFrameBundle{M,C,<:FiberBundle} where C,j::Int,::Val{N},i::Vararg{Int}) where {M,N}
    :(Base.getindex(points(g),$([k≠N ? :(@inbounds i[$k]) : :(@inbounds i[$k]+j) for k ∈ list(1,M)]...)))
end
@generated function Base.getindex(g::GridFrameBundle{M},j::Int,n::Val{N},i::Vararg{Int}) where {M,N}
    :(Base.getindex(points(g),Base.getindex(immersion(g),n,$([k≠N ? :(@inbounds i[$k]) : :(@inbounds i[$k]+j) for k ∈ list(1,M)]...))...))
end

Base.BroadcastStyle(::Type{<:GridFrameBundle{N,C,PA}}) where {N,C,PA} = Broadcast.ArrayStyle{GridFrameBundle{N,C,PA}}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GridFrameBundle{N,C,PA}}}, ::Type{ElType}) where {N,C,PA,ElType<:Coordinate}
    ax = axes(bc)
    # Use the data type to create the output
    GridFrameBundle(similar(Array{pointtype(ElType),N}, ax), similar(Array{metrictype(ElType),N}, ax))
end
#=function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GridFrameBundle{N,C,PA}}}, ::Type{ElType}) where {N,C,PA,ElType}
    t = find_gf(bc)
    # Use the data type to create the output
    GridFrameBundle(similar(Array{ElType,N}, axes(bc)), metricextensor(t))
end=#

"`A = find_gf(As)` returns the first GridFrameBundle among the arguments."
find_gf(bc::Base.Broadcast.Broadcasted) = find_gf(bc.args)
find_gf(bc::Base.Broadcast.Extruded) = find_gf(bc.x)
find_gf(args::Tuple) = find_gf(find_gf(args[1]), Base.tail(args))
find_gf(x) = x
find_gf(::Tuple{}) = nothing
find_gf(a::AbstractFrameBundle, rest) = a
find_gf(::Any, rest) = find_gf(rest)

# SimplexFrameBundle

struct SimplexFrameBundle{P,G,PA<:AbstractVector{P},GA<:AbstractVector{G},TA<:ImmersedTopology} <: AbstractFrameBundle{Coordinate{P,G},1}
    p::PointCloud{P,G,PA,GA}
    t::TA
    SimplexFrameBundle(p::PointCloud{P,G,PA,GA},t::T) where {P,G,PA,GA,T} = new{P,G,PA,GA,T}(p,t)
end

SimplexFrameBundle(id::Int,p,t,g) = SimplexFrameBundle(PointCloud(id,p,g),t)
SimplexFrameBundle(id::Int,p,t) = SimplexFrameBundle(PointCloud(id,p),t)
SimplexFrameBundle(p::P,t,g::G) where {P<:AbstractVector,G<:AbstractVector} = SimplexFrameBundle(PointCloud(p,g),t)
#SimplexFrameBundle(p::AbstractVector,t) = SimplexFrameBundle(PointCloud(p),t)

(p::PointCloud)(t::ImmersedTopology) = SimplexFrameBundle(p,t)
PointCloud(m::SimplexFrameBundle) = m.p
bundle(m::SimplexFrameBundle) = m.id
deletebundle!(m::SimplexFrameBundle) = deletepointcloud!(bundle(m))
basetype(::SimplexFrameBundle{P}) where P = pointtype(P)
basetype(::Type{<:SimplexFrameBundle{P}}) where P = pointtype(P)
fibertype(::SimplexFrameBundle{P}) where P = metrictype(P)
fibertype(::Type{<:SimplexFrameBundle{P}}) where P = metrictype(P)
Base.size(m::SimplexFrameBundle) = size(vertices(m))

isinduced(t::SimplexFrameBundle) = isinduced(base(t))

Base.broadcast(f,t::SimplexFrameBundle) = SimplexFrameBundle(f.(PointCloud(t)),ImmersedTopology(t))

Base.firstindex(m::SimplexFrameBundle) = 1
Base.lastindex(m::SimplexFrameBundle) = length(vertices(m))
Base.length(m::SimplexFrameBundle) = length(vertices(m))
#Base.resize!(m::SimplexFrameBundle,n::Int) = resize!(value(m),n)

(m::SimplexFrameBundle)(i::ImmersedTopology) = SimplexFrameBundle(bundle(m),PointCloud(m),i)
Base.getindex(m::SimplexFrameBundle,i::Chain{V,1}) where V = Chain{Manifold(V),1}(points(m)[value(i)])
Base.getindex(m::SimplexFrameBundle,i::Values{N,Int}) where N = points(m)[value(i)]
getindex(m::AbstractVector,i::ImmersedTopology) = getindex.(Ref(m),topology(i))
getindex(m::AbstractVector,i::SimplexFrameBundle) = m[immersion(i)]
getindex(m::SimplexFrameBundle,i::ImmersedTopology) = points(m)[i]
getindex(m::SimplexFrameBundle,i::SimplexFrameBundle) = points(m)[immersion(i)]

getimage(m,i) = iscover(m) ? i : getindex(vertices(m),i)

@pure Base.eltype(::Type{<:SimplexFrameBundle{P,G}}) where {P,G} = Coordinate{P,G}
function Base.getindex(m::SimplexFrameBundle,i::Int)
    ind = getimage(m,i)
    Coordinate(getindex(points(m),ind), getindex(metricextensor(m),ind))
end
Base.setindex!(m::SimplexFrameBundle{P},s::P,i::Int) where P = setindex!(points(m),s,getimage(m,i))
Base.setindex!(m::SimplexFrameBundle{P,G} where P,s::G,i::Int) where G = setindex!(metricextensor(m),s,getimage(m,i))
function Base.setindex!(m::SimplexFrameBundle,s::Coordinate,i::Int)
    ind = getimage(m,i)
    setindex!(points(m),point(s),ind)
    setindex!(metricextensor(m),metricextensor(s),ind)
    return s
end

# FacetFrameBundle

struct FacetFrameBundle{P,G,PA,GA,TA<:ImmersedTopology} <: AbstractFrameBundle{Coordinate{P,G},1}
    id::Int
    p::PointCloud{P,G,PA,GA}
    t::TA
    FacetFrameBundle(id::Int,p::PointCloud{P,G,PA,GA},t::T) where {P,G,PA,GA,T} = new{P,G,PA,GA,T}(id,p,t)
end

#FacetFrameBundle(id::Int,p,t,g) = FacetFrameBundle(PointCloud(id,p,g),t)
#FacetFrameBundle(id::Int,p,t) = FacetFrameBundle(PointCloud(id,p),t)
#FacetFrameBundle(p::AbstractVector,t) = FacetFrameBundle(PointCloud(p),t)

function SimplexFrameBundle(m::FacetFrameBundle)
    SimplexFrameBundle(PointCloud(m.id,point_cache[m.id],point_metric_cache[m.id]),ImmersedTopology(m))
end
function FacetFrameBundle(m::SimplexFrameBundle)
    et = topology(ImmersedTopology(m))
    FacetFrameBundle(m.id,PointCloud(0,barycenter.(m[et]),barycenter.(getindex.(Ref(metricextensor(m)),et))),ImmersedTopology(m))
end

PointCloud(m::FacetFrameBundle) = m.p
basetype(::FacetFrameBundle{P}) where P = pointtype(P)
basetype(::Type{<:FacetFrameBundle{P}}) where P = pointtype(P)
fibertype(::FacetFrameBundle{P}) where P = metrictype(P)
fibertype(::Type{<:FacetFrameBundle{P}}) where P = metrictype(P)

Base.broadcast(f,t::FacetFrameBundle) = FacetFrameBundle(0,f.(PointCloud(t)),ImmersedTopology(t))

bundle(m::FacetFrameBundle) = m.id
deletebundle!(m::FacetFrameBundle) = deletepointcloud!(bundle(m))
#@pure ispoints(t::Submanifold{V}) where V = isbundle(V) && rank(V) == 1 && !isbundle(Manifold(V))
#@pure ispoints(t) = isbundle(t) && rank(t) == 1 && !isbundle(Manifold(t))
#@pure islocal(t) = isbundle(t) && rank(t)==1 && valuetype(t)==Int && ispoints(Manifold(t))
#@pure iscell(t) = isbundle(t) && islocal(Manifold(t))

@pure Base.eltype(::Type{<:FacetFrameBundle{P,G}}) where {P,G} = Coordinate{P,G}
function Base.getindex(m::FacetFrameBundle,i::Int)
    ind = getimage(m,i)
    Coordinate(getindex(points(m),ind), getindex(metricextensor(m),ind))
end
Base.setindex!(m::FacetFrameBundle{P},s::P,i::Int) where P = setindex!(points(m),s,getimage(m,i))
Base.setindex!(m::FacetFrameBundle{P,G} where P,s::G,i::Int) where G = setindex!(metricextensor(m),s,getimage(m,i))
function Base.setindex!(m::FacetFrameBundle,s::Coordinate,i::Int)
    ind = getimage(m,i)
    setindex!(points(m),point(s),ind)
    setindex!(metricextensor(m),metricextensor(s),ind)
    return s
end

# FiberProductBundle

struct FiberProductBundle{P,N,SA<:AbstractArray,PA<:AbstractArray} <: AbstractFrameBundle{Coordinate{P,InducedMetric},N}
    s::SA
    g::PA
    FiberProductBundle{P}(s::SA,g::PA) where {P,M,N,SA<:AbstractArray{S,M} where S,PA<:AbstractArray{F,N} where F} = new{P,M+N,SA,PA}(s,g)
end

varmanifold(N::Int) = Submanifold(N+1)(list(1,N)...)

function ⊕(a::SimplexFrameBundle,b::AbstractVector{B}) where B<:Real
    V = Manifold(basetype(eltype(a)))
    N = mdims(V)+1
    W = Submanifold(N)
    P = Chain{W,1,promote_type(valuetype(basetype(eltype(a))),B),N}
    #p = PointCloud(Chain{varmanifold(N-1)}.(value.(points(a))))
    FiberProductBundle{P}(a,ProductSpace{W(N)}(Values((b,))))
end

basetype(m::FiberProductBundle{P}) where P = basetype(P)
basetype(::Type{<:FiberProductBundle{P}}) where P = basetype(P)
fibertype(m::FiberProductBundle{P}) where P = fibertype(P)
fibertype(::Type{<:FiberProductBundle{P}}) where P = fibertype(P)
Base.size(m::FiberProductBundle) = (length(m.s),size(m.g)...)

@pure Base.eltype(::Type{<:FiberProductBundle{P}}) where P = Coordinate{P,InducedMetric}
Base.getindex(m::FiberProductBundle,i::Int,j::Vararg{Int}) = Coordinate(getindex(points(m.s),i) ⧺ getindex(m.g,j...), InducedMetric())
#=Base.setindex!(m::FiberProductBundle{P},s::P,i::Int,j::Vararg{Int}) where P = setindex!(points(m),s,i,j...)
Base.setindex!(m::FiberProductBundle{P,G} where P,s::G,i::Int,j::Vararg{Int}) where G = setindex!(metricextensor(m),s,i,j...)
function Base.setindex!(m::FiberProductBundle,s::Coordinate,i::Int,j::Vararg{Int})
    setindex!(points(m),point(s),i...)
    setindex!(metricextensor(m),metricextensor(s),i...)
    return s
end=#

# HomotopyBundle

struct HomotopyBundle{P,N,PA<:AbstractArray{F,N} where F,FA<:AbstractArray,TA<:ImmersedTopology} <: AbstractFrameBundle{Coordinate{P,InducedMetric},N}
    p::FiberProduct{P,N,PA,FA}
    t::TA
    HomotopyBundle(p::FiberProduct{P,N,PA,FA},t::T) where {P,N,PA<:AbstractArray{F,N} where F,FA,T} = new{P,N,PA,FA,T}(p,t)
end

(p::FiberProduct)(t::ImmersedTopology) = HomotopyBundle(p,t)
FiberProduct(m::HomotopyBundle) = m.p
fiberspace(m::HomotopyBundle) = fiberspace(base(m))
basetype(::HomotopyBundle{P}) where P = pointtype(P)
basetype(::Type{<:HomotopyBundle{P}}) where P = pointtype(P)
fibertype(::HomotopyBundle{P}) where P = fibertype(P)
fibertype(::Type{<:HomotopyBundle{P}}) where P = fibertype(P)
Base.size(m::HomotopyBundle) = size(FiberProduct(m))

Base.broadcast(f,t::HomotopyBundle) = HomotopyBundle(f.(FiberProduct(t)),ImmersedTopology(t))

(m::HomotopyBundle)(i::ImmersedTopology) = HomotopyBundle(FiberProduct(m),i)
#Base.getindex(m::HomotopyBundle,i::Chain{V,1}) where V = Chain{Manifold(V),1}(points(m)[value(i)])
#Base.getindex(m::HomotopyBundle,i::Values{N,Int}) where N = points(m)[value(i)]
getindex(m::HomotopyBundle,i::ImmersedTopology) = FiberProduct(m)[i]
getindex(m::HomotopyBundle,i::HomotopyBundle) = FiberProduct(m)[immersion(i)]

@pure Base.eltype(::Type{<:HomotopyBundle{P}}) where P = Coordinate{P,InducedMetric}
function Base.getindex(m::HomotopyBundle,i::Int,j::Vararg{Int})
    ind = getimage(m,i)
    Coordinate(getindex(points(m),ind,j...) ⧺ getindex(fiberspace(m),j...), InducedMetric())
end
#=Base.setindex!(m::HomotopyBundle{P},s::P,i::Int) where P = setindex!(points(m),s,getimage(m,i))
function Base.setindex!(m::HomotopyBundle,s::Coordinate,i::Int)
    ind = getimage(m,i)
    setindex!(points(m),point(s),ind)
    return s
end=#


