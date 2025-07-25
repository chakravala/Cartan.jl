
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

"""
    ProductSpace{V,T,N,M,S} <: AbstractArray{Chain{V,1,T,N},N}

Can be constructed with `\\oplus` operation `⊕` and `AbstractRange`,
```julia
julia> (0:0.1:1)⊕(0:0.1:1)
11×11 ProductSpace{⟨_11_⟩, Float64, 2, 2, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}:
...
```
generating a lazy array of `Chain{V,1}` point vectors from the input ranges.
```Julia
Rectangle (alias for ProductSpace{V, T, 2, 2} where {V, T})
Hyperrectangle (alias for ProductSpace{V, T, 3, 3} where {V, T})
RealRegion{V, T} where {V, T<:Real} (alias for ProductSpace{V, T, N, N, S} where {V, T<:Real, N, S<:AbstractArray{T, 1}})
```
"""
struct ProductSpace{V,T,N,M,S} <: AbstractArray{Chain{V,1,T,N},N}
    v::Values{M,S} # how to deal with T???
    ProductSpace{V,T,N}(v::Values{M,S}) where {V,T,N,M,S} = new{Grassmann.DirectSum.submanifold(V),T,N,M,S}(v)
    ProductSpace{V,T}(v::Values{M,S}) where {V,T,M,S} = new{Grassmann.DirectSum.submanifold(V),T,mdims(V),M,S}(v)
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

"""
    resample(m,i...)

Resamples a ranged `ProductSpace` or related objects.
"""
resample(m::OneTo,i::Int) = LinRange(1,m.stop,i)
resample(m::UnitRange,i::Int) = LinRange(m.start,m.stop,i)
resample(m::StepRange,i::Int) = LinRange(m.start,m.stop,i)
resample(m::LinRange,i::Int) = LinRange(m.start,m.stop,i)
resample(m::StepRangeLen,i::Int) = StepRangeLen(m.ref,(m.step*(m.len-1))/(i-1),i)
resample(m::AbstractRange,i::NTuple{1,Int}) = resample(m,i...)
resample(m::AbstractArray{T,0},::Tuple{}) where T = m
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

"""
    ⊕(v::AbstractVector{<:Real}...)

Constructs a direct sum basis space using the Cartesian `ProductSpace` implementation.
"""
⊕(a::AbstractVector{<:Real}...) = RealRegion(Values(a))
⊕(a::ProductSpace,b::AbstractVector{<:Real}) = RealRegion(Values(a.v...,b))
⊕(a::AbstractVector{<:Real},b::ProductSpace) = RealRegion(Values(a,b.v...))
⊕(a::ProductSpace,b::ProductSpace) = RealRegion(Values(a.v...,b.v...))
cross(a::ProductSpace,b::AbstractVector{<:Real}) = a⊕b
cross(a::ProductSpace,b::ProductSpace) = a⊕b

RealRegion(a::AbstractVector{<:Real}...) = RealRegion(Values(a))
ProductSpace(a::AbstractVector{<:Real}...) = ProductSpace(Values(a))
RealRegion{V}(a::AbstractVector{<:Real}...) where V = RealRegion{V}(Values(a))
ProductSpace{V}(a::AbstractVector{<:Real}...) where V = ProductSpace{V}(Values(a))

@generated ⧺(a::Real...) = :(Chain($([:(a[$i]) for i ∈ 1:length(a)]...)))
@generated ⧺(a::Complex...) = :(Chain($([:(a[$i]) for i ∈ 1:length(a)]...)))
⧺(a::Chain{A,G},b::Chain{B,G}) where {A,B,G} = Chain{A∪B,G}(vcat(a.v,b.v))

widths(t::AbstractVector) = t[end]-t[1]
widths(t::ProductSpace) = widths.(t.v)

remove(t::ProductSpace{V,T,2} where {V,T},::Val{1}) = (@inbounds t.v[2])
remove(t::ProductSpace{V,T,2} where {V,T},::Val{2}) = (@inbounds t.v[1])
@generated remove(t::ProductSpace{V,T,N} where {V,T},::Val{J}) where {N,J} = :(ProductSpace(t.v[$(Values([i for i ∈ 1:N if i≠J]...))]))

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

"""
    ImmersedTopology{N,M} = AbstractArray{Values{N,Int},M}

Any `ImmersedTopology{N,M}` is also an `M`-dimensional `AbstractArray` of `Values{N,Int}`.
"""
const ImmersedTopology{N,M} = AbstractArray{Values{N,Int},M}

"""
    immersion(m) -> ImmersedTopology

Returns the associated `ImmersedTopology` of any `FrameBundle` or related object.
"""
const immersion = ImmersedTopology

"""
    sdims(::ImmersedTopology{N}) where N = N

Dimension `N` of the associated immersed simplex.
"""
sdims(m::ImmersedTopology{N}) where N = N
sdims(m::Type{<:ImmersedTopology{N}}) where N = N

"""
    immersiontype(m) -> DataType

Returns the `typeof` the `immersion` of `m`.
"""
immersiontype(m::ImmersedTopology) = typeof(m)


"""
    fullimmersion(m) -> ImmersedTopology

Returns full associated `ImmersedTopology` of subspace `FrameBundle` or related object.
"""
fullimmersion(m::ImmersedTopology) = m

topology(m::ImmersedTopology{N,1}) where N = m
subelements(m::ImmersedTopology{N,1}) where N = OneTo(length(m))

@pure Base.eltype(::Type{ImmersedTopology{N}}) where N = Values{N,Int}

# ProductTopology

"""
    ProductTopology{N} <: ImmersedTopology{N,N}

Define basic `ProductTopology` by ranges of integers,
```julia
julia> ProductTopology(11,11)
11×11 ProductTopology{2, OneTo{Int64}}:
...

julia> ProductTopology(1:11,1:11)
11×11 ProductTopology{2, UnitRange{Int64}}:
...
```
"""
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

"""
    SimplexTopology{N} <: ImmersedTopology{N,1}

Defines continuous subspaces of a `fulltopology` over `N`-dimensional `Simplex` spaces having `vertices` and  `subelements` defined and indexed in its simplicial complex.
```Julia
bundle(t) # cache identification
fulltopology(t) # full element list
vertices(t) # # subspace vertices
totalnodes(t) # full node count
subelements(t) # list of elements
fullvertices(t) # fulltopology vertices
verticesinv(t) # inverted data of vertices
```
Related methods include `bundle`, `fulltopology`, `topology`, `fullvertices`, `vertices`, `totalelements`, `elements`, `subelements`, `totalnodes`, `nodes`, `istotal`, `isfull`, `iscover`, `fullimmersion`, `subimmersion`.
"""
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

"""
    bundle(m::ImmersedTopology) -> Int

Returns integer identification of `bundle` cache.
"""
bundle(m::SimplexTopology) = m.id

"""
    fulltopology(m) -> Vector{Values{N,Int}}

Returns the `fulltopology` regardless of whether the `ImmersedTopology` subset `isfull`.
"""
fulltopology(m::SimplexTopology) = m.t

"""
    topology(m) -> AbstractVector{Values{N,Int}}

Returns a `view` into the `fulltopology` based on `subelements(m)` structure.
"""
topology(m::SimplexTopology) = isfull(m) ? fulltopology(m) : view(fulltopology(m),subelements(m))

"""
    totalelements(m) -> Int

Return the total number of elements in the `fulltopology(m)`.
"""
totalelements(m::SimplexTopology) = length(fulltopology(m))

"""
    elements(m) -> Int

Return the number of `subelements(m)`.
"""
elements(m::SimplexTopology) = length(subelements(m))

"""
    subelements(m) -> AbstractVector{Int}

Return the subspace element indices associated to `fulltopology(m)`.
"""
subelements(m::SimplexTopology) = m.f

"""
    refnodes(m) -> Base.RefValaue{Int}

Return the shared mutable `Base.RefValue{Int}` which counts `nodes(m)`.
"""
refnodes(m::SimplexTopology) = m.p
totalnodes!(m::SimplexTopology,p) = (refnodes(m).x = p)

"""
    totalnodes(m) -> Int

Return the number which counts the total number of `nodes` regardless of subspace.
"""
totalnodes(m::SimplexTopology) = refval(refnodes(m))

"""
    nodes(m) -> Int

Return the number of `vertices(m)` associated to the subspace `immersion(m)`.
"""
nodes(m::SimplexTopology) = length(vertices(m))

"""
    fullvertices(m) -> AbstractVector{Int}

Return the list of `vertices(m)` associated to the `fullimmersion(m)`.
"""
fullvertices(m::SimplexTopology) = m.I

"""
    vertices(m) -> AbstractVector{Int}

Return the list of `vertices(m)` associated to the subspace `immersion(m)`.
"""
vertices(m::SimplexTopology) = m.i
verticesinv(m::SimplexTopology) = m.v

Base.size(m::SimplexTopology) = size(subelements(m))
Base.length(m::SimplexTopology) = elements(m)
Base.axes(m::SimplexTopology) = axes(subelements(m))
Base.getindex(m::SimplexTopology,i::Int) = getindex(fulltopology(m),getfacet(m,i))
Grassmann.mdims(m::SimplexTopology{N}) where N = N

"""
    getimage(m,i) -> Int

Return the index of vertex subspace `immersion` in reference to the `fullimmersion`.
"""
getimage(m::SimplexTopology{N,<:AbstractVector} where N,i) = iscover(m) ? i : vertices(m)[i]
getimage(m::SimplexTopology{N,<:OneTo} where N,i) = i

"""
    getfacet(m,i) -> Int

Return the index of `subelements` in reference to the `fullimmersion`.
"""
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

function untotal(t::SimplexTopology,p)
    SimplexTopology(bundle(t),topology(t),vertices(t),p,subelements(t),fullvertices(t),false,true)
end

function fullimmersion_vertices(m::ImmersedTopology)
    if istotal(m)
        OneTo(totalnodes(m))
    else
        out = fullvertices(m)
        n = length(out)
        maximum(out) == n ? OneTo(n) : out
    end
end

function fullimmersion(m::SimplexTopology)
    top,ind,ist = fulltopology(m),fullimmersion_vertices(m),istotal(m)
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

"""
    subimmersion(m) -> ImmersedTopology

Return a modified `subimmersion` with all vertices re-indexed based on the subspace.
"""
function subimmersion(m::SimplexTopology{N,<:OneTo} where N)
    iscover(m) && (return m)
    top,ind = topology(m),vertices(m)
    SimplexTopology(0,top,ind,length(ind),OneTo(length(top)),ind,true,true)
end
function subimmersion(m::SimplexTopology{N,<:AbstractVector} where N)
    iscover(m) && (return m)
    top,p = topology(m),nodes(m)
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

"""
    DiscontinuousTopology{N} <: ImmersedTopology{N,1}

Defines discontinuous subspace of a `fulltopology` over `N`-dimensional `Simplex` spaces having `vertices` and  `subelements` defined and indexed in its simplicial complex.
```Julia
isdiscontinuous(t) # true if DiscontinuousTopology
isdisconnected(t) # true if graph of discontinuous
```
Calling `DiscontinuousToppology(::SimplexTopology)` initializes a discontinuous variant, while `SimplexTopology(::DiscontinuousTopology)` returns the initial continuous data.
Related methods include `bundle`, `fulltopology`, `topology`, `fullvertices`, `vertices`, `totalelements`, `elements`, `subelements`, `totalnodes`, `nodes`, `istotal`, `isfull`, `iscover`, `fullimmersion`, `subimmersion`, `isdiscontinuous`, `isdisconnected`, `continuous`, `discontinuous`, `disconnect`.
"""
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

for fun ∈ (:totalelements,:elements,:subelements,:istotal,:isfull,:iscover)
    @eval $fun(m::DiscontinuousTopology) = $fun(SimplexTopology(m))
end
bundle(m::DiscontinuousTopology) = m.id
fulltopology(m::DiscontinuousTopology) = topology(fullimmersion(m))
topology(m::DiscontinuousTopology) = collect(m)
totalnodes(m::DiscontinuousTopology{N}) where N = N*totalelements(m)
nodes(m::DiscontinuousTopology{N}) where N = N*elements(m)
fullvertices(m::DiscontinuousTopology) = m.I
vertices(m::DiscontinuousTopology) = m.i

"""
    isdiscontinuous(m) -> Bool

Return `true` if `m` is a `DiscontinuousTopology`.
"""
isdiscontinuous(m::SimplexTopology) = false
isdiscontinuous(m::DiscontinuousTopology) = true

"""
    isdiscontinuous(m) -> Bool

Return `true` if `m` is a disconnected `DiscontinuousTopology`.
"""
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

function fullimmersion(m::DiscontinuousTopology)
    ind = fullimmersion_vertices(m)
    DiscontinuousTopology(bundle(m),fullimmersion(SimplexTopology(m)),ind,ind)
end

function Base.getindex(m::DiscontinuousTopology,i::AbstractVector{Int})
    DiscontinuousTopology(bundle(m),SimplexTopology(m)[i],fullvertices(m))
end

(m::DiscontinuousTopology)(i::AbstractVector{Int}) = subtopology(m,i)
function subtopology(m::DiscontinuousTopology{N},i::AbstractVector{Int}) where N
    DiscontinuousTopology(bundle(m),subtopology(SimplexTopology(m),i),fullvertices(m))
end

"""
    continuous(m) -> ImmersedTopology

Return the original `continuous` data of possibly a `DiscontinuousTopology`.
"""
continuous(m::SimplexTopology) = m
continuous(m::DiscontinuousTopology) = SimplexTopology(m)

"""
    discontinuous(m) -> DiscontinuousTopology

Return a derived `DiscontinuousTopology` from a `SimplexTopology`.
"""
discontinuous(m::DiscontinuousTopology) = m
discontinuous(m::SimplexTopology) = DiscontinuousTopology(0,m)

"""
    disconnect(m) -> DiscontinuousTopology

Return a disconnected variant of a `DiscontinuousTopology`.
"""
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

# LagrangeTopology

export LagrangeTopology, LagrangeTriangles, LagrangeTetrahedra, cornertopology
export totalcornernodes, totaledgesnodes, totalcenternodes
export cornernodes, edgesnodes, centernodes

@pure simplexnumber(N,n) = Grassmann.binomial(n+N-1,N) # n-th N-Simplex number
trinum(n) = simplexnumber(2,n) # triangular number
tetnum(n) = simplexnumber(3,n) # tetrahedral number

abstract type LagrangeTopology{M,N,P,F,T} <: ImmersedTopology{N,1} end

for fun ∈ (:totalelements,:elements,:subelements,:istotal,:isfull,:iscover,:isdiscontinuous,:isdisconnected)
    @eval $fun(m::LagrangeTopology) = $fun(cornertopology(m))
end
bundle(m::LagrangeTopology) = m.id
fulltopology(m::LagrangeTopology) = topology(fullimmersion(m))
topology(m::LagrangeTopology) = collect(m)
totaledges(m::LagrangeTopology) = totalnodes(edgesindices(m))
totalfacets(m::LagrangeTopology) = totalnodes(facetsindices(m))
totalcornernodes(m::LagrangeTopology) = totalnodes(cornertopology(m))
totaledgesnodes(m::LagrangeTopology{M}) where M = (M-1)*totaledges(m)
totalfacetsnodes(m::LagrangeTopology) = facetsimplex(m)*totalfacets(m)
totalcenternodes(m::LagrangeTopology) = centersimplex(m)*totalelements(m)
cornernodes(m::LagrangeTopology) = nodes(cornertopology(m))
edgesnodes(m::LagrangeTopology{M}) where M= (M-1)*nodes(edgesindices(m))
facetsnodes(m::LagrangeTopology) = facetsimplex(m)*nodes(facetsindices(m))
centernodes(m::LagrangeTopology) = centersimplex(m)*elements(m)
@pure lagrangesimplex(m::LagrangeTopology{M}) where M = lagrangesimplex(sdims(cornertopology(m)),M)
@pure centersimplex(m::LagrangeTopology{M}) where M = centersimplex(sdims(cornertopology(m)),M)
@pure facetsimplex(m::LagrangeTopology{M}) where M = facetsimplex(sdims(cornertopology(m)),M)
@pure edgesimplex(m::LagrangeTopology{M}) where M = M-1
@pure lagrangesimplex(N,M) = simplexnumber(N-1,M+1)
@pure centersimplex(N,M) = simplexnumber(N-1,M-N+1)
@pure facetsimplex(N,M) = centersimplex(N-1,M)
@pure edgesimplex(N,M) = M-1
fullvertices(m::LagrangeTopology) = m.I
vertices(m::LagrangeTopology) = m.i
#verticesinv(m::LagrangeTopology) = m.v

Base.size(m::LagrangeTopology) = size(subelements(m))
Base.length(m::LagrangeTopology) = elements(m)
Base.axes(m::LagrangeTopology) = axes(subelements(m))
Grassmann.mdims(m::LagrangeTopology) = mdims(cornertopology(m))

getimage(m::LagrangeTopology{M,N,<:AbstractVector} where {M,N},i) = iscover(m) ? i : vertices(m)[i]
getimage(m::LagrangeTopology{M,N,<:OneTo} where {M,N},i) = i
getfacet(m::LagrangeTopology,i) = getfacet(cornertopology(m),i)

subtopology(m::LagrangeTopology{M,N,<:OneTo} where {M,N}) = topology(m)
function subtopology(m::LagrangeTopology{M,N,<:AbstractVector} where {M,N})
    iscover(m) ? topology(m) : getelement.(Ref(m),OneTo(elements(m)))
end

# LagrangeEdges, LagrangeTriangles, LagrangeTetrahedra

struct LagrangeEdges{M,N,P<:AbstractVector{Int},F<:AbstractVector{Int},T} <: LagrangeTopology{M,N,P,F,T}
    id::Int # bundle
    t::SimplexTopology{2,P,F,T}
    i::P # vertices
    I::P # fullvertices
    function LagrangeEdges{M}(id::Int,t::SimplexTopology{2,P,F,T},i::P,I::P=i) where {M,P<:AbstractVector{Int},F<:AbstractVector{Int},T}
        new{M,lagrangesimplex(2,M),P,F,T}(id,t,i,I)
    end
    function LagrangeEdges{M}(id::Int,t::SimplexTopology{2,P,F,T},i::P,I::OneTo) where {M,P<:Vector{Int},F<:AbstractVector{Int},T}
        new{M,lagrangesimplex(2,M),P,F,T}(id,t,i,collect(I))
    end
end

struct LagrangeTriangles{M,N,P<:AbstractVector{Int},F<:AbstractVector{Int},T} <: LagrangeTopology{M,N,P,F,T}
    id::Int # bundle
    t::SimplexTopology{3,P,F,T}
    e::SimplexTopology{2,P,F,T}
    ei::SimplexTopology{3,P,F,T}
    i::P # vertices
    I::P # fullvertices
    function LagrangeTriangles{M}(id::Int,t::SimplexTopology{3,P,F,T},e::SimplexTopology{2,P,F,T},ei::SimplexTopology{3,P,F,T},i::P,I::P=i) where {M,P<:AbstractVector{Int},F<:AbstractVector{Int},T}
        new{M,lagrangesimplex(3,M),P,F,T}(id,t,e,ei,i,I)
    end
    function LagrangeTriangles{M}(id::Int,t::SimplexTopology{3,P,F,T},e::SimplexTopology{2,P,F,T},ei::SimplexTopology{3,P,F,T},i::P,I::OneTo) where {M,P<:Vector{Int},F<:AbstractVector{Int},T}
        new{M,lagrangesimplex(3,M),P,F,T}(id,t,e,ei,i,collect(I))
    end
end

struct LagrangeTetrahedra{M,N,P<:AbstractVector{Int},F<:AbstractVector{Int},T} <: LagrangeTopology{M,N,P,F,T}
    id::Int # bundle
    t::SimplexTopology{4,P,F,T}
    f::SimplexTopology{3,P,F,T}
    e::SimplexTopology{2,P,F,T}
    fi::SimplexTopology{4,P,F,T}
    ei::SimplexTopology{6,P,F,T}
    i::P # vertices
    I::P # fullvertices
    function LagrangeTetrahedra{M}(id::Int,t::SimplexTopology{4,P,F,T},f::SimplexTopology{3,P,F,T},e::SimplexTopology{2,P,F,T},fi::SimplexTopology{4,P,F,T},ei::SimplexTopology{6,P,F,T},i::P,I::P=i) where {M,P<:AbstractVector{Int},F<:AbstractVector{Int},T}
        new{M,lagrangesimplex(4,M),P,F,T}(id,t,f,e,fi,ei,i,I)
    end
    function LagrangeTetrahedra{M}(id::Int,t::SimplexTopology{4,P,F,T},f::SimplexTopology{3,P,F,T},e::SimplexTopology{2,P,F,T},fi::SimplexTopology{4,P,F,T},ei::SimplexTopology{6,P,F,T},i::P,I::OneTo) where {M,P<:Vector{Int},F<:AbstractVector{Int},T}
        new{M,lagrangesimplex(4,M),P,F,T}(id,t,f,e,fi,ei,i,collect(I))
    end
end

LagrangeEdges{M}(t::SimplexTopology{2}) where M = LagrangeEdges{M}((global top_id += 1),t)
function LagrangeEdges{M}(id::Int,t::SimplexTopology{2}) where M
    np,ne = totalnodes(t),totalelements(t)
    I = OneTo(np+(M-1)*ne)
    i = iscover(t) ? I : lagrangevertices2(t,np,Val(M))
    LagrangeEdges{M}(id,t,i,iscover(t) ? I : collect(I))
end

function LagrangeTriangles{M}(t::SimplexTopology{3},e=edges(t),ei=edgesindices(t,e)) where M
    LagrangeTriangles{M}((global top_id += 1),t,e,ei)
end
function LagrangeTriangles{M}(id::Int,t::SimplexTopology{3},e=edges(t),ei=edgesindices(t,e)) where M
    np,ne,nc = totalnodes(t),totalnodes(ei),totalelements(t)*simplexnumber(2,M-2)
    I = OneTo(np+(M-1)*ne+nc)
    i = iscover(t) ? I : lagrangevertices3(t,ei,np,ne,Val(M))
    LagrangeTriangles{M}(id,t,e,edgesindices(t,e),i,iscover(t) ? I : collect(I))
end

function LagrangeTetrahedra{M}(t::SimplexTopology{4}) where M
    LagrangeTetrahedra{M}((global top_id += 1),t)
end
function LagrangeTetrahedra{M}(id::Int,t::SimplexTopology{4}) where M
    e = edges(t)
    f,fi = _facetsindices(t)
    LagrangeTetrahedra{M}(id,t,f,e,fi,edgesindices(t,e))
end
function LagrangeTetrahedra{M}(t::SimplexTopology{4},f,e,fi,ei) where M
    LagrangeTetrahedra{M}((global top_id += 1),t,f,e,fi,ei)
end
function LagrangeTetrahedra{M}(id::Int,t::SimplexTopology{4},f,e,fi,ei) where M
    np,ne,nc = totalnodes(t),totalnodes(ei),totalelements(t)*simplexnumber(2,M-2)
    I = OneTo(np+(M-1)*ne+nc)
    i = iscover(t) ? I : lagrangevertices4(t,ei,np,ne,Val(M))
    LagrangeTetrahedra{M}(id,t,f,e,fi,ei,i,iscover(t) ? I : collect(I))
end

cornertopology(m::LagrangeTopology) = m.t
edges(m::LagrangeEdges) = m.t
edges(m::LagrangeTriangles) = m.e
edges(m::LagrangeTetrahedra) = m.e
edgesindices(m::LagrangeEdges) = Values.(subelements(edges(m))) # refine later
edgesindices(m::LagrangeTriangles) = m.ei
edgesindices(m::LagrangeTetrahedra) = m.ei
facets(m::LagrangeEdges) = Values.(vertices(edges(m))) # refine later
facets(m::LagrangeTriangles) = m.e
facets(m::LagrangeTetrahedra) = m.f
facetsindices(m::LagrangeEdges) = facets(m)
facetsindices(m::LagrangeTriangles) = m.ei
facetsindices(m::LagrangeTetrahedra) = m.fi

totalnodes(m::LagrangeEdges) = totalcornernodes(m)+totaledgesnodes(m)
totalnodes(m::LagrangeTriangles) = totalcornernodes(m)+totaledgesnodes(m)+totalcenternodes(m)
totalnodes(m::LagrangeTetrahedra) = totalcornernodes(m)+totaledgesnodes(m)+totalfacetsnodes(m)+totalcenternodes(m)
nodes(m::LagrangeEdges) = cornernodes(m)+edgesnodes(m)
nodes(m::LagrangeTriangles) = cornernodes(m)+edgesnodes(m)+centernodes(m)
nodes(m::LagrangeTetrahedra) = cornernodes(m)+edgesnodes(m)+facetsnodes(m)+centernodes(m)

lagrangevertices2(t,M::Val) = lagrangevertices2(t,totalnodes(t),M)
lagrangevertices2(t,np::Int,M::Val{1}) = vertices(t)
function lagrangevertices2(t,np::Int,M::Val)
    vcat(vertices(t),centerindex(subelements(t),np,M))
end

lagrangevertices3(t,ei,M::Val) = lagrangevertices3(t,ei,totalnodes(t),totalnodes(ei),M)
lagrangevertices3(t,ei,np::Int,ne::Int,M::Val{1}) = vertices(t)
function lagrangevertices3(t,ei,np::Int,ne::Int,M::Val)
    vcat(vertices(t),edgesindex(vertices(ei),np,M),centerindex(subelements(t),np,ne,M))
end

lagrangevertices4(t,ei,fi,M::Val) = lagrangevertices4(t,ei,fi,totalnodes(t),totalnodes(ei),totalnodes(fi),M)
lagrangevertices4(t,ei,fi,np::Int,ne::Int,nf::Int,M::Val{1}) = vertices(t)
function lagrangevertices4(t,ei,fi,np::Int,ne::Int,nf::Int,M::Val)
    vcat(vertices(t),edgesindex(vertices(ei),np,M),
        facetsindex(vertices(fi),np,ne,M),centerindex(subelements(t),np,ne,nf,M))
end

@generated function edgesindex(ei,np,::Val{M}) where M
    isone(M) && (return typeof(ei)<:Values ? Values{0,Int}() : Vector{Int}())
    M == 2 && (return :(ei.+np))
    es = M-1
    Expr(:block,:(Mei = $es*ei),Expr(:call,:vcat,
        [:(Mei.+(np-$(es-i))) for i ∈ list(1,es-1)]...,:(Mei.+np)))
end
@generated function edgesindex(ei::Values{N},σ::Values{N},np,::Val{M}) where {M,N}
    isone(M) && (return Values{0,Int}())
    M == 2 && (return :(ei.+np))
    es = M-1
    Expr(:block,:(Mei = $es*(ei.-1).+np),:((f,r) = ($(list(1,es)),$(reverse(list(1,es))))),Expr(:call,:vcat,
            [:(@inbounds Mei[$i] .+ (isone(σ[$i]) ? f : r)) for i ∈ list(1,N)]...))
end
@generated function facetsindex(fi,np,ne,::Val{M}) where M
    (isone(M) || M==2) && (return typeof(fi)<:Values ? Values{0,Int}() : Vector{Int}())
    M == 3 && (return :(fi.+(np+$(M-1)*ne)))
    es,fs = M-1,facetsimplex(4,M)
    Expr(:block,:(Mfi = $fs*fi),:(n = np+$es*ne),
        Expr(:call,:vcat,[:(Mfi.+(n-$(fs-i))) for i ∈ list(1,fs-1)]...,:(Mfi.+n)))
end
@generated function centerindex(i::AbstractVector,np,::Val{M}) where M
    isone(M) && (return Vector{Int}())
    cs = M-1 # centersimplex(2,M)
    M == 2 && (return :(i.+np))
    Expr(:block,:(csi = $cs*(i.-1)),
        Expr(:call,:vcat,[:(($i+np).+csi) for i ∈ list(1,cs)]...))
end
@generated function centerindex(i::AbstractVector,np,ne,::Val{M}) where M
    (isone(M) || M==2) && (return Vector{Int}())
    es,cs = M-1,centersimplex(3,M)
    M == 3 && (return :(i.+(np+$es*ne)))
    Expr(:block,:(csi = $cs*(i.-1)),:(n = np+$es*ne),
        Expr(:call,:vcat,[:(($i+n).+csi) for i ∈ list(1,cs)]...))
end
@generated function centerindex(i::AbstractVector,np,ne,nf,::Val{M}) where M
    (isone(M) || M==2 || M==3) && (return Vector{Int}())
    es,cs,fs = M-1,centersimplex(4,M),facetsimplex(4,M)
    M == 4 && (return :(i.+(np+$es*ne+$fs*nf)))
    Expr(:block,:(csi = $cs*(i.-1)),:(n = np+$es*ne+$fs*nf),
        Expr(:call,:vcat,[:(($i+n).+csi) for i ∈ list(1,cs)]...))
end
@generated function centerindex(i::Int,np,::Val{M}) where M # N = 2
    isone(M) && (return Values{0,Int}())
    cs = M-1 # centersimplex(2,M)
    M == 2 && (return :(Values(np+i)))
    return :($(list(1,cs)).+(np+$cs*(i-1)))
end
@generated function centerindex(i::Int,np,ne,::Val{M}) where M # N = 3
    (isone(M) || M==2) && (return Values{0,Int}())
    es,cs = M-1,centersimplex(3,M)
    M == 3 && (return :(Values((np+$es*ne)+i)))
    return :($(list(1,cs)).+((np+$es*ne)+$cs*(i-1)))
end
@generated function centerindex(i::Int,np,ne,nf,::Val{M}) where M # N = 4
    (isone(M) || M==2 || M==3) && (return Values{0,Int}())
    es,cs,fs = M-1,centersimplex(4,M),facetsimplex(4,M)
    M == 4 && (return :(Values((np+$es*ne+$fs*nf)+i)))
    return :($(list(1,cs)).+((np+$es*ne+$fs*nf)+$cs*(i-1)))
end
function getedge(m::LagrangeTopology{M},i::Int) where M
    edgesindex(Values(getfacet(edges(m),i)),totalnodes(cornertopology(m)),Val(M))
end
getlagrange1(m::LagrangeTopology,i::Int) = cornertopology(m)[i]
function getlagrange2(m::LagrangeTopology{M},i::Int) where M
    t,e = cornertopology(m),edgesindices(m)
    np,ind = totalnodes(t),getfacet(t,i)
    ti,ei = fulltopology(t)[ind],fulltopology(e)[ind]
    vcat(ti,edgesindex(ei,np,Val(M)))
end
function getlagrange3(m::LagrangeTopology{M},i::Int) where M
    t,e = cornertopology(m),edgesindices(m)
    np,ne,ind = totalnodes(t),totalnodes(e),getfacet(t,i)
    ti,ei = fulltopology(t)[ind],fulltopology(e)[ind]
    vcat(ti,edgesindex(ei,edgesigns(ti),np,Val(M)),centerindex(i,np,ne,Val(M)))
end
function getlagrange4(m::LagrangeTetrahedra{M},i::Int) where M
    t,e,f = cornertopology(m),edgesindices(m),facetsindices(m)
    np,ne,nf,ind,N = totalnodes(t),totalnodes(e),totalnodes(f),getfacet(t,i),Val(M)
    ti,ei,fi = fulltopology(t)[ind],fulltopology(e)[ind],fulltopology(f)[ind]
    vcat(ti,edgesindex(ei,np,N),facetsindex(fi,np,ne,N),centerindex(i,np,ne,nf,N))
end
Base.getindex(m::LagrangeEdges{1},i::Int) = getlagrange1(m,i)
Base.getindex(m::LagrangeEdges{M},i::Int) where M = getlagrange2(m,i)
Base.getindex(m::LagrangeTriangles{1},i::Int) = getlagrange1(m,i)
Base.getindex(m::LagrangeTriangles{2},i::Int) = getlagrange2(m,i)
Base.getindex(m::LagrangeTriangles{M},i::Int) where M = getlagrange3(m,i)
Base.getindex(m::LagrangeTetrahedra{1},i::Int) = getlagrange1(m,i)
Base.getindex(m::LagrangeTetrahedra{2},i::Int) = getlagrange2(m,i)
Base.getindex(m::LagrangeTetrahedra{3},i::Int) = getlagrange4(m,i)
Base.getindex(m::LagrangeTetrahedra{M},i::Int) where M = getlagrange4(m,i)

function fullimmersion(m::LagrangeEdges{M}) where M
    ind = fullimmersion_vertices(m)
    LagrangeEdges{M}(bundle(m),fullimmersion(cornertopology(m)),ind,ind)
end
function fullimmersion(m::LagrangeTriangles{M}) where M
    ind = fullimmersion_vertices(m)
    LagrangeTriangles{M}(bundle(m),fullimmersion(cornertopology(m)),fullimmersion(edges(m)),fullimmersion(edgesindices(m)),ind,ind)
end
function fullimmersion(m::LagrangeTetrahedra{M}) where M
    ind = fullimmersion_vertices(m)
    LagrangeTetrahedra{M}(bundle(m),fullimmersion(cornertopology(m)),fullimmersion(facets(m)),fullimmersion(edges(m)),fullimmersion(facetsindices(m)),fullimmersion(edgesindices(m)),ind,ind)
end

function Base.getindex(m::LagrangeEdges{M},i::AbstractVector{Int}) where M
    t = cornertopology(m)[i]
    I = fullvertices(m)
    i = iscover(t) ? I : lagrangevertices2(t,Val(M))
    LagrangeEdges{M}(bundle(m),t,i,I)
end
function Base.getindex(m::LagrangeTriangles{M},i::AbstractVector{Int}) where M
    t = cornertopology(m)[i]
    ei = edgesindices(m)[i]
    e = edges(m)[vertices(ei)]
    I = fullvertices(m)
    i = iscover(t) ? I : lagrangevertices3(t,ei,Val(M))
    LagrangeTriangles{M}(bundle(m),t,e,ei,i,I)
end
function Base.getindex(m::LagrangeTetrahedra{M},i::AbstractVector{Int}) where M
    t = cornertopology(m)[i]
    fi = refine(facetsindices(m)[i])
    ei = edgesindices(m)[i]
    f = refine(facets(m)[vertices(fi)])
    e = edges(m)[vertices(ei)]
    I = fullvertices(m)
    i = iscover(t) ? I : lagrangevertices4(t,ei,fi,Val(M))
    LagrangeTetrahedra{M}(bundle(m),t,f,e,fi,ei,i,I)
end

(m::LagrangeTopology)(i::AbstractVector{Int}) = subtopology(m,i)
#=function subtopology(m::LagrangeTopology{N},i::AbstractVector{Int}) where N
    t = cornertopology(m)(i)
    ei = edgesindices(m)[subelements(t)]
    e = edges(m)[vertices(ei)]
    LagrangeTopology(bundle(m),subtopology(SimplexTopology(m),i),fullvertices(m))
end=#

function _getelement(t::ImmersedTopology,ind::Values)
    if iscover(t)
        fulltopology(t)[ind]
    else
        getindex.(Ref(verticesinv(t)),fulltopology(t)[ind])
    end
end

getelement1(m::LagrangeTopology{M,N,<:OneTo},i::Int) where {M,N} = m[i]
getelement1(m::LagrangeTopology{M,N,<:AbstractVector},i::Int) where {M,N} = getelement(cornertopology(m),i)
getelement2(m::LagrangeTopology{M,N,<:OneTo},i::Int) where {M,N} = m[i]
function getelement2(m::LagrangeTopology{M,N,<:AbstractVector},i::Int) where {M,N}
    t,e = cornertopology(m),edgesindices(m)
    np,ind = nodes(t),getfacet(t,i)
    ti,ei = _getelemeent(t,ind),fulltopology(e)[ind]
    vcat(ti,edgesindex(ei,np,Val(M)))
end
getelement3(m::LagrangeTopology{M,N,<:OneTo},i::Int) where {M,N} = m[i]
function getelement3(m::LagrangeTopology{M,N,<:AbstractVector},i::Int) where {M,N}
    t,e = cornertopology(m),edgesindices(m)
    np,ne,ind = nodes(t),nodes(e),getfacet(t,i)
    ti,ei = _getelemeent(t,ind),fulltopology(e)[ind]
    vcat(ti,edgesindex(ei,np,Val(M)),centerindex(i,np,ne,Val(M)))
end
getelement4(m::LagrangeTopology{M,N,<:OneTo},i::Int) where {M,N} = m[i]
function getelement4(m::LagrangeTetrahedra{M,N,<:AbstractVector} where N,i::Int) where M
    t,e,f = cornertopology(m),edgesindices(m),facetsindices(m)
    np,ne,nf,ind,N = nodes(t),nodes(e),nodes(f),getfacet(t,i),Val(M)
    ti,ei,fi = _getelement(t,ind),fulltopology(e)[ind],fulltopology(f)[ind]
    vcat(ti,edgesindex(ei,np,N),facetsindex(fi,np,ne,N),centerindex(i,np,ne,nf,N))
end
getelement(m::LagrangeEdges{1},i::Int) = getelement1(m,i)
getelement(m::LagrangeEdges{M},i::Int) where M = getelement2(m,i)
getelement(m::LagrangeTriangles{1},i::Int) = getelement1(m,i)
getelement(m::LagrangeTriangles{2},i::Int) = getelement2(m,i)
getelement(m::LagrangeTriangles{M},i::Int) where M = getelment3(m,i)
getelement(m::LagrangeTetrahedra{1},i::Int) = getelement1(m,i)
getelement(m::LagrangeTetrahedra{2},i::Int) = getelement2(m,i)
getelement(m::LagrangeTetrahedra{3},i::Int) = getelement3(m,i)
getelement(m::LagrangeTetrahedra{M},i::Int) where M = getelement4(m,i)

function subimmersion(m::LagrangeEdges{M,N,<:OneTo} where {M,N})
    iscover(m) && (return m)
    ind = vertices(m)
    LagrangeEdges{M}(0,subimmersion(cornertopology(m)),ind,ind)
end
function subimmersion(m::LagrangeEdges{M,N,<:AbstractVector} where {M,N})
    iscover(m) && (return m)
    ver = OneTo(nodes(m))
    LagrangeEdges{M}(0,subimmersion(cornertopology(m)),ver,ver)
end

function subimmersion(m::LagrangeTriangles{M,N,<:OneTo} where {M,N})
    iscover(m) && (return m)
    ind = vertices(m)
    LagrangeTriangles{M}(0,subimmersion(cornertopology(m)),subimmersion(edges(m)),subimmersion(edgesindices(m)),ind,ind)
end
function subimmersion(m::LagrangeTriangles{M,N,<:AbstractVector} where {M,N})
    iscover(m) && (return m)
    ver = OneTo(nodes(m))
    LagrangeTriangles{M}(0,subimmersion(cornertopology(m)),subimmersion(edges(m)),subimmersion(edgesindices(m)),ver,ver)
end

function subimmersion(m::LagrangeTetrahedra{M,N,<:OneTo} where {M,N})
    iscover(m) && (return m)
    ind = vertices(m)
    LagrangeTriangles{M}(0,subimmersion(cornertopology(m)),subimmersion(facets(m)),subimmersion(edges(m)),subimmersion(facetsindices(m)),subimmersion(edgesindices(m)),ind,ind)
end
function subimmersion(m::LagrangeTetrahedra{M,N,<:AbstractVector} where {M,N})
    iscover(m) && (return m)
    ver = OneTo(nodes(m))
    LagrangeTetrahedra{M}(0,subimmersion(cornertopology(m)),subimmersion(facets(m)),subimmersion(edges(m)),subimmersion(facetsindices(m)),subimmersion(edgesindices(m)),ver,ver)
end

refine(m::LagrangeEdges{M,N,<:AbstractVector,<:AbstractVector}) where {M,N} = m
function refine(m::LagrangeEdges{M,N,<:OneTo,<:AbstractVector}) where {M,N}
    i = collect(vertices(m))
    fi = vertices(m)≠fullvertices(m) ? collect(fullvertices(m)) : i
    LagrangeEdges{M}(bundle(m),refine(cornertopology(m)),i,fi)
end
function refine(m::LagrangeEdges{M,N,<:AbstractVector,<:OneTo}) where {M,N}
    LagrangeEdges{M}(bundle(m),refine(cornertopology(m)),vertices(m),fullvertices(m))
end
function refine(m::LagrangeEdges{M,N,<:OneTo,<:OneTo}) where {M,N}
    i = collect(vertices(m))
    fi = vertices(m)≠fullvertices(m) ? collect(fullvertices(m)) : i
    LagrangeEdges{M}(bundle(m),refine(cornertopology(m)),i,fi)
end

refine(m::LagrangeTriangles{M,N,<:AbstractVector,<:AbstractVector}) where {M,N} = m
function refine(m::LagrangeTriangles{M,N,<:OneTo,<:AbstractVector}) where {M,N}
    i = collect(vertices(m))
    fi = vertices(m)≠fullvertices(m) ? collect(fullvertices(m)) : i
    LagrangeTriangles{M}(bundle(m),refine(cornertopology(m)),refine(edges(m)),refine(edgesindices(m)),i,fi)
end
function refine(m::LagrangeTriangles{M,N,<:AbstractVector,<:OneTo}) where {M,N}
    LagrangeTriangles{M}(bundle(m),refine(cornertopology(m)),refine(edges(m)),refine(edgesindices(m)),vertices(m),fullvertices(m))
end
function refine(m::LagrangeTriangles{M,N,<:OneTo,<:OneTo}) where {M,N}
    i = collect(vertices(m))
    fi = vertices(m)≠fullvertices(m) ? collect(fullvertices(m)) : i
    LagrangeTriangles{M}(bundle(m),refine(cornertopology(m)),refine(edges(m)),refine(edgesindices(m)),i,fi)
end

refine(m::LagrangeTetrahedra{M,N,<:AbstractVector,<:AbstractVector}) where {M,N} = m
function refine(m::LagrangeTetrahedra{M,N,<:OneTo,<:AbstractVector}) where {M,N}
    i = collect(vertices(m))
    fi = vertices(m)≠fullvertices(m) ? collect(fullvertices(m)) : i
    LagrangeTetrahedra{M}(bundle(m),refine(cornertopology(m)),refine(facets(m)),refine(edges(m)),refine(facetsindices(m)),refine(edgesindices(m)),i,fi)
end
function refine(m::LagrangeTetrahedra{M,N,<:AbstractVector,<:OneTo}) where {M,N}
    LagrangeTetrahedra{M}(bundle(m),refine(cornertopology(m)),refine(facets(m)),refine(edges(m)),refine(facetsindices(m)),refine(edgesindices(m)),vertices(m),fullvertices(m))
end
function refine(m::LagrangeTetrahedra{M,N,<:OneTo,<:OneTo}) where {M,N}
    i = collect(vertices(m))
    fi = vertices(m)≠fullvertices(m) ? collect(fullvertices(m)) : i
    LagrangeTetrahedra{M}(bundle(m),refine(cornertopology(m)),refine(facets(m)),refine(edges(m)),refine(facetsindices(m)),refine(edgesindices(m)),i,fi)
end

# Common

_axes(t::ImmersedTopology{N}) where N = (Base.OneTo(length(t)),Base.OneTo(N))

for top ∈ (:SimplexTopology,:DiscontinuousTopology,:LagrangeTopology)#,:VectorTopology)
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

