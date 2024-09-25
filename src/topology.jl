
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

export FiberProduct, FiberProductBundle, HomotopyBundle
export LocalSection, GlobalFiber, LocalFiber
export base, fiber, domain, codomain, ↦, →, ←, ↤, basetype, fibertype, graph
export ProductSpace, RealRegion, NumberLine, Rectangle, Hyperrectangle, ⧺, ⊕

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

@generated Base.size(m::RealRegion{V}) where V = :(($([:(size(m.v[$i])...) for i ∈ 1:mdims(V)]...),))
@generated Base.getindex(m::RealRegion{V,T,N},i::Vararg{Int}) where {V,T,N} = :(Chain{V,1,T}(Values{N,T}($([:(m.v[$j][i[$j]]) for j ∈ 1:N]...))))
Base.getindex(m::NumberLine{V,T},i::Int) where {V,T} = Chain{V,1,T}(Values((m.v[1][i],)))
@pure Base.getindex(t::RealRegion,i::CartesianIndex) = getindex(t,i.I...)
@pure Base.eltype(::Type{ProductSpace{V,T,N}}) where {V,T,N} = Chain{V,1,T,N}

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

⊕(a::AbstractVector{A},b::AbstractVector{B}) where {A<:Real,B<:Real} = RealRegion(Values(a,b))

@generated ⧺(a::Real...) = :(Chain($([:(a[$i]) for i ∈ 1:length(a)]...)))
@generated ⧺(a::Complex...) = :(Chain($([:(a[$i]) for i ∈ 1:length(a)]...)))
⧺(a::Chain{A,G},b::Chain{B,G}) where {A,B,G} = Chain{A∪B,G}(vcat(a.v,b.v))

remove(t::ProductSpace{V,T,2} where {V,T},::Val{1}) = t.v[2]
remove(t::ProductSpace{V,T,2} where {V,T},::Val{2}) = t.v[1]
@generated remove(t::ProductSpace{V,T,N} where {V,T},::Val{J}) where {N,J} = :(ProductSpace(domain(t).v[$(Values([i for i ∈ 1:N if i≠J]...))]))

# ImmersedTopology

export ImmersedTopology, SimplexManifold, topology, immersion, vertices, iscover

abstract type ImmersedTopology{N,P} <: AbstractVector{Values{N,Int}} end
const immersion = ImmersedTopology

top_id = 0

struct SimplexManifold{N,P<:AbstractVector{Int}} <: ImmersedTopology{N,P}
    id::Int
    t::Vector{Values{N,Int}}
    i::P
    p::Int
    SimplexManifold(t::Vector{Values{N,Int}},i::P=vertices(t),p::Int=length(i)) where {N,P} = new{N,P}((global top_id+=1),t,i,p)
end

SimplexManifold(t::Vector{Values{N,Int}},p::Int) where N = SimplexManifold(t,vertices(t),p)

bundle(m::SimplexManifold) = m.id
topology(m::SimplexManifold) = m.t
vertices(m::SimplexManifold) = m.i

Base.size(m::SimplexManifold) = size(topology(m))
Base.length(m::SimplexManifold) = length(topology(m))
Base.axes(m::SimplexManifold) = axes(topology(m))
Base.getindex(m::SimplexManifold,i::Int) = getindex(topology(m),i)
@pure Base.eltype(::Type{ImmersedTopology{N}}) where N = Values{N,Int}
Grassmann.mdims(m::SimplexManifold{N}) where N = N

_axes(t::SimplexManifold{N}) where N = (Base.OneTo(length(t)),Base.OneTo(N))

# anything array-like gets summarized e.g. 10-element Array{Int64,1}
Base.summary(io::IO, a::SimplexManifold) = Base.array_summary(io, a, _axes(a))

iscover(x::ImmersedTopology) = length(vertices(x)) == x.p
subsym(x) = iscover(x) ? "⊆" : "⊂"

function Base.array_summary(io::IO, a::SimplexManifold, inds::Tuple{Vararg{OneTo}})
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

#metrictensor(c::AbstractArray{T,N} where T) where N = Global{N}(InducedMetric(),size(c))
ref(itr::InducedMetric) = Ref(itr)
ref(itr::Global) = Ref(itr.v)
ref(itr) = itr

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
metrictensor(c) = InducedMetric()
metrictensor(c::Coordinate) = fiber(c)

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

@inline Base.:<<(a::LocalFiber,b::LocalFiber) = contraction(b,~a)
@inline Base.:>>(a::LocalFiber,b::LocalFiber) = contraction(~a,b)
@inline Base.:<(a::LocalFiber,b::LocalFiber) = contraction(b,a)
for type ∈ (:Coordinate,:LocalSection,:LocalTensor)
    for tensor ∈ (:Single,:Couple,:PseudoCouple,:Chain,:Spinor,:AntiSpinor,:Multivector,:DiagonalOperator,:TensorOperator,:Outermorphism)
        @eval (T::Type{<:$tensor})(s::$type) = $type(base(s), T(fiber(s)))
    end
    for fun ∈ (:-,:!,:~,:real,:imag,:conj,:deg2rad)
        @eval Base.$fun(s::$type) = $type(base(s), $fun(fiber(s)))
    end
    for fun ∈ (:inv,:exp,:log,:sinh,:cosh,:abs,:sqrt,:cbrt,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:asech,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:abs2)
        @eval Base.$fun(s::$type) = $type(base(s), $fun(fiber(s),metrictensor(base(s))))
    end
    for fun ∈ (:reverse,:involute,:clifford,:even,:odd,:scalar,:vector,:bivector,:volume,:value,:curl,:∂,:d,:complementleft,:realvalue,:imagvalue,:outermorphism)
        @eval Grassmann.$fun(s::$type) = $type(base(s), $fun(fiber(s)))
    end
    for fun ∈ (:⋆,:angle,:radius,:complementlefthodge,:pseudoabs,:pseudoabs2,:pseudoexp,:pseudolog,:pseudoinv,:pseudosqrt,:pseudocbrt,:pseudocos,:pseudosin,:pseudotan,:pseudocosh,:pseudosinh,:pseudotanh,:metric)
        @eval Grassmann.$fun(s::$type) = $type(base(s), $fun(fiber(s),metrictensor(base(s))))
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
            $bop(a::$type{R},b::$type{R}) where R = $type(base(a),Grassmann.$mop(fiber(a),fiber(b),metrictensor(base(a))))
            $bop(a::Number,b::$type) = $type(base(b), Grassmann.$op(a,fiber(b)))
            $bop(a::$type,b::Number) = $type(base(a), Grassmann.$op(fiber(a),b,$((op≠:^ ? () : (:(metrictensor(base(a))),))...)))
        end end
    end
    @eval begin
        $type(b,f::Function) = $type(b,f(b))
        Grassmann.contraction(a::$type{R},b::$type{R}) where R = $type(base(a),Grassmann.contraction(fiber(a),fiber(b)))
        LinearAlgebra.norm(s::$type) = $type(base(s), norm(fiber(s)))
        LinearAlgebra.det(s::$type) = $type(base(s), det(fiber(s)))
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

points(m::PointArray) = m.dom
metrictensor(m::PointArray) = m.cod
pointtype(m::PointArray) = basetype(m)
pointtype(m::Type{<:PointArray}) = basetype(m)
metrictype(m::PointArray) = fibertype(m)
metrictype(m::Type{<:PointArray}) = fibertype(m)
basetype(::PointArray{B}) where B = B
basetype(::Type{<:PointArray{B}}) where B = B
fibertype(::PointArray{B,F} where B) where F = F
fibertype(::Type{<:PointArray{B,F} where B}) where F = F

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

Base.size(m::PointArray) = size(m.dom)
Base.firstindex(m::PointCloud) = 1
Base.lastindex(m::PointCloud) = length(points(m))
Base.length(m::PointCloud) = length(points(m))
Base.resize!(m::PointCloud,i::Int) = (resize!(points(m),i),resize!(metrictensor(m),i))
Base.broadcast(f,t::PointArray) = PointArray(f.(points(t)),f.(metrictensor(t)))
Base.broadcast(f,t::PointCloud) = PointCloud(f.(points(t)),f.(metrictensor(t)))

@pure Base.eltype(::Type{<:PointArray{P,G}}) where {P,G} = Coordinate{P,G}
function Base.getindex(m::PointArray,i::Vararg{Int})
    Coordinate(getindex(points(m),i...), getindex(metrictensor(m),i...))
end
Base.setindex!(m::PointArray{P},s::P,i::Vararg{Int}) where P = setindex!(points(m),s,i...)
Base.setindex!(m::PointArray{P,G} where P,s::G,i::Vararg{Int}) where G = setindex!(metrictensor(m),s,i...)
function Base.setindex!(m::PointArray,s::Coordinate,i::Vararg{Int})
    setindex!(points(m),point(s),i...)
    setindex!(metrictensor(m),metrictensor(s),i...)
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
    PointArray(similar(Array{ElType,N}, axes(bc)), metrictensor(t))
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

topology(m::GlobalFiber) = topology(immersion(m))
vertices(m::GlobalFiber) = vertices(immersion(m))
iscover(m::GlobalFiber) = iscover(immersion(m))
imagepoints(m::GlobalFiber) = iscover(m) ? points(m) : points(m)[vertices(m)]

unitdomain(t::GlobalFiber) = base(t)*inv(base(t)[end])
arcdomain(t::GlobalFiber) = unitdomain(t)*arclength(codomain(t))
graph(t::GlobalFiber) = graph.(t)

Base.size(m::GlobalFiber) = size(m.cod)
Base.resize!(m::GlobalFiber,i) = (resize!(domain(m),i),resize!(codomain(m),i))

# AbstractFrameBundle

export AbstractFrameBundle, GridFrameBundle, SimplexFrameBundle, FacetFrameBundle
export IntervalRange, AlignedRegion, AlignedSpace

abstract type AbstractFrameBundle{M,N} <: GlobalFiber{M,N} end

Base.size(m::AbstractFrameBundle) = size(points(m))

@pure Grassmann.Manifold(m::AbstractFrameBundle) = Manifold(points(m))
@pure LinearAlgebra.rank(m::AbstractFrameBundle) = rank(points(m))
@pure Grassmann.grade(m::AbstractFrameBundle) = grade(points(m))
@pure Grassmann.antigrade(m::AbstractFrameBundle) = antigrade(points(m))
@pure Grassmann.mdims(m::AbstractFrameBundle) = mdims(points(m))

# GridFrameBundle

grid_id = 0

struct GridFrameBundle{C<:Coordinate,N,PA<:FiberBundle{C,N}} <: AbstractFrameBundle{C,N}
    id::Int
    dom::PA
    GridFrameBundle(id::Int,p::PA) where {C<:Coordinate,N,PA<:FiberBundle{C,N}} = new{C,N,PA}(id,p)
    GridFrameBundle(p::PA) where {C<:Coordinate,N,PA<:FiberBundle{C,N}} = new{C,N,PA}((global grid_id+=1),p)
end

GridFrameBundle(id::Int,p::PA,g::GA) where {P,G,N,PA<:AbstractArray{P,N},GA<:AbstractArray{G,N}} = GridFrameBundle(id,PointArray(0,p,g))
GridFrameBundle(p::PA,g::GA) where {P,G,N,PA<:AbstractArray{P,N},GA<:AbstractArray{G,N}} = GridFrameBundle((global grid_id+=1),p,g)

const IntervalRange{P<:Real,G,PA<:AbstractRange,GA} = GridFrameBundle{Coordinate{P,G},1,<:PointVector{P,G,PA,GA}}
const AlignedRegion{N,P<:Chain,G<:InducedMetric,PA<:RealRegion{V,<:Real,N,<:AbstractRange} where V,GA<:Global} = GridFrameBundle{Coordinate{P,G},N,PointArray{P,G,N,PA,GA}}
const AlignedSpace{N,P<:Chain,G<:InducedMetric,PA<:RealRegion{V,<:Real,N,<:AbstractRange} where V,GA} = GridFrameBundle{Coordinate{P,G},N,PointArray{P,G,N,PA,GA}}

GridFrameBundle(id::Int,p::PA) where {N,P,PA<:AbstractArray{P,N}} = GridFrameBundle(id,PointArray(0,p,Global{N}(InducedMetric())))
GridFrameBundle(p::PA) where {N,P,PA<:AbstractArray{P,N}} = GridFrameBundle(p,Global{N}(InducedMetric()))
GridFrameBundle(dom::GridFrameBundle,fun) = GridFrameBundle(base(dom), fun)
GridFrameBundle(dom::GridFrameBundle,fun::Array) = GridFrameBundle(base(dom), fun)
GridFrameBundle(dom::GridFrameBundle,fun::Function) = GridFrameBundle(base(dom), fun)
GridFrameBundle(dom::AbstractArray,fun::Function) = GridFrameBundle(dom, fun.(dom))

base(m::GridFrameBundle) = m.dom
points(m::GridFrameBundle) = points(base(m))
metrictensor(m::GridFrameBundle) = metrictensor(base(m))
coordinates(t::GridFrameBundle) = t
pointtype(m::GridFrameBundle) = basetype(m)
pointtype(m::Type{<:GridFrameBundle}) = basetype(m)
metrictype(m::GridFrameBundle) = fibertype(m)
metrictype(m::Type{<:GridFrameBundle}) = fibertype(m)
basetype(m::GridFrameBundle) = basetype(base(m))
basetype(::Type{<:GridFrameBundle{C}}) where C = basetype(C)
fibertype(m::GridFrameBundle) = fibertype(base(m))
fibertype(::Type{<:GridFrameBundle{C}}) where C = fibertype(C)

Base.resize!(m::GridFrameBundle,i) = resize!(base(m),i)
Base.broadcast(f,t::GridFrameBundle) = GridFrameBundle(f.(base(t)))

@pure Base.eltype(::Type{<:GridFrameBundle{C}}) where C = C
Base.getindex(m::GridFrameBundle,i::Vararg{Int}) = getindex(base(m),i...)
Base.setindex!(m::GridFrameBundle,s,i::Vararg{Int}) = setindex!(base(m),s,i...)

Base.BroadcastStyle(::Type{<:GridFrameBundle{C,N,PA}}) where {C,N,PA} = Broadcast.ArrayStyle{GridFrameBundle{C,N,PA}}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GridFrameBundle{C,N,PA}}}, ::Type{ElType}) where {C,N,PA,ElType<:Coordinate}
    ax = axes(bc)
    # Use the data type to create the output
    GridFrameBundle(similar(Array{pointtype(ElType),N}, ax), similar(Array{metrictype(ElType),N}, ax))
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GridFrameBundle{C,N,PA}}}, ::Type{ElType}) where {C,N,PA,ElType}
    t = find_gf(bc)
    # Use the data type to create the output
    GridFrameBundle(similar(Array{ElType,N}, axes(bc)), metrictensor(t))
end

"`A = find_gf(As)` returns the first GridFrameBundle among the arguments."
find_gf(bc::Base.Broadcast.Broadcasted) = find_gf(bc.args)
find_gf(bc::Base.Broadcast.Extruded) = find_gf(bc.x)
find_gf(args::Tuple) = find_gf(find_gf(args[1]), Base.tail(args))
find_gf(x) = x
find_gf(::Tuple{}) = nothing
find_gf(a::GridFrameBundle, rest) = a
find_gf(::Any, rest) = find_gf(rest)

# SimplexFrameBundle

struct SimplexFrameBundle{P,G,PA<:AbstractVector{P},GA<:AbstractVector{G},TA<:ImmersedTopology} <: AbstractFrameBundle{Coordinate{P,G},1}
    p::PointCloud{P,G,PA,GA}
    t::TA
    SimplexFrameBundle(p::PointCloud{P,G,PA,GA},t::T) where {P,G,PA,GA,T} = new{P,G,PA,GA,T}(p,t)
end

SimplexFrameBundle(id::Int,p,t,g) = SimplexFrameBundle(PointCloud(id,p,g),t)
SimplexFrameBundle(id::Int,p,t) = SimplexFrameBundle(PointCloud(id,p),t)
#SimplexFrameBundle(p::AbstractVector,t) = SimplexFrameBundle(PointCloud(p),t)

(p::PointCloud)(t::ImmersedTopology) = length(p)≠length(t) ? SimplexFrameBundle(p,t) : FacetFrameBundle(p,t)
PointCloud(m::SimplexFrameBundle) = m.p
ImmersedTopology(m::SimplexFrameBundle) = m.t
coordinates(t::SimplexFrameBundle) = t
points(m::SimplexFrameBundle) = points(PointCloud(m))
metrictensor(m::SimplexFrameBundle) = metrictensor(PointCloud(m))
pointtype(m::SimplexFrameBundle) = basetype(m)
pointtype(m::Type{<:SimplexFrameBundle}) = basetype(m)
metrictype(m::SimplexFrameBundle) = fibertype(m)
metrictype(m::Type{<:SimplexFrameBundle}) = fibertype(m)
basetype(::SimplexFrameBundle{P}) where P = pointtype(P)
basetype(::Type{<:SimplexFrameBundle{P}}) where P = pointtype(P)
fibertype(::SimplexFrameBundle{P}) where P = metrictype(P)
fibertype(::Type{<:SimplexFrameBundle{P}}) where P = metrictype(P)
Base.size(m::SimplexFrameBundle) = size(vertices(m))

Base.broadcast(f,t::SimplexFrameBundle) = SimplexFrameBundle(f.(PointCloud(t)),ImmersedTopology(t))

(m::SimplexFrameBundle)(i::ImmersedTopology) = SimplexFrameBundle(bundle(m),points(m),i,metrictensor(m))
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
    Coordinate(getindex(points(m),ind), getindex(metrictensor(m),ind))
end
Base.setindex!(m::SimplexFrameBundle{P},s::P,i::Int) where P = setindex!(points(m),s,getimage(m,i))
Base.setindex!(m::SimplexFrameBundle{P,G} where P,s::G,i::Int) where G = setindex!(metrictensor(m),s,getimage(m,i))
function Base.setindex!(m::SimplexFrameBundle,s::Coordinate,i::Int)
    ind = getimage(m,i)
    setindex!(points(m),point(s),ind)
    setindex!(metrictensor(m),metrictensor(s),ind)
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

PointCloud(m::FacetFrameBundle) = m.p
ImmersedTopology(m::FacetFrameBundle) = m.t
coordinates(t::FacetFrameBundle) = t
points(m::FacetFrameBundle) = points(PointCloud(m))
metrictensor(m::FacetFrameBundle) = metrictensor(PointCloud(m))
pointtype(m::FacetFrameBundle) = basetype(m)
pointtype(m::Type{<:FacetFrameBundle}) = basetype(m)
metrictype(m::FacetFrameBundle) = fibertype(m)
metrictype(m::Type{<:FacetFrameBundle}) = fibertype(m)
basetype(::FacetFrameBundle{P}) where P = pointtype(P)
basetype(::Type{<:FacetFrameBundle{P}}) where P = pointtype(P)
fibertype(::FacetFrameBundle{P}) where P = metrictype(P)
fibertype(::Type{<:FacetFrameBundle{P}}) where P = metrictype(P)

Base.broadcast(f,t::FacetFrameBundle) = FacetFrameBundle(0,f.(PointCloud(t)),ImmersedTopology(t))

function SimplexFrameBundle(p::P,t,g::G) where {P<:AbstractVector,G<:AbstractVector}
    SimplexFrameBundle(PointCloud(p,g),t)
end
function SimplexFrameBundle(m::FacetFrameBundle)
    SimplexFrameBundle(PointCloud(m.id,point_cache[m.id],point_metric_cache[m.id]),ImmersedTopology(m))
end
function FacetFrameBundle(m::SimplexFrameBundle)
    et = topology(ImmersedTopology(m))
    FacetFrameBundle(m.id,PointCloud(0,barycenter.(m[et]),barycenter.(getindex.(Ref(metrictensor(m)),et))),ImmersedTopology(m))
end

bundle(m::SimplexFrameBundle) = m.id
bundle(m::FacetFrameBundle) = m.id
deletebundle!(m::SimplexFrameBundle) = deletepointcloud!(bundle(m))
deletebundle!(m::FacetFrameBundle) = deletepointcloud!(bundle(m))
@pure isbundle(::AbstractFrameBundle) = true
@pure isbundle(t) = false
#@pure ispoints(t::Submanifold{V}) where V = isbundle(V) && rank(V) == 1 && !isbundle(Manifold(V))
#@pure ispoints(t) = isbundle(t) && rank(t) == 1 && !isbundle(Manifold(t))
#@pure islocal(t) = isbundle(t) && rank(t)==1 && valuetype(t)==Int && ispoints(Manifold(t))
#@pure iscell(t) = isbundle(t) && islocal(Manifold(t))

Base.firstindex(m::SimplexFrameBundle) = 1
Base.lastindex(m::SimplexFrameBundle) = length(vertices(m))
Base.length(m::SimplexFrameBundle) = length(vertices(m))
#Base.resize!(m::SimplexFrameBundle,n::Int) = resize!(value(m),n)

@pure Base.eltype(::Type{<:FacetFrameBundle{P,G}}) where {P,G} = Coordinate{P,G}
function Base.getindex(m::FacetFrameBundle,i::Int)
    ind = getimage(m,i)
    Coordinate(getindex(points(m),ind), getindex(metrictensor(m),ind))
end
Base.setindex!(m::FacetFrameBundle{P},s::P,i::Int) where P = setindex!(points(m),s,getimage(m,i))
Base.setindex!(m::FacetFrameBundle{P,G} where P,s::G,i::Int) where G = setindex!(metrictensor(m),s,getimage(m,i))
function Base.setindex!(m::FacetFrameBundle,s::Coordinate,i::Int)
    ind = getimage(m,i)
    setindex!(points(m),point(s),ind)
    setindex!(metrictensor(m),metrictensor(s),ind)
    return s
end

# FiberProductBundle

struct FiberProductBundle{P,N,SA<:AbstractArray,PA<:AbstractArray} <: AbstractFrameBundle{Coordinate{P,InducedMetric},N}
    dom::SA
    cod::PA
    FiberProductBundle{P}(s::SA,p::PA) where {P,M,N,SA<:AbstractArray{S,M} where S,PA<:AbstractArray{F,N} where F} = new{P,M+N,SA,PA}(s,p)
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

Base.size(m::FiberProductBundle) = (length(base(m)),size(fiber(m))...)

@pure Base.eltype(::Type{<:FiberProductBundle{P}}) where P = Coordinate{P,InducedMetric}
Base.getindex(m::FiberProductBundle,i::Int,j::Vararg{Int}) = Coordinate(getindex(points(base(m)),i) ⧺ getindex(fiber(m),j...), InducedMetric())
#=Base.setindex!(m::FiberProductBundle{P},s::P,i::Int,j::Vararg{Int}) where P = setindex!(points(m),s,i,j...)
Base.setindex!(m::FiberProductBundle{P,G} where P,s::G,i::Int,j::Vararg{Int}) where G = setindex!(metrictensor(m),s,i,j...)
function Base.setindex!(m::FiberProductBundle,s::Coordinate,i::Int,j::Vararg{Int})
    setindex!(points(m),point(s),i...)
    setindex!(metrictensor(m),metrictensor(s),i...)
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
ImmersedTopology(m::HomotopyBundle) = m.t
coordinates(t::HomotopyBundle) = t
points(m::HomotopyBundle) = points(FiberProduct(m))
fiberspace(m::HomotopyBundle) = fiberspace(FiberProduct(m))
pointtype(m::HomotopyBundle) = basetype(m)
pointtype(m::Type{<:HomotopyBundle}) = basetype(m)
basetype(::HomotopyBundle{P}) where P = pointtype(P)
basetype(::Type{<:HomotopyBundle{P}}) where P = pointtype(P)
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


