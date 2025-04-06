
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
export GlobalFiber, LocalFiber, localfiber, globalfiber
export base, fiber, domain, codomain, ↦, →, ←, ↤, basetype, fibertype, graph
export fullcoordinates, fullpoints, fullmetricextensor, isinduced
export pointtype, metrictype, coordinates, coordinatetype

# Global

export Global

"""
    Global{N,T} <: AbstractArray{T,N}

Represents an `AbstractArray` where every local value is globally the same.
```julia
julia> Global{1}(InducedMetric())
Global{1}(InducedMetric())

julia> ans[1]
InducedMetric()
```
For example, `Global{N,InducedMetric}` is commonly used for a globally induced metric.
"""
struct Global{N,T} <: AbstractArray{T,N}
    v::T
    #n::NTuple{N,Int}
    #Global{N}(v::T,n=(1,)) where {T,N} = new{N,T}(v,n)
    Global{N}(v::T) where {T,N} = new{N,T}(v)
end

#Base.size(t::Global) = t.n
Base.getindex(t::Global,i::Vararg{Int}) = t.v
Base.getindex(t::Global,i::CartesianIndex) = t.v
Base.setindex!(t::Global{N,InducedMetric} where N,v::InducedMetric,i::Vararg{Int}) = v
Base.resize!(t::Global,i) = t
@pure Base.eltype(::Type{<:Global{T}}) where T = T

Base.vec(t::Global{1}) = t
Base.vec(t::Global) = Global{1}(t.v)

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

"""
    LocalFiber{B,F} <: Number

Defines abstract bundled type with `basetype` of `B` and `fibertype` of `F` in a manifold.
```Julia
base(s) # ::B
fiber(s) # ::F
basetype(s) # B
fibertype(s) # F
```
A `LocalFiber{B,F}` consists of two components: `B`, which represents the `base` manifold, and `F`, which represents the `fiber` bundle over `B`.
"""
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
pointtype(::LocalFiber{B}) where B = basetype(B)
fibertype(::LocalFiber{B,F} where B) where F = F
metrictype(::LocalFiber{B,F} where B) where F = fibertype(B)
basetype(::Type{<:LocalFiber{B}}) where B = B
pointtype(::Type{<:LocalFiber{B}}) where B = basetype(B)
fibertype(::Type{<:LocalFiber{B,F} where B}) where F = F
metrictype(::Type{<:LocalFiber{B,F} where B}) where F = fibertype(B)

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

"""
    Coordinate{P,G} <: LocalFiber{P,G} <: Number

Defines a `Coordinate` bundled type with `pointtype` of `P` and `metrictype` of `G`.
```Julia
point(s) # ::P
metricextensor(s) # ::G
pointtype(s) # P
metrictype(s) # G
```
A `Coordinate{P,G}` consists of two components: `P`, which represents the `point` manifold, and `G`, which represents the `metricextensor` bundle over `P`.
"""
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
pointtype(::Coordinate{P}) where P = P
pointtype(::Type{<:Coordinate{P}}) where P = P
metrictype(::Coordinate{P,G} where P) where G = G
metrictype(::Type{<:Coordinate{P,G} where P}) where G = G

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

# LocalTensor

"""
    LocalTensor{B,F} <: LocalFiber{B,F} <: Number

Defines a local bundled type with `basetype` of `B` and `fibertype` of `F`.
```Julia
base(s) # ::B
fiber(s) # ::F
basetype(s) # B
fibertype(s) # F
```
A `LocalTensor{B,F}` consists of two components: `B`, which represents the `base` manifold, and `F`, which represents the `fiber` bundle over `B`.
"""
struct LocalTensor{B,F} <: LocalFiber{B,F}
    v::Pair{B,F}
    LocalTensor(v::Pair{B,F}) where {B,F} = new{B,F}(v)
    LocalTensor(b::B,f::F) where {B,F} = new{B,F}(b=>f)
    LocalTensor(b::B,f::LocalTensor{R,F} where R) where {B,F} = new{B,F}(b=>f.v.second)
    LocalTensor(b::LocalTensor{B,R} where R,f::F) where {B,F} = new{B,F}(base(b)=>f)
end

export Section, LocalTensor
const Section = LocalTensor
const ↦, domain, codomain = LocalTensor, base, fiber
↤(F,B) = B ↦ F

localfiber(x) = x
localfiber(x::LocalTensor) = fiber(x)

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
for type ∈ (:Coordinate,:LocalTensor)
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
        (V::Submanifold)(s::$type) = $type(base(s), V(fiber(s)))
        (::Type{T})(s::$type) where T<:Real = $type(base(s), T(fiber(s)))
        (::Type{Complex})(s::$type) = $type(base(s), Complex(fiber(s)))
        (::Type{Complex{T}})(s::$type) where T = $type(base(s), Complex{T}(fiber(s)))
        Grassmann.Phasor(s::$type) = $type(base(s), Phasor(fiber(s)))
        Grassmann.Couple(s::$type) = $type(base(s), Couple(fiber(s)))
        (::Type{T})(s::$type...) where T<:Chain = @inbounds $type(base(s[1]), Chain(fiber.(s)...))
    end
end

# FiberBundle

"""
    FiberBundle{T,N} <: Number

Defines a global `FiberBundle` type with `basetype` and `fibertype` over `N` dimensions.
"""
abstract type FiberBundle{T,N} <: AbstractArray{T,N} end


"""
    Coordinates{P,G,N} <: FiberBundle{Coordinate{P,G},N} <: Number

Defines a `FiberBundle` type with `pointtype` of `P` and `metrictype` of `G`.
```Julia
coordinates(s) # ::AbstractArray{Coordinate{P,G},N}
points(s) # ::AbstractArray{P,N}
metricextensor(s) # ::AbstractArray{G,N}
coordinatetype(s) # Coordinate{P,G}
pointtype(s) # P
metrictype(s) # G
```
Various methods work on any `Coordinates`, such as `base`, `fiber`, `coordinates`, `points`, `metricextensor`, `basetype`, `fibertype`, `coordinatetype`, `pointtype`, `metrictype`.
"""
const Coordinates{P,G,N} = FiberBundle{Coordinate{P,G},N}

"""
    coordinates(m::FiberBundle) -> FiberBundle{Coordinate{P,G}}

Return a `FiberBundle{Coordinate{P,G}}` if object `m` is defined as a `FiberBundle`.
"""
const coordinates = Coordinates

"""
    base(m::LocalFiber{B,F}) -> B

Return the `base` of a `FiberBundle` or `LocalSection`.
"""
base(t::FiberBundle) = t.dom

"""
    fiber(m::LocalFiber{B,F}) -> F

Return the `fiber` of a `FiberBundle` or `LocalSection`.
"""
fiber(t::FiberBundle) = t.cod
base(t::Array) = ProductSpace(Values(axes(t)))
fiber(t::Array) = t

"""
    basetype(m::LocalFiber{B,F}) -> DataType

Return the `basetype` of a `FiberBundle` or `LocalSection`.
"""
basetype(::Array{T}) where T = T

"""
    fibertype(m::LocalFiber{B,F}) -> DataType

Return the `fibertype` of a `FiberBundle` or `LocalSection`.
"""
fibertype(::Array{T}) where T = T

"""
    pointtype(m::Coordinate{P,G}) -> DataType

Return the `pointtype` of a `FiberBundle` or `LocalSection`.
"""
pointtype(m::FiberBundle) = basetype(coordinatetype(m))
pointtype(m::Type{<:FiberBundle}) = basetype(coordinatetype(m))

"""
    metrictype(m::Coordinate{P,G}) -> DataType

Return the `metrictype` of a `FiberBundle` or `LocalSection`.
"""
metrictype(m::FiberBundle) = fibertype(coordinatetype(m))
metrictype(m::Type{<:FiberBundle}) = fibertype(coordinatetype(m))

"""
    coordinatetype(m::Coordinate{P,G}) -> DataType

Return the `coordinatetype` of a `FiberBundle` or `LocalSection`.
"""
coordinatetype(m::Coordinates) = eltype(m)
coordinatetype(m::Type{<:Coordinates}) = eltype(m)

coordinates(m::Coordinates) = m

"""
    fullcoordinates(m::FiberBundle) -> FiberBundle{Coordinate{P,G}}

Return full `FiberBundle{Coordinate{P,G}}` instead of a possible subspace of it.
"""
fullcoordinates(m::Coordinates) = m

"""
    fullpoints(m::FiberBundle) -> AbstractArray{P}

Return full `AbstractArray{P}` instead of a possible subspace of it.
"""
fullpoints(m::FiberBundle) = base(fullcoordinates(m))

"""
    fullmetricextensor(m::FiberBundle) -> AbstractArray{G}

Return full `AbstractArray{G}` instead of a possible subspace of it.
"""
fullmetricextensor(m::FiberBundle) = fiber(fullcoordinates(m))
fullmetrictensor(m::FiberBundle) = submetric(fullmetricextensor(m))
metrictensor(m::FiberBundle) = submetric(metricextensor(m))
submetric(x::Global) = x
submetric(x::AbstractArray) = submetric.(x)
submetric(x::DiagonalOperator) = DiagonalOperator(getindex(x,1))
submetric(x::Outermorphism) = TensorOperator(getindex(x,1))

"""
    isinduced(m) -> Bool

Return `true` if the `metrictype` is an `InducedMetric` type.
"""
isinduced(m::FiberBundle) = isinduced(fullcoordinates(m))
isinduced(::DenseArray) = false
isinduced(::Global) = false
isinduced(::Global{N,<:InducedMetric} where N) = true

fullpoints(m::AbstractArray{<:Chain{V,1} where V}) = m
points(m::AbstractArray{<:Chain{V,1} where V}) = m
metricextensor(m::AbstractArray{<:Chain{V,1} where V,N}) where N = Global{N}(InducedMetric())

@pure Grassmann.Manifold(m::FiberBundle) = Manifold(pointtype(m))
@pure LinearAlgebra.rank(m::FiberBundle) = rank(pointtype(m))
@pure Grassmann.mdims(m::FiberBundle) = mdims(pointtype(m))

# PointCloud

export PointArray, PointVector, PointMatrix, PointCloud, Coordinates, FiberBundle

point_id = 0

"""
    PointArray{P,G,N} <: FiberBundle{Coordinate{P,G},N} <: Number

Defines a `FiberBundle` type with `pointtype` of `P` and `metrictype` of `G`.
```Julia
coordinates(s) # ::PointArray{P,G,N}
points(s) # ::AbstractArray{P,N}
metricextensor(s) # ::AbstractArray{G,N}
coordinatetype(s) # ::Coordinate{P,G}
pointtype(s) # P
metrictype(s) # G
```
Various methods work on any `PointArray`, such as `base`, `fiber`, `coordinates`, `points`, `metricextensor`, `basetype`, `fibertype`, `coordinatetype`, `pointtype`, `metrictype`.
"""
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

totalnodes(m::PointArray) = length(m)
nodes(m::PointArray) = length(m)

"""
    points(m) -> AbstractArray{P}

Return the `points` as an `AbstractArray{P}`.
"""
points(m::PointArray) = base(m)

metricextensor(m::PointArray) = fiber(m)
isinduced(t::PointArray) = isinduced(metricextensor(t))
basetype(::PointArray{B}) where B = B
basetype(::Type{<:PointArray{B}}) where B = B
fibertype(::PointArray{B,F} where B) where F = F
fibertype(::Type{<:PointArray{B,F} where B}) where F = F

PointCloud(id::Int,p::PA,g::GA) where {P,G,PA<:AbstractVector{P},GA<:AbstractVector{G}} = PointArray(id,p,g)
PointCloud(id::Int,dom) = PointArray(id,dom)
PointCloud(dom::AbstractVector) = PointCloud(dom,Global{1}(InducedMetric()))

const point_cache = (Vector{Chain{V,G,T,X}} where {V,G,T,X})[]
const point_metric_cache = (AbstractVector{T} where T)[]

PointCloud(m::Int) = PointCloud(m,point_cache[m],point_metric_cache[m])
function PointCloud(p::AbstractVector,g::AbstractVector)
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

Base.size(m::PointArray) = size(points(m))
Base.resize!(m::PointCloud,i::Int) = ((resize!(points(m),i),resize!(metricextensor(m),i)); m)
Base.broadcast(f,t::PointArray) = PointArray(f.(points(t)),f.(metricextensor(t)))
Base.broadcast(f,t::PointCloud) = PointCloud(f.(points(t)),f.(metricextensor(t)))
resize_lastdim!(m::Global,i) = m
resize_lastdim!(m::PointArray,i) = ((resize_lastdim!(m.dom,i),resize_lastdim!(m.cod,i)); m)

getindex(m::PointCloud,i::ImmersedTopology) = points(m)[i]

function (m::PointArray)(i::Vararg{Union{Int,Colon}})
    pa = points(m)(i...)
    ga = if isinduced(m)
        Global{ndims(pa)}(InducedMetric())
    else
        error("missing functionality")
    end
    return PointArray(0,pa,ga)
end

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
    !isinduced(m) && setindex!(metricextensor(m),metricextensor(s),i...)
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

#"`A = find_pa(As)` returns the first PointArray among the arguments."
find_pa(bc::Base.Broadcast.Broadcasted) = find_pa(bc.args)
find_pa(bc::Base.Broadcast.Extruded) = find_pa(bc.x)
find_pa(args::Tuple) = find_pa(find_pa(args[1]), Base.tail(args))
find_pa(x) = x
find_pa(::Tuple{}) = nothing
find_pa(a::PointArray, rest) = a
find_pa(::Any, rest) = find_pa(rest)

# FiberProduct

export FiberProduct

"""
    FiberProduct{P,N} <: FiberBundle{Coordinate{P,InducedMetric},N}

Represents a `FiberProduct` over a `base` and `fiber` with `InducedMetric`.
"""
struct FiberProduct{P,N,PA<:AbstractArray{F,N} where F,FA<:AbstractArray} <: FiberBundle{Coordinate{P,InducedMetric},N}
    p::PA
    f::FA
    FiberProduct{P}(p::PA,f::FA) where {P,N,PA<:AbstractArray{F,N} where F,FA<:AbstractArray} = new{P,N,PA,FA}(p,f)
end

fiber(m::FiberProduct{P,N} where P) where N = Gloabl{N}(InducedMetric())
metricextensor(m::FiberProduct) = fiber(m)
isinduced(t::FiberProduct) = isinduced(metricextensor(t))
pointspace(m::FiberProduct) = m.p
fiberspace(m::FiberProduct) = m.f
basetype(::FiberProduct{B}) where B = B
basetype(::Type{<:FiberProduct{B}}) where B = B
fibertype(::FiberProduct) = InducedMetric
fibertype(::Type{<:FiberProduct}) = InducedMetric

Base.size(m::FiberProduct) = (size(pointspace(m))...,size(fiberspace(m))...)
#Base.broadcast(f,t::FiberProduct{P}) where P = FiberProduct{P}(f.(pointspace(t)),f.(fiberspace(t)))

function Base.getindex(m::FiberProduct,i::Int,j::Vararg{Int})
    Coordinate(getindex(pointspace(m),i,j...) ⧺ getindex(fiberspace(m),j...), InducedMetric())
end
Base.setindex!(m::FiberProduct{P},s::P,i::Int,j::Vararg{Int}) where P = setindex!(pointspace(m),s,i,j...)
function Base.setindex!(m::FiberProduct,s::Coordinate,i::Int,j::Vararg{Int})
    setindex!(pointspace(m),point(s),i,j...)
    return s
end

# GlobalFiber

"""
    GlobalFiber{E,N} <: FiberBundle{E,N} <: Number

Defines a `FiberBundle` type with `basetype` and `fibertype`.
"""
abstract type GlobalFiber{E,N} <: FiberBundle{E,N} end
Base.@pure isfiberbundle(::GlobalFiber) = true
Base.@pure isfiberbundle(::Any) = false

globalfiber(x) = x
globalfiber(x::GlobalFiber) = fiber(x)

for fun ∈ (:sdims,:subimmersion,:fullimmersion,:fulltopology,:topology,:subtopology,:totalelements,:elements,:subelements,:totalnodes,:nodes,:vertices,:verticesinv,:isopen,:iscompact,:isfull,:iscover,:istotal,:immersiontype,:refnodes,:isdisconnected,:isdiscontinuous)
    @eval export $fun
    @eval $fun(m::GlobalFiber) = $fun(immersion(m))
end
sdims(m::Type{<:GlobalFiber}) = sdims(immersiontype(m))
#imagepoints(m::GlobalFiber) = iscover(m) ? points(m) : points(m)[vertices(m)]

unitdomain(t::GlobalFiber) = base(t)*inv(base(t)[end])
arcdomain(t::GlobalFiber) = unitdomain(t)*arclength(codomain(t))
graph(t::GlobalFiber) = graph.(t)

Base.size(m::GlobalFiber) = size(m.cod)
Base.resize!(m::GlobalFiber,i) = ((resize!(domain(m),i),resize!(codomain(m),i)); m)
resize_lastdim!(m::GlobalFiber,i) = ((resize_lastdim!(domain(m),i),resize_lastdim!(codomain(m),i)); m)

# FrameBundle

export FrameBundle, GridBundle, SimplexBundle, FaceBundle, ElementBundle
export IntervalRange, AlignedRegion, AlignedSpace

"""
    FrameBundle{C,N} <: GlobalFiber{C,N} <: FiberBundle{C,N}

Defines a `GlobalFiber` type with `coordinatetype` of `C` and `immersion`.
```Julia
coordinates(s) # ::AbstractArray{C,N}
points(s) # ::AbstractArray{P,N}
metricextensor(s) # ::AbstractArray{G,N}
coordinatetype(s) # C
pointtype(s) # P
metrictype(s) # G
immersion(s) # ::ImmersedTopology
```
Various methods work on any `FrameBundle`, such as `isbundle`, `base`, `fiber`, `coordinates`, `points`, `metricextensor`, `basetype`, `fibertype`, `coordinatetype`, `pointtype`, `metrictype`, `immersion`.
"""
abstract type FrameBundle{C,N} <: GlobalFiber{C,N} end

base(m::FrameBundle) = points(m)
fiber(m::FrameBundle) = metricextensor(m)
metricextensor(m::FrameBundle) = metricextensor(coordinates(m))
metrictensor(m::FrameBundle) = metrictensor(coordinates(m))
coordinatetype(m::FrameBundle{C}) where C = C
coordinatetype(m::Type{<:FrameBundle{C}}) where C = C
basetype(m::FrameBundle) = basetype(coordinates(m))
basetype(m::Type{<:FrameBundle}) = basetype(coordinatetype(m))
fibertype(m::FrameBundle) = fibertype(coordinates(m))
fibertype(m::Type{<:FrameBundle}) = fibertype(coordinatetype(m))
Base.size(m::FrameBundle) = size(points(m))

@pure isbundle(::FrameBundle) = true
@pure isbundle(t) = false
@pure Grassmann.grade(m::FrameBundle) = grade(pointtype(m))
@pure Grassmann.antigrade(m::FrameBundle) = antigrade(pointtype(m))

# GridBundle

"""
    GridBundle{N,C} <: FrameBundle{C,N} <: FiberBundle{C,N}

Defines a `FrameBundle` over grid points with `coordinatetype` of `C` and `immersion`.
```Julia
coordinates(s) # ::AbstractArray{C,N}
points(s) # ::AbstractArray{P,N}
metricextensor(s) # ::AbstractArray{G,N}
coordinatetype(s) # C
pointtype(s) # P
metrictype(s) # G
immersion(s) # ::QuotientTopology
```
Various methods work on any `FrameBundle`, such as `isbundle`, `base`, `fiber`, `coordinates`, `points`, `metricextensor`, `basetype`, `fibertype`, `coordinatetype`, `pointtype`, `metrictype`, `immersion`.
```Julia
IntervalRange{P, G, PA, GA} where {P<:Real, G, PA<:AbstractRange, GA} (alias for GridBundle{1, Coordinate{P, G}, <:PointArray{P, G, 1, PA, GA}} where {P<:Real, G, PA<:AbstractRange, GA})
AlignedRegion{N} where N (alias for GridBundle{N, Coordinate{P, G}, PointArray{P, G, N, PA, GA}} where {N, P<:Chain, G<:InducedMetric, PA<:(ProductSpace{V, <:Real, N, N, <:AbstractRange} where V), GA<:Global})
AlignedSpace{N} where N (alias for GridBundle{N, Coordinate{P, G}, PointArray{P, G, N, PA, GA}} where {N, P<:Chain, G<:InducedMetric, PA<:(ProductSpace{V, <:Real, N, N, <:AbstractRange} where V), GA})
```
"""
struct GridBundle{N,C<:Coordinate,PA<:FiberBundle{C,N},TA<:ImmersedTopology} <: FrameBundle{C,N}
    p::PA
    t::TA
    GridBundle(p::PA,t::TA=OpenTopology(size(p))) where {N,C<:Coordinate,PA<:FiberBundle{C,N},TA<:ImmersedTopology} = new{N,C,PA,TA}(p,t)
end

const IntervalRange{P<:Real,G,PA<:AbstractRange,GA} = GridBundle{1,Coordinate{P,G},<:PointVector{P,G,PA,GA}}
const AlignedRegion{N,P<:Chain,G<:InducedMetric,PA<:RealRegion{V,<:Real,N,<:AbstractRange} where V,GA<:Global} = GridBundle{N,Coordinate{P,G},PointArray{P,G,N,PA,GA}}
const AlignedSpace{N,P<:Chain,G<:InducedMetric,PA<:RealRegion{V,<:Real,N,<:AbstractRange} where V,GA} = GridBundle{N,Coordinate{P,G},PointArray{P,G,N,PA,GA}}

GridBundle(p::AbstractArray{P,N},g::AbstractArray=Global{N}(InducedMetric())) where {N,P} = GridBundle(PointArray(0,p,g))
GridBundle(dom::GridBundle,fun) = GridBundle(coordinates(dom), fun)
GridBundle(dom::GridBundle,fun::Array) = GridBundle(coordinates(dom), fun)
GridBundle(dom::GridBundle,fun::Function) = GridBundle(coordinates(dom), fun)
GridBundle(dom::AbstractArray,fun::Function) = GridBundle(dom, fun.(dom))

⊕(a::GridBundle{1},b::GridBundle{1}) = GridBundle(coordinates(a)⊕coordinates(b),immersion(a)×immersion(b))
⊕(a::GridBundle,b::AbstractVector{<:Real}) = GridBundle(coordinates(a)⊕b,immersion(a)×length(b))
cross(a::GridBundle,b::GridBundle) = a⊕b
cross(a::GridBundle,b::AbstractVector{<:Real}) = a⊕b

fullcoordinates(m::GridBundle) = m.p
coordinates(m::GridBundle) = m.p
immersion(m::GridBundle) = m.t
immersiontype(::Type{<:GridBundle{N,C,PA,TA} where {N,C,PA}}) where TA = TA
points(m::GridBundle) = base(coordinates(m))
metricextensor(m::GridBundle) = fiber(coordinates(m))

function resample(m::GridBundle,i::NTuple)
    rp,rq = resample(points(m),i),resample(immersion(m),i)
    pid = iszero(bundle(coordinates(m))) ? 0 : (global point_id+=1)
    if isinduced(m)
        GridBundle(PointArray(pid,rp),rq)
    else
        GridBundle(PointArray(pid,rp,m.(rp)),rq)
    end
end

resize_lastdim!(m::GridBundle,i) = (resize_lastdim!(coordinates(m),i); m)
resize(m::GridBundle) = GridBundle(coordinates(m),resize(immersion(m),size(coordinates(m))[end]))
Base.resize!(m::GridBundle,i) = (resize!(coordinates(m),i); m)
Base.broadcast(f,t::GridBundle) = GridBundle(f.(coordinates(t)))

(m::GridBundle)(i::ImmersedTopology) = GridBundle(coordinates(m),i)
(m::GridBundle)(i::Vararg{Union{Int,Colon}}) = GridBundle(coordinates(m)(i...),immersion(m)(i...))
@pure Base.eltype(::Type{<:GridBundle{N,C} where N}) where C = C
Base.getindex(m::GridBundle,i::Vararg{Int}) = getindex(coordinates(m),i...)
Base.getindex(m::GridBundle,i::Vararg{Union{Int,Colon}}) = GridBundle(getindex(base(m),i...),immersion(m)(i...))
Base.setindex!(m::GridBundle,s,i::Vararg{Int}) = setindex!(coordinates(m),s,i...)

export Grid
const Grid = GridBundle

#Grid(v::A,t::I=OpenTopology(size(v))) where {N,T,A<:AbstractArray{T,N},I} = GridBundle(0,PointArray(0,v),t)

Base.getindex(g::GridBundle{M,C,<:FiberBundle,<:OpenTopology} where C,j::Int,n::Val,i::Vararg{Int}) where M = getpoint(g,j,n,i...)
@generated function getpoint(g::GridBundle{M,C,<:FiberBundle} where C,j::Int,::Val{N},i::Vararg{Int}) where {M,N}
    :(Base.getindex(points(g),$([k≠N ? :(@inbounds i[$k]) : :(@inbounds i[$k]+j) for k ∈ list(1,M)]...)))
end
@generated function Base.getindex(g::GridBundle{M},j::Int,n::Val{N},i::Vararg{Int}) where {M,N}
    :(Base.getindex(points(g),Base.getindex(immersion(g),n,$([k≠N ? :(@inbounds i[$k]) : :(@inbounds i[$k]+j) for k ∈ list(1,M)]...))...))
end

Base.BroadcastStyle(::Type{<:GridBundle{N,C,PA,TA}}) where {N,C,PA,TA} = Broadcast.ArrayStyle{GridBundle{N,C,PA,TA}}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GridBundle{N,C,PA,TA}}}, ::Type{ElType}) where {N,C,PA,TA,ElType<:Coordinate}
    ax = axes(bc)
    # Use the data type to create the output
    GridBundle(similar(Array{pointtype(ElType),N}, ax), similar(Array{metrictype(ElType),N}, ax))
end
#=function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GridBundle{N,C,PA}}}, ::Type{ElType}) where {N,C,PA,ElType}
    t = find_gf(bc)
    # Use the data type to create the output
    GridBundle(similar(Array{ElType,N}, axes(bc)), metricextensor(t))
end=#

#"`A = find_gf(As)` returns the first GridBundle among the arguments."
find_gf(bc::Base.Broadcast.Broadcasted) = find_gf(bc.args)
find_gf(bc::Base.Broadcast.Extruded) = find_gf(bc.x)
find_gf(args::Tuple) = find_gf(find_gf(args[1]), Base.tail(args))
find_gf(x) = x
find_gf(::Tuple{}) = nothing
find_gf(a::FrameBundle, rest) = a
find_gf(::Any, rest) = find_gf(rest)

# ElementBundle

abstract type ElementBundle{N,C<:Coordinate,PA<:FiberBundle{C,1},TA} <: FrameBundle{C,1} end

const DiscontinuousBundle{N,C,PA,TA<:DiscontinuousTopology} = ElementBundle{N,C,PA,TA}
export DiscontinuousBundle

fullcoordinates(m::ElementBundle) = m.p
immersion(m::ElementBundle) = m.t
immersiontype(::Type{<:ElementBundle{N,C,PA,TA} where {N,C,PA}}) where TA = TA
function affinehull(m::ElementBundle)
    if isdisconnected(m)
        points(m)[immersion(m)]
    else
        fullpoints(m)[SimplexTopology(immersion(m))]
    end
end
function affinehull(m::ElementBundle,i::Int)
    if isdisconnected(m)
        points(m)[immersion(m)[i]]
    else
        fullpoints(m)[SimplexTopology(immersion(m))[i]]
    end
end

# SimplexBundle

"""
    SimplexBundle{N,C} <: FrameBundle{C,1} <: FiberBundle{C,1}

Defines a `FrameBundle` over simplex vertices with `coordinatetype` of `C` and `immersion`.
```Julia
coordinates(s) # ::AbstractArray{C,N}
points(s) # ::AbstractArray{P,N}
metricextensor(s) # ::AbstractArray{G,N}
coordinatetype(s) # C
pointtype(s) # P
metrictype(s) # G
immersion(s) # ::ImmersedTopology
```
Various methods work on any `FrameBundle`, such as `isbundle`, `base`, `fiber`, `coordinates`, `points`, `metricextensor`, `basetype`, `fibertype`, `coordinatetype`, `pointtype`, `metrictype`, `immersion`.
"""
struct SimplexBundle{N,C<:Coordinate,PA<:FiberBundle{C,1},TA<:ImmersedTopology} <: ElementBundle{N,C,PA,TA}
    p::PA
    t::TA
    SimplexBundle(p::PA,t::TA) where {C,PA<:FiberBundle{C,1},TA} = new{mdims(pointtype(p))-1,C,PA,TA}(p,t)
end

SimplexBundle(m::SimplexBundle) = m
SimplexBundle(id::Int,p,t,g) = SimplexBundle(PointCloud(id,p,g),t)
SimplexBundle(id::Int,p,t) = SimplexBundle(PointCloud(id,p),t)
SimplexBundle(p::P,t,g::G) where {P<:AbstractVector,G<:AbstractVector} = SimplexBundle(PointCloud(p,g),t)
#SimplexBundle(p::AbstractVector,t) = SimplexBundle(PointCloud(p),t)

GridBundle{1}(m::SimplexBundle) = GridBundle(getindex.(fullpoints(m),2))

(p::PointCloud)(t::ImmersedTopology) = SimplexBundle(p,t)
function coordinates(m::SimplexBundle)
    if isdiscontinuous(m) ? isdisconnected(m) : iscover(m)
        fullcoordinates(m)
    else
        PointCloud(0,points(m),metricextensor(m))
    end
end
function points(m::SimplexBundle)
    if isdiscontinuous(m) ? isdisconnected(m) : iscover(m)
        base(fullcoordinates(m))
    else
        view(base(fullcoordinates(m)),vertices(m))
    end
end
function metricextensor(m::SimplexBundle)
    if isdisconnected(m) ? isdisconnected(m) : (iscover(m) || isinduced(m))
        fiber(fullcoordinates(m))
    else
        view(fiber(fullcoordinates(m)),vertices(m))
    end
end
#bundle(m::SimplexBundle) = bundle(coordinates(m))
#deletebundle!(m::SimplexBundle) = deletepointcloud!(bundle(m))
Base.size(m::SimplexBundle) = size(vertices(m))
refine(m::SimplexBundle) = SimplexBundle(fullcoordinates(m),refine(immersion(m)))
function continuous(m::SimplexBundle)
    isdiscontinuous(m) ? isdisconnected(m) ? error("disconnected") : SimplexBundle(fullcoordinates(m),continuous(immersion(m))) : m
end
function discontinuous(m::SimplexBundle)
    isdiscontinuous(m) ? m : SimplexBundle(fullcoordinates(m),discontinuous(immersion(m)))
end
function disconnect(m::SimplexBundle)
    if isdiscontinuous(m)
        isdisconnected(m) ? m : SimplexBundle(coordinates(m),disconnect(immersion(m)))
    else
        disconnect(discontinuous(m))
    end
end

#Base.broadcast(f,t::SimplexBundle) = SimplexBundle(f.(coordinates(t)),immersion(t))

#Base.resize!(m::SimplexBundle,n::Int) = resize!(value(m),n)

(m::SimplexBundle)(i::ImmersedTopology) = SimplexBundle(fullcoordinates(m),i)
Base.getindex(m::SimplexBundle,i::Chain{V,1}) where V = Chain{Manifold(V),1}(points(m)[value(i)])
Base.getindex(m::SimplexBundle,i::Values{N,Int}) where N = fullpoints(m)[i]
getindex(m::AbstractVector,i::ImmersedTopology) = getindex.(Ref(m),i)
getindex(m::AbstractVector,i::SimplexBundle) = m[immersion(i)]
getindex(m::SimplexBundle,i::ImmersedTopology) = fullpoints(m)[i]
getindex(m::SimplexBundle,i::SimplexBundle) = fullpoints(m)[topology(i)]

@pure Base.eltype(::Type{<:SimplexBundle{N,C} where N}) where C = C
function Base.getindex(m::SimplexBundle,i::Int)
    ind = getimage(immersion(m),i)
    Coordinate(fullpoints(m)[ind],fullmetricextensor(m)[ind])
end
Base.setindex!(m::SimplexBundle{N,<:Coordinate{P}} where N,s::P,i::Int) where P = setindex!(fullpoints(m),s,getimage(immersion(m),i))
Base.setindex!(m::SimplexBundle{N,<:Coordinate{P,G} where P} where N,s::G,i::Int) where G = setindex!(fullmetricextensor(m),s,getimage(immersion(m),i))
function Base.setindex!(m::SimplexBundle,s::Coordinate,i::Int)
    ind = getimage(immersion(m),i)
    setindex!(fullpoints(m),point(s),ind)
    setindex!(fullmetricextensor(m),metricextensor(s),ind)
    return s
end

Base.BroadcastStyle(::Type{<:SimplexBundle{N,C,PA,TA}}) where {N,C,PA,TA} = Broadcast.ArrayStyle{SimplexBundle{N,C,PA,TA}}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{SimplexBundle{N,C,PA,TA}}}, ::Type{ElType}) where {N,C,PA,TA,ElType<:Coordinate}
    ax = axes(bc)
    # Use the data type to create the output
    SimplexBundle(similar(Array{pointtype(ElType),N}, ax), similar(Array{metrictype(ElType),N}, ax))
end

function Base.findall(f::Function,pt::SimplexBundle)
    vt = vertices(pt)
    vt[findall(f,fullpoints(pt)[vt])]
end

function (m::SimplexBundle)(fixed::AbstractVector{Int})
    fullcoordinates(m)(subtopology(immersion(m),fixed))
end

# FaceBundle

"""
    FaceBundle{N,C} <: FrameBundle{C,1} <: FiberBundle{C,1}

Defines a `FrameBundle` over element faces with `coordinatetype` of `C` and `immersion`.
```Julia
coordinates(s) # ::AbstractArray{C,N}
points(s) # ::AbstractArray{P,N}
metricextensor(s) # ::AbstractArray{G,N}
coordinatetype(s) # C
pointtype(s) # P
metrictype(s) # G
immersion(s) # ::ImmersedTopology
```
Various methods work on any `FrameBundle`, such as `isbundle`,`base`, `fiber`, `coordinates`, `points`, `metricextensor`, `basetype`, `fibertype`, `coordinatetype`, `pointtype`, `metrictype`, `immersion`.
"""
struct FaceBundle{N,C<:Coordinate,PA<:FiberBundle{C,1},TA<:ImmersedTopology} <: ElementBundle{N,C,PA,TA}
    p::PA
    t::TA
    FaceBundle(p::PA,t::TA) where {C,PA<:FiberBundle{C,1},TA} = new{mdims(pointtype(p))-1,C,PA,TA}(p,t)
end

#FaceBundle(id::Int,p,t,g) = FaceBundle(PointCloud(id,p,g),t)
#FaceBundle(id::Int,p,t) = FaceBundle(PointCloud(id,p),t)
#FaceBundle(p::AbstractVector,t) = FaceBundle(PointCloud(p),t)

SimplexBundle(m::FaceBundle) = SimplexBundle(fullcoordinates(m),immersion(m))
FaceBundle(m::SimplexBundle) = FaceBundle(fullcoordinates(m),immersion(m))
FaceBundle(m::FaceBundle) = m

coordinates(m::FaceBundle) = PointCloud(0,points(m),metricextensor(m))
#vertices(m::FaceBundle) = OneTo(length(m))
function points(m::FaceBundle)
    if isdisconnected(m)
        fullpoints(m)
    else
        means(SimplexTopology(immersion(m)),fullpoints(m))
    end
end
function metricextensor(m::FaceBundle)
    if isdisconnected(m) || isinduced(m)
        fullmetricextensor(m)
    else
        means(SimplexTopology(immersion(m)),fullmetricextensor(m))
    end
end

refine(m::FaceBundle) = FaceBundle(fullcoordinates(m),refine(immersion(m)))
function continuous(m::FaceBundle)
    isdiscontinuous(m) ? isdisconnected(m) ? error("disconnected") : FaceBundle(fullcoordinates(m),continuous(immersion(m))) : m
end
function discontinuous(m::FaceBundle)
    isdiscontinuous(m) ? m : FaceBundle(fullcoordinates(m),discontinuous(immersion(m)))
end
function disconnect(m::FaceBundle)
    if isdiscontinuous(m)
        isdisconnected(m) ? m : FaceBundle(coordinates(SimplexBundle(m)),disconnect(immersion(m)))
    else
        disconnect(discontinuous(m))
    end
end

#Base.broadcast(f,t::FaceBundle) = FaceBundle(f.(t.p),immersion(t))
Base.size(m::FaceBundle) = size(immersion(m))
@pure Base.eltype(::Type{<:FaceBundle{N,C} where N}) where C = C
Base.getindex(m::FaceBundle,i::AbstractVector{Int}) = FaceBundle(fullcoordinates(m),immersion(m)[i])
function Base.getindex(m::FaceBundle,i::Int)
    ind = isdisconnected(m) ? immersion(m)[i] : SimplexTopology(immersion(m))[i]
    Coordinate(mean(fullpoints(m)[ind]),
        isinduced(m) ? fullmetricextensor(m)[i] : mean(fullmetricextensor(m)[ind]))
end

Base.BroadcastStyle(::Type{<:FaceBundle{N,C,PA,TA}}) where {N,C,PA,TA} = Broadcast.ArrayStyle{FaceBundle{N,C,PA,TA}}()

# MultilinearBundle

export MultilinearBundle, VolumeBundle

struct MultilinearBundle{N,C<:Coordinate,PA<:FiberBundle{C,N},TA<:MultilinearTopology} <: FrameBundle{C,1}
    p::PA
    t::TA
end

MultilinearBundle(m::GridBundle) = MultilinearBundle(fullcoordinates(m),MultilinearTopology(immersion(m)))
GridBundle(m::MultilinearBundle) = GridBundle(fullcoordinates(m),QuotientTopology(immersion(m)))

fullcoordinates(m::MultilinearBundle) = m.p
coordinates(m::MultilinearBundle) = PointCloud(0,points(m),metricextensor(m))
immersion(m::MultilinearBundle) = m.t
immersiontype(::Type{<:MultilinearBundle{N,C,PA,TA} where {N,C,PA}}) where TA = TA
points(m::MultilinearBundle) = fullpoints(m)[verticesinv(m)]
metricextensor(m::MultilinearBundle) = isinduced(m) ? vec(fullmetricextensor(m)) : fullmetricextensor(m)[verticesinv(m)]

Base.size(m::MultilinearBundle) = size(verticesinv(m))

@pure Base.eltype(::Type{<:MultilinearBundle{N,C} where N}) where C = C
function Base.getindex(m::MultilinearBundle,i::Int)
    ind = verticesinv(immersion(m))[i]
    Coordinate(fullpoints(m)[ind],fullmetricextensor(m)[ind])
end

# VolumeBundle

struct VolumeBundle{N,C<:Coordinate,PA<:FiberBundle{C,N},TA<:MultilinearTopology} <: FrameBundle{C,1}
    p::PA
    t::TA
end

VolumeBundle(m::MultilinearBundle) = VolumeBundle(fullcoordinates(m),immersion(m))
MultilinearBundle(m::VolumeBundle) = MultilinearBundle(fullcoordinates(m),immersion(m))
VolumeBundle(m::GridBundle) = VolumeBundle(fullcoordinates(m),MultilinearTopology(immersion(m)))
GridBundle(m::VolumeBundle) = GridBundle(fullcoordinates(m),QuotientTopology(immersion(m)))

fullcoordinates(m::VolumeBundle) = m.p
coordinates(m::VolumeBundle) = PointCloud(0,points(m),metricextensor(m))
immersion(m::VolumeBundle) = m.t
immersiontype(::Type{<:VolumeBundle{N,C,PA,TA} where {N,C,PA}}) where TA = TA
function points(m::VolumeBundle)
    i = immersion(m)
    p = vec(fullpoints(m))
    vcat(mean.(p[i.q]),mean.(p[i.t]))[last.(i.s)]
end
function metricextensor(m::VolumeBundle)
    if isinduced(m)
        vec(fullmetricextensor(m))
    else
        i = immersion(m)
        p = vec(fullmetricextensor(m))
        vcat(mean.(p[i.q]),mean.(p[i.t]))[last.(i.s)]
    end
end

Base.size(m::VolumeBundle) = (prod(size(QuotientTopology(immersion(m))).-1),)

@pure Base.eltype(::Type{<:VolumeBundle{N,C} where N}) where C = C
function Base.getindex(m::VolumeBundle{N,C,PA,<:BilinearTopology} where {N,C,PA},i::Int)
    s = elementsplit(immersion(m))[i]
    ind = (first(s) ≠ 3 ? immersion(m).q : immersion(m).t)[last(s)]
    met = isinduced(m) ? fullmetricextensor(m).v : mean(fullmetricextensor(m)[ind])
    Coordinate(mean(fullpoints(m)[ind]),met)
end

# FiberProductBundle

"""
    FiberProductBundle{P,N} <: FrameBundle{Coordinate{P,InducedMetric},N}

Represents a `FiberProductBundle` over a `base` and `fiber` with `InducedMetric`.
"""
struct FiberProductBundle{P,N,SA<:AbstractArray,PA<:AbstractArray} <: FrameBundle{Coordinate{P,InducedMetric},N}
    s::SA
    g::PA
    FiberProductBundle{P}(s::SA,g::PA) where {P,M,N,SA<:AbstractArray{S,M} where S,PA<:AbstractArray{F,N} where F} = new{P,M+N,SA,PA}(s,g)
end

varmanifold(N::Int) = Submanifold(N+1)(list(1,N)...)

function ⊕(a::SimplexBundle,b::AbstractVector{B}) where B<:Real
    V = Manifold(basetype(eltype(a)))
    N = mdims(V)+1
    W = Submanifold(N)
    P = Chain{W,1,promote_type(valuetype(basetype(eltype(a))),B),N}
    #p = PointCloud(Chain{varmanifold(N-1)}.(value.(points(a))))
    FiberProductBundle{P}(a,ProductSpace{W(N)}(Values((b,))))
end

basetype(m::FiberProductBundle{P}) where P = P
basetype(::Type{<:FiberProductBundle{P}}) where P = P
fibertype(m::FiberProductBundle{P}) where P = InducedMetric
fibertype(::Type{<:FiberProductBundle{P}}) where P = InducedMetric
Base.size(m::FiberProductBundle) = (length(m.s),size(m.g)...)
metricextensor(m::FiberProductBundle) = Global{mdims(basetype(m))}(InducedMetric())

(m::FaceBundle)(i::ImmersedTopology) = FaceBundle(fullcoordinates(m),i)

@pure Base.eltype(::Type{<:FiberProductBundle{P}}) where P = Coordinate{P,InducedMetric}
Base.getindex(m::FiberProductBundle,i::Int,j::Vararg{Int}) = Coordinate(getindex(points(m.s),i) ⧺ getindex(m.g,j...), InducedMetric())
#=Base.setindex!(m::FiberProductBundle{P},s::P,i::Int,j::Vararg{Int}) where P = setindex!(points(m),s,i,j...)
Base.setindex!(m::FiberProductBundle{P,G} where P,s::G,i::Int,j::Vararg{Int}) where G = setindex!(metricextensor(m),s,i,j...)
function Base.setindex!(m::FiberProductBundle,s::Coordinate,i::Int,j::Vararg{Int})
    setindex!(points(m),point(s),i...)
    setindex!(metricextensor(m),metricextensor(s),i...)
    return s
end=#

GridBundle(f::FiberProductBundle{P,2} where P) = OpenTopology(getindex.(points(f.s),2)⊕points(f.g))

export TimeParameter
function TimeParameter(m,time::AbstractVector)
    TensorField(m⊕time,[time[l] for j ∈ 1:length(m), l ∈ 1:length(time)])
end
function TimeParameter(m,fixed::AbstractVector,time::AbstractVector)
    TimeParameter(m(fixed),time)
end
TimeParameter(m,fixed::Function,time::AbstractVector) = TimeParameter(m,findall(fixed,m),time)

# HomotopyBundle

"""
    HomotopyBundle{P,N} <: FrameBundle{Coordinate{P,InducedMetric},N}

Represents a `HomotopyBundle` over a `base` and `fiber` with `InducedMetric`.
"""
struct HomotopyBundle{P,N,PA<:AbstractArray{F,N} where F,FA<:AbstractArray,TA<:ImmersedTopology} <: FrameBundle{Coordinate{P,InducedMetric},N}
    p::FiberProduct{P,N,PA,FA}
    t::TA
    HomotopyBundle(p::FiberProduct{P,N,PA,FA},t::T) where {P,N,PA<:AbstractArray{F,N} where F,FA,T} = new{P,N,PA,FA,T}(p,t)
end

(p::FiberProduct)(t::ImmersedTopology) = HomotopyBundle(p,t)
FiberProduct(m::HomotopyBundle) = m.p
coordinates(m::HomotopyBundle) = m.p
pointspace(m::HomotopyBundle) = pointspace(coordinates(m))
fiberspace(m::HomotopyBundle) = fiberspace(coordinates(m))
Base.size(m::HomotopyBundle) = size(FiberProduct(m))

Base.broadcast(f,t::HomotopyBundle) = HomotopyBundle(f.(coordinates(t)),immersion(t))

(m::HomotopyBundle)(i::ImmersedTopology) = HomotopyBundle(coordinates(m),i)
#Base.getindex(m::HomotopyBundle,i::Chain{V,1}) where V = Chain{Manifold(V),1}(pointspace(m)[value(i)])
#Base.getindex(m::HomotopyBundle,i::Values{N,Int}) where N = pointspace(m)[value(i)]
getindex(m::HomotopyBundle,i::ImmersedTopology) = coordinates(m)[i]
getindex(m::HomotopyBundle,i::HomotopyBundle) = coordinates(m)[immersion(i)]

@pure Base.eltype(::Type{<:HomotopyBundle{P}}) where P = Coordinate{P,InducedMetric}
function Base.getindex(m::HomotopyBundle,i::Int,j::Vararg{Int})
    ind = getimage(m,i)
    Coordinate(getindex(pointspace(m),ind,j...) ⧺ getindex(fiberspace(m),j...), InducedMetric())
end
#=Base.setindex!(m::HomotopyBundle{P},s::P,i::Int) where P = setindex!(pointspace(m),s,getimage(m,i))
function Base.setindex!(m::HomotopyBundle,s::Coordinate,i::Int)
    ind = getimage(m,i)
    setindex!(pointspace(m),point(s),ind)
    return s
end=#

