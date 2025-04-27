module Cartan

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
# _________                __                  __________
# \_   ___ \_____ ________/  |______    ____   \\       /
# /    \  \/\__  \\_  __ \   __\__  \  /    \   \\     /
# \     \____/ __ \|  | \/|  |  / __ \|   |  \   \\   /
#  \______  (____  /__|   |__| (____  /___|  /    \\ /
#         \/     \/                 \/     \/      \/
#  _____                           ___ _      _     _
# /__   \___ _ __  ___  ___  _ __ / __(_) ___| | __| |___
#   / /\/ _ \ '_ \/ __|/ _ \| '__/ _\ | |/ _ \ |/ _` / __|
#  / / |  __/ | | \__ \ (_) | | / /   | |  __/ | (_| \__ \
#  \/   \___|_| |_|___/\___/|_| \/    |_|\___|_|\__,_|___/

using SparseArrays, LinearAlgebra, ElasticArrays, Base.Threads, Grassmann, Requires
import Grassmann: value, vector, valuetype, tangent, istangent, Derivation, radius, ⊕
import Grassmann: realvalue, imagvalue, points, metrictensor, metricextensor
import Grassmann: Values, Variables, FixedVector, list, volume, compound
import Grassmann: Scalar, GradedVector, Bivector, Trivector
import Base: @pure, OneTo, getindex
import LinearAlgebra: cross
import ElasticArrays: resize_lastdim!

export Values, Derivation
export initmesh, pdegrad, det, graphbundle

macro elastic(itr)
    :(elastic($(itr.args[1]),$(itr.args[2])))
end
function elastic(T,itr)
    out = ElasticArray{T}(undef,size(itr))
    for e ∈ enumerate(itr)
        @inbounds out[e[1]] = e[2]
    end
    return out
end

include("topology.jl")
include("quotient.jl")
include("fiber.jl")

export ElementMap, SimplexMap, FaceMap
export IntervalMap, RectangleMap, HyperrectangleMap, PlaneCurve, SpaceCurve
export TensorField, ScalarField, VectorField, BivectorField, TrivectorField
export ElementFunction, SurfaceGrid, VolumeGrid, ScalarGrid, Variation
export RealFunction, ComplexMap, SpinorField, CliffordField
export ScalarMap, GradedField, QuaternionField, PhasorField
export GlobalFrame, DiagonalField, EndomorphismField, OutermorphismField
export ParametricMap, RectangleMap, HyperrectangleMap, AbstractCurve
export metrictensorfield, metricextensorfield
export alteration, variation, modification

# TensorField

"""
    TensorField{B,F,N} <: GlobalFiber{LocalTensor{B,F},N}

Defines a `GlobalFiber` type with `eltype` of `LocalTensor{B,F}` and `immersion`.
```Julia
coordinates(s) # ::AbstractArray{B,N}
fiber(s) # ::AbstractArray{F,N}
points(s) # ::AbstractArray{P,N}
metricextensor(s) # ::AbstractArray{G,N}
coordinatetype(s) # B
fibertype(s) # F
pointtype(s) # P
metrictype(s) # G
immersion(s) # ::ImmersedTopology
```
Various methods work on any `TensorField`, such as `base`, `fiber`, `coordinates`, `points`, `metricextensor`, `basetype`, `fibertype`, `coordinatetype`, `pointtype`, `metrictype`, `immersion`.
Due to the versatility of the `TensorField` type instances, it's possible to disambiguate them into these type alias specifications with associated methods:
`ElementMap`, `SimplexMap`, `FaceMap`, `IntervalMap`, `RectangleMap`, `HyperrectangleMap`, `ParametricMap`, `Variation`, `RealFunction`, `PlaneCurve`, `SpaceCurve`, `AbstractCurve`,
`SurfaceGrid`, `VolumeGrid`, `ScalarGrid`, `DiagonalField`, `EndomorphismField`, `OutermorphismField`, `CliffordField`, `QuaternionField`, `ComplexMap`, `PhasorField`, `SpinorField`, `GradedField{G} where G`, `ScalarField`, `VectorField`, `BivectorField`, `TrivectorField`.

In the `Cartan` package, a technique is employed where a `TensorField` is constructed from an interval, product manifold, or topology, to generate an algebra of sections which can be used to compose parametric maps on manifolds.
Constructing a `TensorField` can be accomplished in various ways,
there are explicit techniques to construct a `TensorField` as well as implicit methods.
Additional packages such as `Adapode` build on the `TensorField` concept by generating them from differential equations.
Many of these methods can automatically generalize to higher dimensional manifolds and are compatible with discrete differential geometry.
"""
struct TensorField{B,F,N,M<:FrameBundle{B,N},A<:AbstractArray{F,N}} <: GlobalFiber{LocalTensor{B,F},N}
    dom::M
    cod::A
    function TensorField(dom::M,cod::A) where {B,F,N,M<:FrameBundle{B,N},A<:AbstractArray{F,N}}
        new{B,F,N,M,A}(dom,cod)
    end
end

#TensorField(dom::FrameBundle,cod::Array) = TensorField(dom,ElasticArray(cod))
function TensorField(dom::AbstractArray{P,N},cod::AbstractArray{F,N},met::AbstractArray=Global{N}(InducedMetric())) where {N,P,F}
    TensorField(GridBundle(PointArray(0,dom,met)),cod)
end
TensorField(dom::PointArray,cod::AbstractArray) = TensorField(GridBundle(dom),cod)
TensorField(dom,cod::AbstractArray,met::GlobalFiber) = TensorField(dom,cod,fiber(met))
TensorField(dom::FrameBundle,cod::FrameBundle) = TensorField(dom,points(cod))
TensorField(a::TensorField,b::TensorField) = TensorField(fiber(a),fiber(b))

#const ParametricMesh{B,F,P<:AbstractVector{<:Chain},A} = TensorField{B,F,1,P,A}
const ScalarMap{B,F<:AbstractReal,P<:SimplexBundle,A} = TensorField{B,F,1,P,A}
const ElementMap{B,F,P<:ElementBundle,A} = TensorField{B,F,1,P,A}
const SimplexMap{B,F,P<:SimplexBundle,A} = TensorField{B,F,1,P,A}
const FaceMap{B,F,P<:FaceBundle,A} = TensorField{B,F,1,P,A}
#const ElementFunction{B,F<:AbstractReal,P<:AbstractVector,A} = TensorField{B,F,1,P,A}
const IntervalMap{B,F,P<:Interval,A} = TensorField{B,F,1,P,A}
const RectangleMap{B,F,P<:RealSpace{2},A} = TensorField{B,F,2,P,A}
const HyperrectangleMap{B,F,P<:RealSpace{3},A} = TensorField{B,F,3,P,A}
const ParametricMap{B,F,N,P<:RealSpace,A} = TensorField{B,F,N,P,A}
const Variation{B,F<:TensorField,N,P,A} = TensorField{B,F,N,P,A}
const RealFunction{B,F<:AbstractReal,P<:Interval,A} = TensorField{B,F,1,P,A}
const PlaneCurve{B,F<:Chain{V,G,Q,2} where {V,G,Q},P<:Interval,A} = TensorField{B,F,1,P,A}
const SpaceCurve{B,F<:Chain{V,G,Q,3} where {V,G,Q},P<:Interval,A} = TensorField{B,F,1,P,A}
const AbstractCurve{B,F<:Chain,P<:Interval,A} = TensorField{B,F,1,P,A}
const SurfaceGrid{B,F<:AbstractReal,P<:RealSpace{2},A} = TensorField{B,F,2,P,A}
const VolumeGrid{B,F<:AbstractReal,P<:RealSpace{3},A} = TensorField{B,F,3,P,A}
const ScalarGrid{B,F<:AbstractReal,N,P<:RealSpace{N},A} = TensorField{B,F,N,P,A}
#const GlobalFrame{B<:LocalFiber{P,<:TensorNested} where P,N} = GlobalSection{B,N}
const DiagonalField{B,F<:DiagonalOperator,N,P,A} = TensorField{B,F,N,P,A}
const EndomorphismField{B,F<:Endomorphism,N,P,A} = TensorField{B,F,N,P,A}
const OutermorphismField{B,F<:Outermorphism,N,P,A} = TensorField{B,F,N,P,A}
const CliffordField{B,F<:Multivector,N,P,A} = TensorField{B,F,N,P,A}
const QuaternionField{B,F<:Quaternion,N,P,A} = TensorField{B,F,N,P,A}
const ComplexMap{B,F<:AbstractComplex,N,P,A} = TensorField{B,F,N,P,A}
const PhasorField{B,F<:Phasor,N,P,A} = TensorField{B,F,N,P,A}
const SpinorField{B,F<:AbstractSpinor,N,P,A} = TensorField{B,F,N,P,A}
const GradedField{G,B,F<:Chain{V,G} where V,N,P,A} = TensorField{B,F,N,P,A}
const ScalarField{B,F<:AbstractReal,N,P,A} = TensorField{B,F,N,P,A}
const VectorField = GradedField{1}
const BivectorField = GradedField{2}
const TrivectorField = GradedField{3}

TensorField(dom,fun::BitArray) = TensorField(dom, Float64.(fun))
TensorField(dom,fun::TensorField) = TensorField(dom, fiber(fun))
TensorField(dom::TensorField,fun) = TensorField(base(dom), fun)
TensorField(dom::TensorField,fun::DenseArray) = TensorField(base(dom), fun)
TensorField(dom::TensorField,fun::Function) = TensorField(base(dom), fun)
#TensorField(dom::AbstractArray,fun::AbstractRange) = TensorField(dom, collect(fun))
#TensorField(dom::AbstractArray,fun::RealRegion) = TensorField(dom, collect(fun))
TensorField(dom::AbstractArray,fun::Function) = TensorField(dom, fun.(dom))
TensorField(dom::FrameBundle,fun::Function) = fun.(dom)
TensorField(dom::AbstractArray,fun::Number) = TensorField(dom, fill(fun,size(dom)...))
TensorField(dom::AbstractArray) = TensorField(dom, dom)
TensorField(f::F,r::AbstractVector{<:Real}=-2π:0.0001:2π) where F<:Function = TensorField(r,vector.(f.(r)))
base(t::TensorField) = t.dom
fiber(t::TensorField) = t.cod
coordinatetype(t::TensorField) = basetype(t)
coordinatetype(t::Type{<:TensorField}) = basetype(t)
basetype(::TensorField{B}) where B = B
basetype(::Type{<:TensorField{B}}) where B = B
fibertype(::TensorField{B,F} where B) where F = F
fibertype(::Type{<:TensorField{B,F} where B}) where F = F
Base.broadcast(f,t::TensorField) = TensorField(domain(t), f.(codomain(t)))
#TensorField(dom::SimplexBundle{1}) = TensorField(dom,getindex.(points(dom),2))

for fun ∈ (:points,:metricextensor,:coordinates,:immersion,:vertices,:fullcoordinates)
    @eval $fun(t::TensorField) = $fun(base(t))
end

←(F,B) = B → F
const → = TensorField
metricextensorfield(t::TensorField) = metricextensorfield(base(t))
metrictensorfield(t::TensorField) = metrictensorfield(base(t))
metricextensorfield(t::GridBundle) = TensorField(GridBundle(PointArray(0,points(t)),immersion(t)),metricextensor(t))
metrictensorfield(t::GridBundle) = TensorField(GridBundle(PointArray(0,points(t)),immersion(t)),metrictensor(t))
metricextensorfield(t::SimplexBundle) = TensorField(SimplexBundle(PointCloud(0,points(t)),immersion(t)),metricextensor(t))
metrictensorfield(t::SimplexBundle) = TensorField(SimplexBundle(PointCloud(0,points(t)),immersion(t)),metrictensor(t))
Grassmann.grade(::GradedField{G}) where G = G
Grassmann.antigrade(t::GradedField) = antigrade(fibertype(t))

resize(t::TensorField) = TensorField(resize(domain(t)),codomain(t))

function resample(t::TensorField,i::NTuple)
    rg = resample(base(t),i)
    TensorField(rg,t.(points(rg)))
end

@pure Base.eltype(::Type{<:TensorField{B,F}}) where {B,F} = LocalTensor{B,F}
Base.getindex(m::TensorField,i::Vararg{Int}) = LocalTensor(getindex(domain(m),i...), getindex(codomain(m),i...))
Base.getindex(m::TensorField,i::Vararg{Union{Int,Colon}}) = TensorField(domain(m)(i...), getindex(codomain(m),i...))
#Base.setindex!(m::TensorField{B,F,1,<:Interval},s::LocalTensor,i::Vararg{Int}) where {B,F} = setindex!(codomain(m),fiber(s),i...)
Base.setindex!(m::TensorField{B,Fm} where Fm,s::F,i::Vararg{Int}) where {B,F} = setindex!(codomain(m),s,i...)
Base.setindex!(m::TensorField{B,Fm} where Fm,s::TensorField,::Colon,i::Int) where B = setindex!(codomain(m),codomain(s),:,i)
Base.setindex!(m::TensorField{B,Fm} where Fm,s::TensorField,::Colon,::Colon,i::Int) where B = setindex!(codomain(m),codomain(s),:,:,i)
Base.setindex!(m::TensorField{B,Fm} where Fm,s::TensorField,::Colon,::Colon,::Colon,i::Int) where B = setindex!(codomain(m),codomain(s),:,:,:,i)
Base.setindex!(m::TensorField{B,Fm} where Fm,s::TensorField,::Colon,::Colon,::Colon,::Colon,i::Int) where B = setindex!(codomain(m),codomain(s),:,:,:,:,i)
function Base.setindex!(m::TensorField{B,F,N,<:IntervalRange} where {B,F,N},s::LocalTensor,i::Vararg{Int})
    setindex!(codomain(m),fiber(s),i...)
    return s
end
function Base.setindex!(m::TensorField{B,F,N,<:AlignedSpace} where {B,F,N},s::LocalTensor,i::Vararg{Int})
    setindex!(codomain(m),fiber(s),i...)
    return s
end
function Base.setindex!(m::TensorField,s::LocalTensor,i::Vararg{Int})
    #setindex!(domain(m),base(s),i...)
    setindex!(codomain(m),fiber(s),i...)
    return s
end

extract(x::AbstractVector,i) = (@inbounds x[i])
extract(x::AbstractMatrix,i) = (@inbounds LocalTensor(points(x).v[end][i],x[:,i]))
extract(x::AbstractArray{T,3} where T,i) = (@inbounds LocalTensor(points(x).v[end][i],x[:,:,i]))
extract(x::AbstractArray{T,4} where T,i) = (@inbounds LocalTensor(points(x).v[end][i],x[:,:,:,i]))
extract(x::AbstractArray{T,5} where T,i) = (@inbounds LocalTensor(points(x).v[end][i],x[:,:,:,:,i]))

assign!(x::AbstractVector,i,s) = (@inbounds x[i] = s)
assign!(x::AbstractMatrix,i,s) = (@inbounds x[:,i] = s)
assign!(x::AbstractArray{T,3} where T,i,s) = (@inbounds x[:,:,i] = s)
assign!(x::AbstractArray{T,4} where T,i,s) = (@inbounds x[:,:,:,i] = s)
assign!(x::AbstractArray{T,5} where T,i,s) = (@inbounds x[:,:,:,:,i] = s)

Base.BroadcastStyle(::Type{<:TensorField{B,F,N,P,L}}) where {B,F,N,P,L} = Broadcast.ArrayStyle{TensorField{B,F,N,P,L}}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TensorField{B,F,N,P,L}}}, ::Type{ElType}) where {B,F,N,P,L,ElType}
    # Scan the inputs for the TensorField:
    t = find_tf(bc)
    # Use the domain field of t to create the output
    TensorField(domain(t), similar(Array{fibertype(ElType),N}, axes(bc)))
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GridBundle{N,C,PA,TA}}}, ::Type{ElType}) where {N,C,PA,TA,ElType}
    t = find_gf(bc)
    # Use the data type to create the output
    TensorField(t,similar(Array{ElType,N}, axes(bc)))
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{SimplexBundle{N,C,PA,TA}}}, ::Type{ElType}) where {N,C,PA,TA,ElType}
    t = find_gf(bc)
    # Use the data type to create the output
    TensorField(t,similar(Vector{ElType}, axes(bc)))
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{FaceBundle{N,C,PA,TA}}}, ::Type{ElType}) where {N,C,PA,TA,ElType}
    t = find_gf(bc)
    # Use the data type to create the output
    TensorField(t,similar(Vector{ElType}, axes(bc)))
end

#"`A = find_tf(As)` returns the first TensorField among the arguments."
find_tf(bc::Base.Broadcast.Broadcasted) = find_tf(bc.args)
find_tf(bc::Base.Broadcast.Extruded) = find_tf(bc.x)
find_tf(args::Tuple) = find_tf(find_tf(args[1]), Base.tail(args))
find_tf(x) = x
find_tf(::Tuple{}) = nothing
find_tf(a::TensorField, rest) = a
find_tf(::Any, rest) = find_tf(rest)

(m::TensorField{B,F,N,<:SimplexBundle} where {B,F,N})(i::ImmersedTopology) = TensorField(coordinates(m)(i),fiber(m)[vertices(i)])
(m::TensorField{B,F,N,<:GridBundle} where {B,F,N})(i::ImmersedTopology) = TensorField(base(m)(i),fiber(m))
for fun ∈ (:Open,:Mirror,:Clamped,:Torus,:Ribbon,:Wing,:Mobius,:Klein,:Cone,:Polar,:Sphere,:Geographic,:Hopf)
    for top ∈ (Symbol(fun,:Topology),)
        @eval begin
            $top(m::TensorField{B,F,N,<:GridBundle} where {B,F,N}) = TensorField($top(base(m)),fiber(m))
            $top(m::GridBundle) = m($top(size(m)))
            $top(p::PointArray) = TensorField(GridBundle(p,$top(size(p))))
            $(Symbol(fun,:Parameter))(p::PointArray) = $top(p)
        end
    end
end

spacing(x::AbstractVector) = sum(norm.(diff(fiber(x))))/(length(x)-1)
spacing(x::AbstractArray{T,N}) where {T,N} = minimum(spacing.(Ref(x),list(1,N)))
function spacing(x::AbstractArray,i)
    n = norm.(diff(fiber(x),dims=i))
    sum(n)/length(n)
end

valmat(t::Values{N,<:Vector},s=size(t[1])) where N = [Values((t[q][i] for q ∈ OneTo(N))...) for i ∈ OneTo(s[1])]
valmat(t::Values{N,<:Matrix},s=size(t[1])) where N = [Values((t[q][i,j] for q ∈ OneTo(N))...) for i ∈ OneTo(s[1]), j ∈ OneTo(s[2])]
valmat(t::Values{N,<:Array{T,3} where T},s=size(t[1])) where N = [Values((t[q][i,j,k] for q ∈ OneTo(N))...) for i ∈ OneTo(s[1]), j ∈ OneTo(s[2]), k ∈ OneTo(s[3])]

fromany(t::Chain{V,G,Any}) where {V,G} = Chain{V,G}(value(t)...)
fromany(t::Chain) = t

TensorField(t::Chain{V,G}) where {V,G} = TensorField(base(t[1]), Chain{V,G}.(valmat(fiber.(value(t)))))
Grassmann.Chain(t::TensorField{B,<:Union{Real,Complex}} where B) = Chain{Submanifold(ndims(t)),0}(t)
function Grassmann.Chain(t::TensorField{B,<:Chain{V,G}} where B) where {V,G}
    Chain{V,G}((TensorField(base(t), getindex.(fiber(t),j)) for j ∈ 1:binomial(mdims(V),G))...)
end
Base.:^(t::TensorField,n::Int) = TensorField(domain(t), .^(codomain(t),n,refmetric(t)))
for op ∈ (:+,:-,:&,:∧,:∨)
    let bop = op ∈ (:∧,:∨) ? :(Grassmann.$op) : :(Base.$op)
        @eval begin
            $bop(a::TensorField,b::TensorField) = checkdomain(a,b) && TensorField(domain(a), $op.(codomain(a),codomain(b)))
            $bop(a::TensorField,b::Number) = TensorField(domain(a), $op.(codomain(a),Ref(b)))
            $bop(a::Number,b::TensorField) = TensorField(domain(b), $op.(Ref(a),codomain(b)))
        end
    end
end
(m::TensorNested)(t::TensorField) = TensorField(base(t), m.(fiber(t)))
(m::TensorField{B,<:TensorNested} where B)(t::TensorField) = m⋅t
@inline Base.:<<(a::GlobalFiber,b::GlobalFiber) = contraction(b,~a)
@inline Base.:>>(a::GlobalFiber,b::GlobalFiber) = contraction(~a,b)
@inline Base.:<(a::GlobalFiber,b::GlobalFiber) = contraction(b,a)
Base.sign(a::TensorField) = TensorField(base(a), sign.(fiber(Real(a))))
Base.inv(a::TensorField{B,<:Real} where B) = TensorField(base(a), inv.(fiber(a)))
Base.inv(a::TensorField{B,<:Complex} where B) = TensorField(base(a), inv.(fiber(a)))
Base.:*(a::TensorField{B,<:Real} where B,b::TensorField{B,<:Real} where B) = TensorField(base(a), fiber(a).*fiber(b))
Base.:*(a::TensorField{B,<:Complex} where B,b::TensorField{B,<:Complex} where B) = TensorField(base(a), fiber(a).*fiber(b))
Base.:*(a::TensorField{B,<:Real} where B,b::TensorField{B,<:Complex} where B) = TensorField(base(a), fiber(a).*fiber(b))
Base.:*(a::TensorField{B,<:Complex} where B,b::TensorField{B,<:Real} where B) = TensorField(base(a), fiber(a).*fiber(b))
Base.:*(a::TensorField{B,<:Real} where B,b::TensorField) = TensorField(base(a), fiber(a).*fiber(b))
Base.:*(a::TensorField{B,<:Complex} where B,b::TensorField) = TensorField(base(a), fiber(a).*fiber(b))
Base.:*(a::TensorField,b::TensorField{B,<:Real} where B) = TensorField(base(a), fiber(a).*fiber(b))
Base.:*(a::TensorField,b::TensorField{B,<:Complex} where B) = TensorField(base(a), fiber(a).*fiber(b))
Base.:/(a::TensorField,b::TensorField{B,<:Real} where B) = TensorField(base(a), fiber(a)./fiber(b))
Base.:/(a::TensorField,b::TensorField{B,<:Complex} where B) = TensorField(base(a), fiber(a)./fiber(b))
LinearAlgebra.:×(a::TensorField{R},b::TensorField{R}) where R = TensorField(base(a), .⋆(fiber(a).∧fiber(b),refmetric(base(a))))
Grassmann.compound(t::TensorField,i::Val) = TensorField(base(t), compound.(fiber(t),i))
Grassmann.compound(t::TensorField,i::Int) = TensorField(base(t), compound.(fiber(t),Val(i)))
Grassmann.eigen(t::TensorField,i::Val) = TensorField(base(t), eigen.(fiber(t),i))
Grassmann.eigen(t::TensorField,i::Int) = TensorField(base(t), eigen.(fiber(t),Val(i)))
Grassmann.eigvals(t::TensorField,i::Val) = TensorField(base(t), eigvals.(fiber(t),i))
Grassmann.eigvals(t::TensorField,i::Int) = TensorField(base(t), eigvals.(fiber(t),Val(i)))
Grassmann.eigvecs(t::TensorField,i::Val) = TensorField(base(t), eigvecs.(fiber(t),i))
Grassmann.eigvecs(t::TensorField,i::Int) = TensorField(base(t), eigvecs.(fiber(t),Val(i)))
Grassmann.eigpolys(t::TensorField,G::Val) = TensorField(base(t), eigpolys.(fiber(t),G))
for (op,mop) ∈ ((:*,:wedgedot_metric),(:wedgedot,:wedgedot_metric),(:veedot,:veedot_metric),(:⋅,:contraction_metric),(:contraction,:contraction_metric),(:>,:contraction_metric),(:⊘,:⊘),(:>>>,:>>>),(:/,:/),(:^,:^))
    let bop = op ∈ (:*,:>,:>>>,:/,:^) ? :(Base.$op) : :(Grassmann.$op)
    @eval begin
        $bop(a::TensorField{R},b::TensorField{R}) where R = TensorField(base(a),Grassmann.$mop.(fiber(a),fiber(b),refmetric(base(a))))
        $bop(a::Number,b::TensorField) = TensorField(base(b), Grassmann.$op.(a,fiber(b)))
        $bop(a::TensorField,b::Number) = TensorField(base(a), Grassmann.$op.(fiber(a),b,$((op≠:^ ? () : (:(refmetric(base(a))),))...)))
    end end
end
for fun ∈ (:-,:!,:~,:real,:imag,:conj,:deg2rad,:transpose)
    @eval Base.$fun(t::TensorField) = TensorField(domain(t), $fun.(codomain(t)))
end
for fun ∈ (:exp,:exp2,:exp10,:log,:log2,:log10,:sinh,:cosh,:abs,:sqrt,:cbrt,:cos,:sin,:tan,:cot,:sec,:csc,:asec,:acsc,:sech,:csch,:asech,:tanh,:coth,:asinh,:acosh,:atanh,:acoth,:asin,:acos,:atan,:acot,:sinc,:cosc,:cis,:abs2,:inv)
    @eval Base.$fun(t::TensorField) = TensorField(domain(t), $fun.(codomain(t),ref(metricextensor(t))))
end
for fun ∈ (:reverse,:clifford,:even,:odd,:scalar,:vector,:bivector,:volume,:value,:complementleft,:realvalue,:imagvalue,:outermorphism,:Outermorphism,:DiagonalOperator,:TensorOperator,:eigen,:eigvecs,:eigvals,:eigvalsreal,:eigvalscomplex,:eigvecsreal,:eigvecscomplex,:eigpolys,:∧)
    @eval Grassmann.$fun(t::TensorField) = TensorField(domain(t), $fun.(codomain(t)))
end
for fun ∈ (:⋆,:angle,:radius,:complementlefthodge,:pseudoabs,:pseudoabs2,:pseudoexp,:pseudolog,:pseudoinv,:pseudosqrt,:pseudocbrt,:pseudocos,:pseudosin,:pseudotan,:pseudocosh,:pseudosinh,:pseudotanh,:metric,:unit)
    @eval Grassmann.$fun(t::TensorField) = TensorField(domain(t), $fun.(codomain(t),ref(metricextensor(t))))
end
for fun ∈ (:sum,:prod)
    @eval Base.$fun(t::TensorField) = LocalTensor(domain(t)[end], $fun(codomain(t)))
end
for fun ∈ (:cumsum,:cumprod)
    @eval function Base.$fun(t::TensorField)
         out = $fun(codomain(t))
         pushfirst!(out,zero(eltype(out)))
         TensorField(domain(t), out)
    end
end

Grassmann.signbit(::TensorField) = false
#Base.inv(t::TensorField) = TensorField(codomain(t), domain(t))
Base.diff(t::TensorField) = TensorField(diff(domain(t)), diff(codomain(t)))
absvalue(t::TensorField) = TensorField(domain(t), value.(abs.(codomain(t))))
LinearAlgebra.tr(t::TensorField) = TensorField(domain(t), tr.(codomain(t)))
LinearAlgebra.det(t::TensorField) = TensorField(domain(t), det.(codomain(t)))
LinearAlgebra.norm(t::TensorField) = TensorField(domain(t), norm.(codomain(t)))
(V::Submanifold)(t::TensorField) = TensorField(domain(t), V.(codomain(t)))
(::Type{T})(t::TensorField) where T<:Real = TensorField(domain(t), T.(codomain(t)))
(::Type{Complex})(t::TensorField) = TensorField(domain(t), Complex.(codomain(t)))
(::Type{Complex{T}})(t::TensorField) where T = TensorField(domain(t), Complex{T}.(codomain(t)))
Grassmann.Phasor(s::TensorField) = TensorField(domain(s), Phasor(codomain(s)))
Grassmann.Couple(s::TensorField) = TensorField(domain(s), Couple(codomain(s)))

checkdomain(a::GlobalFiber,b::GlobalFiber) = domain(a)≠domain(b) ? error("GlobalFiber base not equal") : true

disconnect(t::FaceMap) = TensorField(disconnect(base(t)),fiber(t))
discontinuous(t::FaceMap) = discontinuous(t,discontinuous(base(t)))
discontinuous(t::FaceMap,m) = TensorField(m,fiber(t))
discontinuous(t::SimplexMap) = discontinuous(t,discontinuous(base(t)))
discontinuous(t::SimplexMap,m) = isdiscontinuous(t) ? t : TensorField(m,view(fiber(t),vertices(m)))
graphbundle(t::TensorField{B,F,N,<:SimplexBundle} where {B,F,N}) = SimplexBundle(PointCloud(0,fiber(graph.(t))),isdiscontinuous(t) ? disconnect(immersion(t)) : immersion(t))

Variation(dom,cod::TensorField) = TensorField(dom,cod.(dom))
function Variation(cod::TensorField); p = points(cod).v[end]
    TensorField(p,leaf.(Ref(cod),1:length(p)))
end
function Variation(cod::TensorField{B,F,2,<:FiberProductBundle} where {B,F})
    p = base(cod).g.v[1]
    TensorField(p,leaf.(Ref(cod),1:length(p)))
end
const variation = Variation

alteration(dom,cod::TensorField) = TensorField(dom,cod.(dom))
function alteration(cod::TensorField); p = points(cod).v[1]
    TensorField(p,leaf.(Ref(cod),1:length(p),1))
end

modification(dom,cod::TensorField) = TensorField(dom,cod.(dom))
function modification(cod::TensorField); p = points(cod).v[2]
    TensorField(p,leaf.(Ref(cod),1:length(p),2))
end

variation(v::TensorField,fun::Function,args...) = variation(v,0.0,fun,args...)
variation(v::TensorField,fun::Function,fun!::Function,args...) = variation(v,0.0,fun,fun!,args...)
variation(v::TensorField,fun::Function,fun!::Function,f::Function,args...) = variation(v,0.0,fun,fun!,f,args...)
function variation(v::Variation,t,fun::Function,args...)
    for i ∈ 1:length(v)
        display(fun(v.cod[i],args...))
        sleep(t)
    end
end
function variation(v::Variation,t,fun::Function,fun!::Function,args...)
    out = fun(v.cod[1],args...)
    fig,ax,plt = out
    display(out)
    sleep(t)
    for i ∈ 2:length(v)
        empty!(ax)
        fun!(v.cod[i],args...)
        sleep(t)
    end
end
function variation(v::TensorField,t,fun::Function,args...)
    for i ∈ 1:length(points(v).v[end])
        display(fun(leaf(v,i),args...))
        sleep(t)
    end
end
function variation(v::TensorField,t,fun::Function,fun!::Function,args...)
    out = fun(leaf(v,1),args...)
    fig,ax,plt = out
    display(out)
    sleep(t)
    for i ∈ 2:length(points(v).v[end])
        empty!(ax)
        fun!(leaf(v,i),args...)
        sleep(t)
    end
end
function variation(v::TensorField{B,F,2,<:FiberProductBundle} where {B,F},t,fun::Function,args...)
    for i ∈ 1:length(base(v).g.v[1])
        display(fun(leaf(v,i),args...))
        sleep(t)
    end
end
function variation(v::TensorField{B,F,2,<:FiberProductBundle} where {B,F},t,fun::Function,fun!::Function,args...)
    out = fun(leaf(v,1),args...)
    fig,ax,plt = out
    display(out)
    sleep(t)
    for i ∈ 2:length(base(v).g.v[1])
        empty!(ax)
        fun!(leaf(v,i),args...)
        sleep(t)
    end
end

alteration(v::TensorField,fun::Function,args...) = alteration(v,0.0,fun,args...)
alteration(v::TensorField,fun::Function,fun!::Function,args...) = alteration(v,0.0,fun,fun!,args...)
alteration(v::TensorField,fun::Function,fun!::Function,f::Function,args...) = alteration(v,0.0,fun,fun!,f,args...)
function alteration(v::TensorField,t,fun::Function,args...)
    for i ∈ 1:length(points(v).v[1])
        display(fun(leaf(v,i,1),args...))
        sleep(t)
    end
end
function alteration(v::TensorField,t,fun::Function,fun!::Function,args...)
    out = fun(leaf(v,1,1),args...)
    fig,ax,plt = out
    display(out)
    sleep(t)
    for i ∈ 2:length(points(v).v[1])
        empty!(ax)
        fun!(leaf(v,i,1),args...)
        sleep(t)
    end
end

modification(v::TensorField,fun::Function,args...) = modification(v,0.0,fun,args...)
modification(v::TensorField,fun::Function,fun!::Function,args...) = modification(v,0.0,fun,fun!,args...)
modification(v::TensorField,fun::Function,fun!::Function,f::Function,args...) = modification(v,0.0,fun,fun!,f,args...)
function modification(v::TensorField,t,fun::Function,args...)
    for i ∈ 1:length(points(v).v[2])
        display(fun(leaf(v,i,2),args...))
        sleep(t)
    end
end
function modification(v::TensorField,t,fun::Function,fun!::Function,args...)
    out = fun(leaf(v,1,2),args...)
    fig,ax,plt = out
    display(out)
    sleep(t)
    for i ∈ 2:length(points(v).v[2])
        empty!(ax)
        fun!(leaf(v,i,2),args...)
        sleep(t)
    end
end

include("grid.jl")
include("element.jl")
include("diffgeo.jl")

function __init__()
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        export linegraph, linegraph!
        funsym(sym) = String(sym)[end] == '!' ? sym : Symbol(sym,:!)
        for lines ∈ (:lines,:lines!,:linesegments,:linesegments!)
            @eval begin
                Makie.$lines(t::ScalarMap;args...) = Makie.$lines(TensorField(GridBundle{1}(base(t)),fiber(t));args...)
                Makie.$lines(t::SpaceCurve,f::Function=speed;args...) = Makie.$lines(vec(fiber(t));color=Real.(vec(fiber(f(t)))),args...)
                Makie.$lines(t::PlaneCurve,f::Function=speed;args...) = Makie.$lines(vec(fiber(t));color=Real.(vec(fiber(f(t)))),args...)
                Makie.$lines(t::RealFunction,f::Function=speed;args...) = Makie.$lines(Real.(points(t)),Real.(fiber(t));color=Real.(vec(fiber(f(t)))),args...)
                Makie.$lines(t::ComplexMap{B,F,1},f::Function=speed;args...) where {B<:Coordinate{<:AbstractReal},F} = Makie.$lines(realvalue.(fiber(t)),imagvalue.(fiber(t));color=Real.(vec(fiber(f(t)))),args...)
                #Makie.$lines(t::TensorField{B,F<:AbstractReal,N,<:SimplexBundle};args...) = Makie.$lines(TensorField(GridBundle{1}(base(t)),fiber(t));args...)
            end
        end
        #Makie.lines(t::TensorField{B,F,1};args...) where {B<:Coordinate{<:AbstractReal},F<:AbstractReal} = linegraph(t;args...)
        #Makie.lines!(t::TensorField{B,F,1};args...) where {B<:Coordinate{<:AbstractReal},F<:AbstractReal} = linegraph(t;args...)
        for fun ∈ (:linegraph,:linegraph!)
            @eval begin
                $fun(t::SurfaceGrid;args...) = $fun(graph(t);args...)
                $fun(t::TensorField{B,<:AbstractReal,2,<:FiberProductBundle} where B;args...) = $fun(TensorField(GridBundle(base(t)),fiber(t)))
            end
        end
        function linegraph(t::GradedField{G,B,F,1} where G;args...) where {B<:Coordinate{<:AbstractReal},F}
            x,y = Real.(points(t)),value.(codomain(t))
            display(Makie.lines(x,Real.(getindex.(y,1));args...))
            for i ∈ 2:binomial(mdims(codomain(t)),grade(t))
                Makie.lines!(x,Real.(getindex.(y,i));args...)
            end
        end
        function linegraph!(t::GradedField{G,B,F,1} where G;args...) where {B<:Coordinate{<:AbstractReal},F}
            x,y = Real.(points(t)),value.(codomain(t))
            display(Makie.lines!(x,Real.(getindex.(y,1));args...))
            for i ∈ 2:binomial(mdims(codomain(t)),grade(t))
                Makie.lines!(x,Real.(getindex.(y,i));args...)
            end
        end
        function Makie.lines(t::IntervalMap{B,<:TensorOperator},f::Function=speed;args...) where B<:Coordinate{<:AbstractReal}
            display(Makie.lines(getindex.(t,1),f;args...))
            for i ∈ 2:mdims(eltype(codomain(t)))
                Makie.lines!(getindex.(t,i),f;args...)
            end
        end
        function Makie.lines!(t::IntervalMap{B,<:TensorOperator},f::Function=speed;args...) where B<:Coordinate{<:AbstractReal}
            display(Makie.lines!(getindex.(t,1),f;args...))
            for i ∈ 2:mdims(eltype(codomain(t)))
                Makie.lines!(getindex.(t,i),f;args...)
            end
        end
        export scaledarrows, scaledarrows!
        for fun ∈ (:arrows,:arrows!)
            @eval begin
                function $(Symbol(:scaled,fun))(M::VectorField,t::VectorField;args...)
                    kwargs = if haskey(args,:gridsize)
                        wargs = Dict(args)
                        delete!(wargs,:gridsize)
                        return $(Symbol(:scaled,fun))(resample(M,args[:gridsize]),resample(t,args[:gridsize]);(;wargs...)...)
                    elseif haskey(args,:arcgridsize)
                        wargs = Dict(args)
                        delete!(wargs,:arcgridsize)
                        aM = arcresample(M,args[:arcgridsize])
                        return $(Symbol(:scaled,fun))(aM,TensorField(base(aM),t.(points(aM)));(;wargs...)...)
                    else
                        args
                    end
                    s = spacing(M)/(sum(fiber(norm(t)))/length(t))
                    Makie.$fun(M,t;lengthscale=s/3,arrowsize=s/17,kwargs...)
                end
                function $(Symbol(:scaled,fun))(M::VectorField,t::TensorField{B,<:TensorOperator} where B;args...)
                    kwargs = if haskey(args,:gridsize)
                        wargs = Dict(args)
                        delete!(wargs,:gridsize)
                        return $(Symbol(:scaled,fun))(resample(M,args[:gridsize]),resample(t,args[:gridsize]);(;wargs...)...)
                    elseif haskey(args,:arcgridsize)
                        wargs = Dict(args)
                        delete!(wargs,:arcgridsize)
                        aM = arcresample(M,args[:arcgridsize])
                        return $(Symbol(:scaled,fun))(aM,TensorField(base(aM),t.(points(aM)));(;wargs...)...)
                    else
                        args
                    end
                    s = spacing(M)/maximum(value(sum(map.(norm,fiber(value(t))))/length(t)))
                    Makie.$fun(M,t;lengthscale=s/3,arrowsize=s/17,kwargs...)
                end
            end
        end
        function Makie.arrows(M::VectorField,t::TensorField{B,<:TensorOperator,N,<:GridBundle} where B;args...) where N
            Makie.arrows(TensorField(fiber(M),fiber(t));args...)
        end
        function Makie.arrows!(M::VectorField,t::TensorField{B,<:TensorOperator,N,<:GridBundle} where B;args...) where N
            Makie.arrows!(TensorField(fiber(M),fiber(t));args...)
        end
        function Makie.arrows(t::VectorField{<:Coordinate{<:Chain},<:TensorOperator,2,<:RealSpace{2}};args...)
            display(Makie.arrows(getindex.(t,1);args...))
            for i ∈ 2:mdims(eltype(codomain(t)))
                Makie.arrows!(getindex.(t,i);args...)
            end
        end
        function Makie.arrows!(t::VectorField{<:Coordinate{<:Chain},<:TensorOperator,2,<:RealSpace{2}};args...)
            display(Makie.arrows!(getindex.(t,1);args...))
            for i ∈ 2:mdims(eltype(codomain(t)))
                Makie.arrows!(getindex.(t,i);args...)
            end
        end
        for (fun,fun!) ∈ ((:arrows,:arrows!),(:streamplot,:streamplot!))
            @eval begin
                function Makie.$fun(t::TensorField{<:Coordinate{<:Chain},<:TensorOperator,N,<:GridBundle};args...) where N
                    display(Makie.$fun(getindex.(t,1);args...))
                    for i ∈ 2:mdims(eltype(codomain(t)))
                        Makie.$fun!(getindex.(t,i);args...)
                    end
                end
                function Makie.$fun!(t::TensorField{<:Coordinate{<:Chain},<:TensorOperator,N,<:GridBundle};args...) where N
                    display(Makie.$fun!(getindex.(t,1);args...))
                    for i ∈ 2:mdims(eltype(codomain(t)))
                        Makie.$fun!(getindex.(t,i);args...)
                    end
                end
            end
        end
        function Makie.streamplot(t::TensorField{<:Coordinate{<:Chain},<:TensorOperator,2,<:RealSpace{2}},m::U;args...) where U<:Union{<:VectorField,<:Function}
            display(Makie.streamplot(getindex.(t,1),m;args...))
            for i ∈ 2:mdims(eltype(codomain(t)))
                Makie.streamplot!(getindex.(t,i),m;args...)
            end
        end
        function Makie.streamplot!(t::TensorField{<:Coordinate{<:Chain},<:TensorOperator,2,<:RealSpace{2}},m::U;args...) where U<:Union{<:VectorField,<:Function}
            display(Makie.streamplot!(getindex.(t,1),m;args...))
            for i ∈ 2:mdims(eltype(codomain(t)))
                Makie.streamplot!(getindex.(t,i),m;args...)
            end
        end
        Makie.volume(t::VolumeGrid;args...) = Makie.volume(points(t).v...,Real.(codomain(t));args...)
        Makie.volume!(t::VolumeGrid;args...) = Makie.volume!(points(t).v...,Real.(codomain(t));args...)
        Makie.volumeslices(t::VolumeGrid;args...) = Makie.volumeslices(points(t).v...,Real.(codomain(t));args...)
        for fun ∈ (:surface,:surface!)
            @eval begin
                Makie.$fun(t::SurfaceGrid,f::Function=gradient_fast;args...) = Makie.$fun(points(t).v...,Real.(codomain(t));color=Real.(abs.(codomain(f(Real(t))))),args...)
                Makie.$fun(t::ComplexMap{B,<:AbstractComplex,2,<:RealSpace{2}} where B;args...) = Makie.$fun(points(t).v...,Real.(radius.(codomain(t)));color=Real.(angle.(codomain(t))),colormap=:twilight,args...)
                Makie.$fun(t::TensorField{B,<:AbstractReal,2,<:FiberProductBundle} where B;args...) = Makie.$fun(TensorField(GridBundle(base(t)),fiber(t)))
                function Makie.$fun(t::GradedField{G,B,F,2,<:RealSpace{2}} where G,f::Function=gradient_fast;args...) where {B,F<:Chain}
                    x,y = points(t),value.(codomain(t))
                    yi = Real.(getindex.(y,1))
                    display(Makie.$fun(x.v...,yi;color=Real.(abs.(codomain(f(x→yi)))),args...))
                    for i ∈ 2:binomial(mdims(eltype(codomain(t))),grade(t))
                        yi = Real.(getindex.(y,i))
                        Makie.$(funsym(fun))(x.v...,yi;color=Real.(abs.(codomain(f(x→yi)))),args...)
                    end
                end
            end
        end
        for fun ∈ (:contour,:contour!,:contourf,:contourf!,:contour3d,:contour3d!,:wireframe,:wireframe!)
            @eval begin
                Makie.$fun(t::ComplexMap{B,<:AbstractComplex,2,<:RealSpace{2}} where B;args...) = Makie.$fun(points(t).v...,Real.(radius.(codomain(t)));args...)
            end
        end
        for fun ∈ (:heatmap,:heatmap!)
            @eval begin
                Makie.$fun(t::ComplexMap{B,<:AbstractComplex,2,<:RealSpace{2}} where B;args...) = Makie.$fun(points(t).v...,Real.(angle.(codomain(t)));colormap=:twilight,args...)
            end
        end
        for fun ∈ (:contour,:contour!,:contourf,:contourf!,:contour3d,:contour3d!,:heatmap,:heatmap!)
            @eval begin
                Makie.$fun(t::SurfaceGrid;args...) = Makie.$fun(points(t).v...,Real.(codomain(t));args...)
                Makie.$fun(t::TensorField{B,<:AbstractReal,2,<:FiberProductBundle} where B;args...) = Makie.$fun(TensorField(GridBundle(base(t)),fiber(t)))
                function Makie.$fun(t::GradedField{G,B,F,2,<:RealSpace{2}} where G;args...) where {B,F}
                    x,y = points(t),value.(codomain(t))
                    display(Makie.$fun(x.v...,Real.(getindex.(y,1));args...))
                    for i ∈ 2:binomial(mdims(eltype(codomain(t))),G)
                        Makie.$(funsym(fun))(x.v...,Real.(getindex.(y,i));args...)
                    end
                end
            end
        end
        for fun ∈ (:wireframe,:wireframe!)
            @eval begin
                Makie.$fun(t::SurfaceGrid;args...) = Makie.$fun(graph(t);args...)
                Makie.$fun(t::TensorField{B,<:AbstractReal,2,<:FiberProductBundle} where B;args...) = Makie.$fun(TensorField(GridBundle(base(t)),fiber(t)))
            end
        end
        point2chain(x,V=Submanifold(2)) = Chain(x[1],x[2])
        chain3vec(x) = Makie.Vec3(x[1],x[2],x[3])
        for fun ∈ (:streamplot,:streamplot!)
            @eval begin
                Makie.$fun(f::Function,t::Rectangle;args...) = Makie.$fun(f,t.v...;args...)
                Makie.$fun(f::Function,t::Hyperrectangle;args...) = Makie.$fun(f,t.v...;args...)
                Makie.$fun(m::ScalarField{<:Coordinate{<:Chain},<:AbstractReal,N,<:RealSpace} where N;args...) = Makie.$fun(gradient_fast(m);args...)
                Makie.$fun(m::ScalarMap,dims...;args...) = Makie.$fun(gradient_fast(m),dims...;args...)
                Makie.$fun(m::VectorField{R,F,1,<:SimplexBundle} where {R,F},dims...;args...) = Makie.$fun(p->Makie.Point(m(Chain(one(eltype(p)),p.data...))),dims...;args...)
                Makie.$fun(m::VectorField{<:Coordinate{<:Chain},F,N,<:RealSpace} where {F,N};args...) = Makie.$fun(p->Makie.Point(m(Chain(p.data...))),points(m).v...;args...)
                Makie.$fun(t::TensorField{B,<:AbstractReal,2,<:FiberProductBundle} where B;args...) = Makie.$fun(TensorField(GridBundle(base(t)),fiber(t)))
                function Makie.$fun(M::VectorField,m::VectorField{<:Coordinate{<:Chain{V}},<:Chain,2,<:RealSpace{2}};args...) where V
                    kwargs = if haskey(args,:gridsize)
                        wargs = Dict(args)
                        delete!(wargs,:gridsize)
                        (;:gridsize => (args[:gridsize]...,1),wargs...)
                    else
                        pairs((;:gridsize => (32,32,1),args...))
                    end
                    w,gs = widths(points(m)),kwargs[:gridsize]
                    scale = 0.2sqrt(surfacearea(M)/prod(w))
                    st = Makie.$fun(p->(z=m(Chain{V}(p[1],p[2]));Makie.Point(z[1],z[2],0)),points(m).v...,Makie.ClosedInterval(-1e-15,1e-15);arrow_size=scale*minimum(w)/minimum((gs[1],gs[2])),kwargs...)
                    $(fun≠:streamplot ? :pl : :((fig,ax,pl))) = st
                    pl.transformation.transform_func[] = Makie.PointTrans{3}() do p
                        return Makie.Point(M(Chain{V}(p[1],p[2])))
                    end
                    jac,arr = jacobian(M),pl.plots[2]
                    arr.rotation[] = chain3vec.(jac.(point2chain.(arr.args[1][],V)).⋅point2chain.(arr.rotation[],V))
                    return st
                end
            end
        end
        for fun ∈ (:arrows,:arrows!)
            @eval begin
                Makie.$fun(t::ScalarField{<:Coordinate{<:Chain},F,2,<:RealSpace{2}} where F;args...) = Makie.$fun(vec(Makie.Point.(fiber(graph(Real(t))))),vec(Makie.Point.(fiber(normal(Real(t)))));args...)
                Makie.$fun(t::VectorField{<:Coordinate{<:Chain{W,L,F,2} where {W,L,F}},<:Chain{V,G,T,2} where {V,G,T},2,<:AlignedRegion{2}};args...) = Makie.$fun(points(t).v...,getindex.(codomain(t),1),getindex.(codomain(t),2);args...)
                Makie.$fun(t::VectorField{<:Coordinate{<:Chain},<:Chain,2,<:RealSpace{2}};args...) = Makie.$fun(Makie.Point.(vec(points(t))),Makie.Point.(vec(fiber(t)));args...)
                Makie.$fun(t::VectorField{<:Coordinate{<:Chain},F,N,<:GridBundle} where {F,N};args...) = Makie.$fun(vec(Makie.Point.(points(t))),vec(Makie.Point.(fiber(t)));args...)
                Makie.$fun(t::VectorField,f::VectorField;args...) = Makie.$fun(vec(Makie.Point.(fiber(t))),vec(Makie.Point.(fiber(f)));args...)
                Makie.$fun(t::Rectangle,f::Function;args...) = Makie.$fun(t.v...,f;args...)
                Makie.$fun(t::Hyperrectangle,f::Function;args...) = Makie.$fun(t.v...,f;args...)
                Makie.$fun(t::ScalarMap;args...) = Makie.$fun(points(points(t)),Real.(codomain(t));args...)
            end
        end
        #Makie.wireframe(t::ElementFunction;args...) = Makie.wireframe(value(domain(t));color=Real.(codomain(t)),args...)
        #Makie.wireframe!(t::ElementFunction;args...) = Makie.wireframe!(value(domain(t));color=Real.(codomain(t)),args...)
        Makie.convert_arguments(P::Makie.PointBased, a::SimplexBundle) = Makie.convert_arguments(P, Vector(points(a)))
        Makie.convert_single_argument(a::LocalFiber) = convert_arguments(P,Point(a))
        Makie.arrows(t::TensorField{B,F,N,<:SimplexBundle} where {B,F,N};args...) = Makie.arrows(GeometryBasics.Point.(↓(Manifold(base(t))).(points(t))),GeometryBasics.Point.(fiber(t));args...)
        Makie.arrows!(t::TensorField{B,F,N,<:SimplexBundle} where {B,F,N};args...) = Makie.arrows!(GeometryBasics.Point.(↓(Manifold(base(t))).(points(t))),GeometryBasics.Point.(fiber(t));args...)
        Makie.scatter(t::TensorField{B,F,N,<:SimplexBundle} where {B,F,N};args...) = Makie.scatter(submesh(base(t))[:,1],fiber(t);args...)
        Makie.scatter!(t::TensorField{B,F,N,<:SimplexBundle} where {B,F,N};args...) = Makie.scatter!(submesh(base(t))[:,1],fiber(t);args...)
        Makie.scatter(p::SimplexBundle;args...) = Makie.scatter(submesh(p);args...)
        Makie.scatter!(p::SimplexBundle;args...) = Makie.scatter!(submesh(p);args...)
        Makie.scatter(p::FaceBundle;args...) = Makie.scatter(submesh(fiber(means(p)));args...)
        Makie.scatter!(p::FaceBundle;args...) = Makie.scatter!(submesh(fiber(means(p)));args...)
        Makie.text(p::SimplexBundle;args...) = Makie.text(submesh(p);text=string.(vertices(p)),args...)
        Makie.text!(p::SimplexBundle;args...) = Makie.text!(submesh(p);text=string.(vertices(p)),args...)
        Makie.text(p::FaceBundle;args...) = Makie.text(submesh(fiber(means(p)));text=string.(subelements(p)),args...)
        Makie.text!(p::FaceBundle;args...) = Makie.text!(submesh(fiber(means(p)));text=string.(subelements(p)),args...)
        Makie.lines(p::SimplexBundle;args...) = Makie.lines(Vector(points(p));args...)
        Makie.lines!(p::SimplexBundle;args...) = Makie.lines!(Vector(points(p));args...)
        #Makie.lines(p::Vector{<:TensorAlgebra};args...) = Makie.lines(GeometryBasics.Point.(p);args...)
        #Makie.lines!(p::Vector{<:TensorAlgebra};args...) = Makie.lines!(GeometryBasics.Point.(p);args...)
        #Makie.lines(p::Vector{<:TensorTerm};args...) = Makie.lines(value.(p);args...)
        #Makie.lines!(p::Vector{<:TensorTerm};args...) = Makie.lines!(value.(p);args...)
        #Makie.lines(p::Vector{<:Chain{V,G,T,1} where {V,G,T}};args...) = Makie.lines(getindex.(p,1);args...)
        #Makie.lines!(p::Vector{<:Chain{V,G,T,1} where {V,G,T}};args...) = Makie.lines!(getindex.(p,1);args...)
        function Makie.linesegments(e::SimplexBundle;args...)
            sdims(immersion(e)) ≠ 2 && (return Makie.linesegments(edges(e)))
            Makie.linesegments(Grassmann.pointpair.(e[immersion(e)],↓(Manifold(e)));args...)
        end
        function Makie.linesegments!(e::SimplexBundle;args...)
            sdims(immersion(e)) ≠ 2 && (return Makie.linesegments!(edges(e)))
            Makie.linesegments!(Grassmann.pointpair.(e[immersion(e)],↓(Manifold(e)));args...)
        end
        Makie.wireframe(t::SimplexBundle;args...) = Makie.linesegments(edges(t);args...)
        Makie.wireframe!(t::SimplexBundle;args...) = Makie.linesegments!(edges(t);args...)
        for fun ∈ (:mesh,:mesh!,:wireframe,:wireframe!)
            @eval Makie.$fun(M::GridBundle;args...) = Makie.$fun(GeometryBasics.Mesh(M);args...)
        end
        function linegraph(M::TensorField{B,<:Chain,2,<:GridBundle} where B,f::Function=speed;args...)
            variation(M,Makie.lines,Makie.lines!,f;args...)
            alteration(M,Makie.lines!,Makie.lines!,f;args...)
        end
        function linegraph!(M::TensorField{B,<:Chain,2,<:GridBundle} where B,f::Function=speed;args...)
            variation(M,Makie.lines!,Makie.lines!,f;args...)
            alteration(M,Makie.lines!,Makie.lines!,f;args...)
        end
        function linegraph(v::TensorField{B,<:Chain,3,<:GridBundle} where B,f::Function=speed;args...)
            display(Makie.lines(leaf2(v,1,1,1),f;args...))
            c = (2,3),(1,3),(1,2)
            for k ∈ (1,2,3)
                for i ∈ 1:length(points(v).v[c[k][1]])
                    for j ∈ 1:length(points(v).v[c[k][2]])
                        Makie.lines!(leaf2(v,i,j,k),f;args...)
                    end
                end
            end
        end
        Makie.mesh(M::TensorField,f::Function;args...) = Makie.mesh(M,f(M);args...)
        Makie.mesh!(M::TensorField,f::Function;args...) = Makie.mesh!(M,f(M);args...)
        function Makie.mesh(M::TensorField{B,<:Chain,2,<:GridBundle} where B;args...)
            Makie.mesh(GridBundle(fiber(M));args...)
        end
        function Makie.mesh!(M::TensorField{B,<:Chain,2,<:GridBundle} where B;args...)
            Makie.mesh!(GridBundle(fiber(M));args...)
        end
        function Makie.mesh(M::TensorField{B,<:AbstractReal,2,<:GridBundle} where B;args...)
            Makie.mesh(GeometryBasics.Mesh(base(M));color=vec(fiber(Real(M))),args...)
        end
        function Makie.mesh!(M::TensorField{B,<:AbstractReal,2,<:GridBundle} where B;args...)
            Makie.mesh!(GeometryBasics.Mesh(base(M));color=vec(fiber(Real(M))),args...)
        end
        function Makie.mesh(M::TensorField{B,<:Chain,2,<:GridBundle} where B,f::TensorField;args...)
            Makie.mesh(GridBundle(fiber(M));color=vec(fiber(Real(f))),args...)
        end
        function Makie.mesh!(M::TensorField{B,<:Chain,2,<:GridBundle} where B,f::TensorField;args...)
            Makie.mesh!(GridBundle(fiber(M));color=vec(fiber(Real(f))),args...)
        end
        Makie.wireframe(M::TensorField{B,<:Chain,N,<:GridBundle} where {B,N};args...) = Makie.wireframe(GridBundle(fiber(M));args...)
        Makie.wireframe!(M::TensorField{B,<:Chain,N,<:GridBundle} where {B,N};args...) = Makie.wireframe!(GridBundle(fiber(M));args...)
        Makie.mesh(M::TensorField{B,F,N,<:FaceBundle} where {B,F,N};args...) = Makie.mesh(interp(M);args...)
        Makie.mesh!(M::TensorField{B,F,N,<:FaceBundle} where {B,F,N};args...) = Makie.mesh!(interp(M);args...)
        Makie.mesh(t::ScalarMap;args...) = Makie.mesh(domain(t);color=Real.(codomain(t)),args...)
        Makie.mesh!(t::ScalarMap;args...) = Makie.mesh!(domain(t);color=Real.(codomain(t)),args...)
        function Makie.mesh(M::SimplexBundle;args...)
            if mdims(M) == 2
                sm = submesh(M)[:,1]
                Makie.lines(sm,args[:color])
                Makie.plot!(sm,args[:color])
            else
                Makie.mesh(submesh(M),array(immersion(M));args...)
            end
        end
        function Makie.mesh!(M::SimplexBundle;args...)
            if mdims(M) == 2
                sm = submesh(M)[:,1]
                Makie.lines!(sm,args[:color])
                Makie.plot!(sm,args[:color])
            else
                Makie.mesh!(submesh(M),array(immersion(M));args...)
            end
        end
        function Makie.surface(M::ScalarMap,f::Function=laplacian;args...)
            fM = f(M)
            col = isdiscontinuous(M) && !isdiscontinuous(fM) ? discontinuous(fM,base(M)) : fM
            Makie.mesh(hcat(submesh(base(M)),Real.(fiber(M))),array(immersion(M));color=fiber(col),args...)
        end
        function Makie.surface!(M::ScalarMap,f::Function=laplacian;args...)
            fM = f(M)
            col = isdiscontinuous(M) && !isdiscontinuous(fM) ? discontinuous(fM,base(M)) : fM
            Makie.mesh!(hcat(submesh(base(M)),Real.(fiber(M))),array(immersion(M));color=fiber(col),args...)
        end
        function Makie.surface(M::TensorField{B,<:AbstractReal,1,<:FaceBundle} where B,f::Function=laplacian;args...)
            Makie.surface(interp(M),f;args...)
        end
        function Makie.surface!(M::TensorField{B,<:AbstractReal,1,<:FaceBundle} where B,f::Function=laplacian;args...)
            Makie.surface!(interp(M),f;args...)
        end
    end
    @require UnicodePlots="b8865327-cd53-5732-bb35-84acbb429228" begin
        function UnicodePlots.scatterplot(p::SimplexBundle;args...)
            s = submesh(p)
            UnicodePlots.scatterplot(s[:,1],s[:,2];args...)
        end
        function UnicodePlots.scatterplot(p::FaceBundle;args...)
            s = submesh(fiber(means(p)))
            UnicodePlots.scatterplot(s[:,1],s[:,2];args...)
        end
        function UnicodePlots.scatterplot!(P,p::SimplexBundle;args...)
            s = submesh(p)
            UnicodePlots.scatterplot(P,s[:,1],s[:,2];args...)
        end
        function UnicodePlots.scatterplot!(P,p::FaceBundle;args...)
            s = submesh(fiber(means(p)))
            UnicodePlots.scatterplot(P,s[:,1],s[:,2];args...)
        end
        UnicodePlots.scatterplot(t::TensorField{B,F,N,<:SimplexBundle} where {B,F,N};args...) = UnicodePlots.scatterplot(submesh(base(t))[:,1],fiber(t);args...)
        UnicodePlots.scatterplot!(P,t::TensorField{B,F,N,<:SimplexBundle} where {B,F,N};args...) = UnicodePlots.scatterplot!(P,submesh(base(t))[:,1],fiber(t);args...)
        UnicodePlots.lineplot(t::ScalarMap;args...) = UnicodePlots.lineplot(getindex.(domain(t),2),codomain(t);args...)
        UnicodePlots.lineplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::ScalarMap;args...) = UnicodePlots.lineplot!(p,getindex.(domain(t),2),codomain(t);args...)
        UnicodePlots.lineplot(t::PlaneCurve;args...) = UnicodePlots.lineplot(getindex.(codomain(t),1),getindex.(codomain(t),2);args...)
        UnicodePlots.lineplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::PlaneCurve;args...) = UnicodePlots.lineplot!(p,getindex.(codomain(t),1),getindex.(codomain(t),2);args...)
        UnicodePlots.lineplot(t::RealFunction;args...) = UnicodePlots.lineplot(Real.(points(t)),Real.(codomain(t));args...)
        UnicodePlots.lineplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::RealFunction;args...) = UnicodePlots.lineplot!(p,Real.(points(t)),Real.(codomain(t));args...)
        UnicodePlots.lineplot(t::ComplexMap{B,<:AbstractComplex,1};args...) where B<:Coordinate{<:AbstractReal} = UnicodePlots.lineplot(real.(Complex.(codomain(t))),imag.(Complex.(codomain(t)));args...)
        UnicodePlots.lineplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::ComplexMap{B,<:AbstractComplex,1};args...) where B<:Coordinate{<:AbstractReal} = UnicodePlots.lineplot!(p,real.(Complex.(codomain(t))),imag.(Complex.(codomain(t)));args...)
        UnicodePlots.lineplot(t::GradedField{G,B,F,1} where {G,F};args...) where B<:Coordinate{<:AbstractReal} = UnicodePlots.lineplot(Real.(points(t)),Grassmann.array(codomain(t));args...)
        UnicodePlots.lineplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::GradedField{G,B,F,1} where {G,F};args...) where B<:Coordinate{<:AbstractReal} = UnicodePlots.lineplot!(p,Real.(points(t)),Grassmann.array(codomain(t));args...)
        UnicodePlots.contourplot(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F};args...) = UnicodePlots.contourplot(points(t).v[1][2:end-1],points(t).v[2][2:end-1],(x,y)->radius(t(Chain(x,y)));args...)
        UnicodePlots.contourplot(t::SurfaceGrid;args...) = UnicodePlots.contourplot(points(t).v[1][2:end-1],points(t).v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
        UnicodePlots.surfaceplot(t::SurfaceGrid;args...) = UnicodePlots.surfaceplot(points(t).v[1][2:end-1],points(t).v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
        UnicodePlots.surfaceplot(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F};args...) = UnicodePlots.surfaceplot(points(t).v[1][2:end-1],points(t).v[2][2:end-1],(x,y)->radius(t(Chain(x,y)));colormap=:twilight,args...)
        UnicodePlots.spy(t::SurfaceGrid;args...) = UnicodePlots.spy(Real.(codomain(t));args...)
        UnicodePlots.spy(p::SimplexBundle) = UnicodePlots.spy(antiadjacency(p))
        UnicodePlots.heatmap(t::SurfaceGrid;args...) = UnicodePlots.heatmap(Real.(codomain(t));xfact=step(points(t).v[1]),yfact=step(points(t).v[2]),xoffset=points(t).v[1][1],yoffset=points(t).v[2][1],args...)
        UnicodePlots.heatmap(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F};args...) = UnicodePlots.heatmap(Real.(angle.(codomain(t)));xfact=step(points(t).v[1]),yfact=step(points(t).v[2]),xoffset=points(t).v[1][1],yoffset=points(t).v[2][1],colormap=:twilight,args...)
        Base.display(t::PlaneCurve) = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::RealFunction) = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::ComplexMap{B,<:AbstractComplex,1,<:Interval}) where B = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::ComplexMap{B,<:AbstractComplex,2,<:RealSpace{2}} where B) = (display(typeof(t)); display(UnicodePlots.heatmap(t)))
        Base.display(t::GradedField{G,B,<:TensorGraded,1,<:Interval} where {G,B}) = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
        Base.display(t::SurfaceGrid) = (display(typeof(t)); display(UnicodePlots.heatmap(t)))
    end
    @require Meshes = "eacbb407-ea5a-433e-ab97-5258b1ca43fa" begin
        function SimplexBundle(m::Meshes.SimpleMesh{N}) where N
            c,f = Meshes.vertices(m),m.topology.connec
            s = N+1; V = Submanifold(ℝ^s) # s
            n = length(f[1].indices)
            p = PointCloud([Chain{V,1}(Values{s,Float64}(1.0,k.coords...)) for k ∈ c])
            p(SimplexTopology([Values{n,Int}(k.indices) for k ∈ f],length(p)))
        end
    end
    @require GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326" begin
        Base.convert(::Type{GeometryBasics.Point},t::T) where T<:LocalFiber = GeometryBasics.Point(base(t))
        #GeometryBasics.Point(t::T) where T<:TensorAlgebra = convert(GeometryBasics.Point,t)
        function GeometryBasics.Mesh(m::GridBundle{2})
            nm = size(points(m))
            faces = GeometryBasics.Tesselation(GeometryBasics.Rect(0, 0, 1, 1), nm)
            uv = Chain(0.0,0.0):map(inv,Chain((nm.-1)...)):Chain(1.0,1.0)
            GeometryBasics.Mesh(GeometryBasics.Point.(vec(points(m))), GeometryBasics.decompose(GeometryBasics.QuadFace{GeometryBasics.GLIndex}, faces), uv=GeometryBasics.Vec{2}.(value.(vec(uv))))
        end
        function SimplexBundle(m::GeometryBasics.Mesh)
            c,f = GeometryBasics.coordinates(m),GeometryBasics.faces(m)
            s = size(eltype(c))[1]+1; V = varmanifold(s) # s
            n = size(eltype(f))[1]
            p = PointCloud([Chain{V,1}(Values{s,Float64}(1.0,k...)) for k ∈ c])
            p(SimplexTopology([Values{n,Int}(k) for k ∈ f],length(p)))
        end
    end
    @require Delaunay="07eb4e4e-0c6d-46ef-bc4e-83d5e5d860a9" begin
        Delaunay.delaunay(p::PointCloud) = Delaunay.delaunay(points(p))
        Delaunay.delaunay(p::Vector{<:Chain}) = initmesh(Delaunay.delaunay(submesh(p)))
        initmesh(t::Delaunay.Triangulation) = initmeshdata(t.points',t.convex_hull',t.simplices')
    end
    @require QHull="a8468747-bd6f-53ef-9e5c-744dbc5c59e7" begin
        QHull.chull(p::Vector{<:Chain},n=1:length(p)) = QHull.chull(PointCloud(p),n)
        function QHull.chull(p::PointCloud,n=1:length(p))
            T = QHull.chull(submesh(length(n)==length(p) ? p : p[n]))
            p(SimplexTopology([Values(getindex.(Ref(n),k)) for k ∈ T.simplices],length(p)))
        end
        function SimplexBundle(t::Chull)
            p = PointCloud(initpoints(t.points'))
            p(SimplexTopology(Values.(t.simplices),length(p)))
        end
    end
    @require MiniQhull="978d7f02-9e05-4691-894f-ae31a51d76ca" begin
        MiniQhull.delaunay(p::Vector{<:Chain},args...) = MiniQhull.delaunay(PointCloud(p),1:length(p),args...)
        MiniQhull.delaunay(p::Vector{<:Chain},n::AbstractVector,args...) = MiniQhull.delaunay(PointCloud(p),n,args...)
        MiniQhull.delaunay(p::PointCloud,args...) = MiniQhull.delaunay(p,1:length(p),args...)
        function MiniQhull.delaunay(p::PointCloud,n::AbstractVector,args...)
            N,m = mdims(p),length(n)
            l = list(1,N)
            T = MiniQhull.delaunay(Matrix(submesh(m==length(p) ? p : fullpoints(p)[n])'),args...)
            p(SimplexTopology([Values{N,Int}(getindex.(Ref(n),Int.(T[l,k]))) for k ∈ 1:size(T,2)],length(p)))
        end
    end
    @require Triangulate = "f7e6ffb2-c36d-4f8f-a77e-16e897189344" begin
        const triangle_point_cache = (Array{T,2} where T)[]
        const triangle_simplex_cache = (Array{T,2} where T)[]
        function triangle_point(p::Array{T,2} where T,B)
            for k ∈ length(triangle_point_cache):B
                push!(triangle_point_cache,Array{Any,2}(undef,0,0))
            end
            triangle_point_cache[B] = p
        end
        function triangle_simplex(p::Array{T,2} where T,B)
            for k ∈ length(triangle_simplex_cache):B
                push!(triangle_simplex_cache,Array{Any,2}(undef,0,0))
            end
            triangle_simplex_cache[B] = p
        end
        function triangle(p::PointCloud)
            B = bundle(p)
            iszero(B) && (return array(p)'[2:end,:])
            if length(triangle_point_cache)<B || isempty(triangle_point_cache[B])
                triangle_point(array(p)'[2:end,:],B)
            else
                return triangle_point_cache[B]
            end
        end
        function triangle(p::SimplexTopology)
            B = p.id
            if length(triangle_simplex_cache)<B || isempty(triangle_simplex_cache[B])
                triangle_simplex(Cint.(array(p)'),B)
            else
                return triangle_simplex_cache[B]
            end
        end
        triangle(p::Vector{<:Chain{V,1,T} where V}) where T = array(p)'[2:end,:]
        triangle(p::Vector{<:Values}) = Cint.(array(p)')
        function Triangulate.TriangulateIO(e::SimplexBundle,h=nothing)
            triin=Triangulate.TriangulateIO()
            triin.pointlist=triangle(fullcoordinates(e))
            triin.segmentlist=triangle(immersion(e))
            !isnothing(h) && (triin.holelist=triangle(h))
            return triin
        end
        function Triangulate.triangulate(i,e::SimplexBundle;holes=nothing)
            initmesh(Triangulate.triangulate(i,Triangulate.TriangulateIO(e,holes))[1])
        end
        initmesh(t::Triangulate.TriangulateIO) = initmeshdata(t.pointlist,t.segmentlist,t.trianglelist,Val(2))
        #aran(area=0.001,angle=20) = "pa$(Printf.@sprintf("%.15f",area))q$(Printf.@sprintf("%.15f",angle))Q"
    end
    @require TetGen="c5d3f3f7-f850-59f6-8a2e-ffc6dc1317ea" begin
        function TetGen.JLTetGenIO(mesh::SimplexBundle;
                marker = :markers, holes = TetGen.Point{3, Float64}[])
            f = TetGen.TriangleFace{Cint}.(immersion(mesh))
            kw_args = Any[:facets => f,:holes => holes]
            if hasproperty(f, marker)
                push!(kw_args, :facetmarkers => getproperty(f, marker))
            end
            pm = points(mesh); V = Manifold(pm)
            TetGen.JLTetGenIO(TetGen.Point.(↓(V).(pm)); kw_args...)
        end
        function initmesh(tio::TetGen.JLTetGenIO, command = "Qp")
            r = TetGen.tetrahedralize(tio, command); V = Submanifold(ℝ^4)
            p = PointCloud([Chain{V,1}(Values{4,Float64}(1.0,k...)) for k ∈ r.points])
            t = Values{4,Int}.(r.tetrahedra)
            e = Values{3,Int}.(r.trifaces) # Values{2,Int}.(r.edges)
            n = Ref(length(p))
            return p(SimplexTopology(t,n)),p(SimplexTopology(e,n))
        end
        function TetGen.tetrahedralize(mesh::SimplexBundle, command = "Qp";
                marker = :markers, holes = TetGen.Point{3, Float64}[])
            initmesh(TetGen.JLTetGenIO(mesh;marker=marker,holes=holes),command)
        end
    end
    @require MATLAB="10e44e05-a98a-55b3-a45b-ba969058deb6" begin
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
            B = bundle(p)
            if length(matlab_cache)<B || isempty(matlab_cache[B])
                ap = array(p)'
                matlab(ap[2:end,:],B)
            else
                return matlab_cache[B]
            end
        end
        function matlab(p::SimplexTopology)
            B = bundle(p)
            if length(matlab_top_cache)<B || isempty(matlab_top_cache[B])
                ap = array(p)'
                matlab_top(vcat(ap,ones(length(p))'),B)
            else
                return matlab_top_cache[B]
            end
        end
        initmesh(g,args...) = initmeshall(g,args...)[list(1,3)]
        initmeshall(g::Matrix{Int},args...) = initmeshall(Matrix{Float64}(g),args...)
        function initmeshall(g,args...)
            P,E,T = MATLAB.mxcall(:initmesh,3,g,args...)
            pt,pe = initmeshdata(P,E,T,Val(2))
            return (pt,pe,T,E,P)
        end
        function initmeshes(g,args...)
            pt,pe,T = initmeshall(g,args...)
            pt,pe,TensorField(FaceBundle(pt),Int[T[end,k] for k ∈ 1:size(T,2)])
        end
        totalmesh(g,args...) = totalmeshall(g,args...)[list(1,3)]
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
        function refinemesh(g,args...)
            pt,pe,T,E,P = initmeshall(g,args...)
            matlab(P,bundle(fullcoordinates(pt)))
            matlab_top(E,bundle(immersion(pe)))
            matlab_top(T,bundle(immersion(pt)))
            return (g,refine(pt),refine(pe))
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
            p,e,t = unbundle(pt,pe)
            V = Manifold(p)
            P,E,T = refinemesh(g,p,e,t,s...); l = size(P,1)+1
            matlab(P,bundle(p)); matlab(E,bundle(e)); matlab(T,bundle(t))
            submesh!(p); array!(p); array!(t)
            deletepointcloud!(bundle(p))
            el,tl = list(1,l-1),list(1,l)
            np,ne,nt = size(P,2),size(E,2),size(T,2)
            ip = length(p)+1:np
            it = length(t)+1:nt
            totalnodes!(t,np)
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
    end
end

end # module Cartan
