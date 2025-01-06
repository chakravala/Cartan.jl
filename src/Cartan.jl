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

using SparseArrays, LinearAlgebra, Base.Threads, Grassmann, Requires
import Grassmann: value, vector, valuetype, tangent, istangent, Derivation, radius, ⊕
import Grassmann: realvalue, imagvalue, points, metrictensor, metricextensor
import Grassmann: Values, Variables, FixedVector, list, volume, compound
import Grassmann: Scalar, GradedVector, Bivector, Trivector
import Base: @pure, OneTo, getindex
import LinearAlgebra: cross
#import ElasticArrays: resize_lastdim!

export Values, Derivation
export initmesh, pdegrad, det

include("topology.jl")

export IntervalMap, RectangleMap, HyperrectangleMap, PlaneCurve, SpaceCurve
export TensorField, ScalarField, VectorField, BivectorField, TrivectorField
export ElementFunction, SurfaceGrid, VolumeGrid, ScalarGrid, Variation
export RealFunction, ComplexMap, SpinorField, CliffordField
export ScalarMap, GradedField, QuaternionField, PhasorField
export GlobalFrame, DiagonalField, EndomorphismField, OutermorphismField
export ParametricMap, RectangleMap, HyperrectangleMap, AbstractCurve
export metrictensorfield, metricextensorfield
export alteration, variation, modification

# GlobalSection

struct GlobalSection{B,F,N,BA<:AbstractFrameBundle{B,N},FA<:AbstractFrameBundle{F,N}} <: GlobalFiber{LocalSection{B,F},N}
    dom::BA
    cod::FA
    GlobalSection(dom::BA,cod::FA) where {B,F,N,BA<:AbstractFrameBundle{B,N},FA<:AbstractFrameBundle{F,N}} = new{B,F,N,BA,FA}(dom,cod)
end

# TensorField

struct TensorField{B,F,N,M<:AbstractFrameBundle{B,N}} <: GlobalFiber{LocalTensor{B,F},N}
    dom::M
    cod::Array{F,N}#ElasticArray{F,N,L,Vector{F}}
    #function TensorField(dom::M,cod::ElasticArray{F,N,L,Vector{F}}) where {B,F,N,M<:AbstractFrameBundle{B,N},L}
    function TensorField(dom::M,cod::Array{F,N}) where {B,F,N,M<:AbstractFrameBundle{B,N}}
        new{B,F,N,M}(dom,cod)
    end
end

#function TensorField(dom::M,cod::Array{F,N}) where {B,F,N,M<:AbstractFrameBundle{B,N}}
#    TensorField(dom,ElasticArray(cod))
#end
function TensorField(id::Int,dom::PA,cod::DenseArray{F,N},met::GA=Global{N}(InducedMetric())) where {N,P,F,PA<:AbstractArray{P,N},GA<:AbstractArray}
    TensorField(GridFrameBundle(id,PointArray(0,dom,met)),cod)
end
function TensorField(id::Int,dom::P,cod::DenseVector{F},met::G=Global{N}(InducedMetric())) where {F,P<:PointCloud,G<:AbstractVector}
    TensorField(SimplexFrameBundle(id,dom,met),cod)
end
TensorField(id::Int,dom,cod::DenseArray,met::GlobalFiber) = TensorField(id,dom,cod,fiber(met))
TensorField(dom::AbstractFrameBundle,cod::AbstractFrameBundle) = TensorField(dom,points(cod))
TensorField(dom::AbstractArray{B,N} where B,cod::DenseArray{F,N} where F,met::AbstractArray=Global{N}(InducedMetric())) where N = TensorField((global grid_id+=1),dom,cod,fiber(met))
TensorField(dom::ChainBundle,cod::DenseVector,met::AbstractVector=Global{1}(InducedMetric())) = TensorField((global grid_id+=1),dom,cod,met)
TensorField(a::TensorField,b::TensorField) = TensorField(fiber(a),fiber(b))

#const ParametricMesh{B,F,P<:AbstractVector{<:Chain}} = TensorField{B,F,1,P}
const ScalarMap{B,F<:AbstractReal,P<:SimplexFrameBundle} = TensorField{B,F,1,P}
#const ElementFunction{B,F<:AbstractReal,P<:AbstractVector} = TensorField{B,F,1,P}
const IntervalMap{B,F,P<:Interval} = TensorField{B,F,1,P}
const RectangleMap{B,F,P<:RealSpace{2}} = TensorField{B,F,2,P}
const HyperrectangleMap{B,F,P<:RealSpace{3}} = TensorField{B,F,3,P}
const ParametricMap{B,F,N,P<:RealSpace} = TensorField{B,F,N,P}
const Variation{B,F<:TensorField,N,P} = TensorField{B,F,N,P}
const RealFunction{B,F<:AbstractReal,P<:Interval} = TensorField{B,F,1,P}
const PlaneCurve{B,F<:Chain{V,G,Q,2} where {V,G,Q},P<:Interval} = TensorField{B,F,1,P}
const SpaceCurve{B,F<:Chain{V,G,Q,3} where {V,G,Q},P<:Interval} = TensorField{B,F,1,P}
const AbstractCurve{B,F<:Chain,P<:Interval} = TensorField{B,F,1,P}
const SurfaceGrid{B,F<:AbstractReal,P<:RealSpace{2}} = TensorField{B,F,2,P}
const VolumeGrid{B,F<:AbstractReal,P<:RealSpace{3}} = TensorField{B,F,3,P}
const ScalarGrid{B,F<:AbstractReal,N,P<:RealSpace{N}} = TensorField{B,F,N,P}
const GlobalFrame{B<:LocalFiber{P,<:TensorNested} where P,N} = GlobalSection{B,N}
const DiagonalField{B,F<:DiagonalOperator,N,P} = TensorField{B,F,N,P}
const EndomorphismField{B,F<:Endomorphism,N,P} = TensorField{B,F,N,P}
const OutermorphismField{B,F<:Outermorphism,N,P} = TensorField{B,F,N,P}
const CliffordField{B,F<:Multivector,N,P} = TensorField{B,F,N,P}
const QuaternionField{B,F<:Quaternion,N,P} = TensorField{B,F,N,P}
const ComplexMap{B,F<:AbstractComplex,N,P} = TensorField{B,F,N,P}
const PhasorField{B,F<:Phasor,N,P} = TensorField{B,F,N,P}
const SpinorField{B,F<:AbstractSpinor,N,P} = TensorField{B,F,N,P}
const GradedField{G,B,F<:Chain{V,G} where V,N,P} = TensorField{B,F,N,P}
const ScalarField{B,F<:AbstractReal,N,P} = TensorField{B,F,N,P}
const VectorField = GradedField{1}
const BivectorField = GradedField{2}
const TrivectorField = GradedField{3}

for bundle ∈ (:TensorField,:GlobalSection)
    @eval begin
        $bundle(dom,fun::BitArray) = $bundle(dom, Float64.(fun))
        $bundle(dom,fun::$bundle) = $bundle(dom, fiber(fun))
        $bundle(dom::$bundle,fun) = $bundle(base(dom), fun)
        $bundle(dom::$bundle,fun::DenseArray) = $bundle(base(dom), fun)
        $bundle(dom::$bundle,fun::Function) = $bundle(base(dom), fun)
        $bundle(dom::AbstractArray,fun::AbstractRange) = $bundle(dom, collect(fun))
        $bundle(dom::AbstractArray,fun::RealRegion) = $bundle(dom, collect(fun))
        $bundle(dom::AbstractArray,fun::Function) = $bundle(dom, fun.(dom))
        $bundle(dom::AbstractArray) = $bundle(dom, dom)
        $bundle(dom::ChainBundle,fun::Function) = $bundle(dom, fun.(value(points(dom))))
        basetype(::$bundle{B}) where B = B
        basetype(::Type{<:$bundle{B}}) where B = B
        Base.broadcast(f,t::$bundle) = $bundle(domain(t), f.(codomain(t)))
    end
end

←(F,B) = B → F
const → = TensorField
base(t::GlobalSection) = t.dom
fiber(t::GlobalSection) = t.cod
base(t::TensorField) = t.dom
fiber(t::TensorField) = t.cod
coordinates(t::TensorField) = base(t)
points(t::TensorField) = points(base(t))
metricextensor(t::TensorField) = metricextensor(base(t))
metrictensor(t::TensorField) = metrictensor(base(t))
metricextensorfield(t::TensorField) = metricextensorfield(base(t))
metrictensorfield(t::TensorField) = metrictensorfield(base(t))
metricextensorfield(t::GridFrameBundle) = TensorField(GridFrameBundle(PointArray(points(t)),immersion(t)),metricextensor(t))
metrictensorfield(t::GridFrameBundle) = TensorField(GridFrameBundle(PointArray(points(t)),immersion(t)),metrictensor(t))
metricextensorfield(t::SimplexFrameBundle) = TensorField(SimplexFrameBundle(PointCloud(points(t)),immersion(t)),metricextensor(t))
metrictensorfield(t::SimplexFrameBundle) = TensorField(SimplexFrameBundle(PointCloud(points(t)),immersion(t)),metrictensor(t))
immersion(t::TensorField) = immersion(base(t))
pointtype(m::TensorField) = pointtype(base(m))
pointtype(m::Type{<:TensorField}) = pointtype(basetype(m))
metrictype(m::TensorField) = metrictype(base(m))
metrictype(m::Type{<:TensorField}) = metrictype(basetype(m))
fibertype(::GlobalSection{B,F} where B) where F = F
fibertype(::Type{<:GlobalSection{B,F} where B}) where F = F
fibertype(::TensorField{B,F} where B) where F = F
fibertype(::Type{<:TensorField{B,F} where B}) where F = F
isopen(t::TensorField) = isopen(immersion(t))
iscompact(t::TensorField) = iscompact(immersion(t))
PointCloud(t::TensorField) = PointCloud(base(t))
Grassmann.grade(::GradedField{G}) where G = G

resize(t::TensorField) = TensorField(resize(domain(t)),codomain(t))
resize(t::GlobalSection) = GlobalSection(resize(domain(t)),codomain(t))

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
    setindex!(domain(m),base(s),i...)
    setindex!(codomain(m),fiber(s),i...)
    return s
end

@pure Base.eltype(::Type{<:GlobalSection{B,F}}) where {B,F} = LocalSection{B,F}
Base.getindex(m::GlobalSection,i::Vararg{Int}) = LocalSection(getindex(domain(m),i...), getindex(codomain(m),i...))
Base.getindex(m::GlobalSection,i::Vararg{Union{Int,Colon}}) = TensorField(domain(m)(i...), getindex(codomain(m),i...))
#Base.setindex!(m::GlobalSection{B,F,1,<:Interval},s::LocalSection,i::Vararg{Int}) where {B,F} = setindex!(codomain(m),fiber(s),i...)
#Base.setindex!(m::GlobalSection{B,F,N,<:RealRegion{V,T,N,<:AbstractRange} where {V,T}},s::LocalSection,i::Vararg{Int}) where {B,F,N} = setindex!(codomain(m),fiber(s),i...)
Base.setindex!(m::GlobalSection{B,Fm} where Fm,s::F,i::Vararg{Int}) where {B,F} = setindex!(codomain(m),s,i...)
function Base.setindex!(m::GlobalSection,s::LocalSection,i::Vararg{Int})
    setindex!(domain(m),base(s),i...)
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

Base.BroadcastStyle(::Type{<:TensorField{B,F,N,P}}) where {B,F,N,P} = Broadcast.ArrayStyle{TensorField{B,F,N,P}}()
Base.BroadcastStyle(::Type{<:GlobalSection{B,F,N,BA,FA}}) where {B,F,N,BA,FA} = Broadcast.ArrayStyle{TensorField{B,F,N,BA,FA}}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TensorField{B,F,N,P}}}, ::Type{ElType}) where {B,F,N,P,ElType}
    # Scan the inputs for the TensorField:
    t = find_tf(bc)
    # Use the domain field of t to create the output
    TensorField(domain(t), similar(Array{fibertype(ElType),N}, axes(bc)))
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GlobalSection{B,F,N,BA,FA}}}, ::Type{ElType}) where {B,F,N,BA,FA,ElType}
    # Scan the inputs for the TensorField:
    t = find_gs(bc)
    # Use the domain field of t to create the output
    GlobalSection(domain(t), similar(Array{fibertype(ElType),N}, axes(bc)))
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{<:AbstractFrameBundle{P,N}}}, ::Type{ElType}) where {N,P,ElType}
    t = find_gf(bc)
    # Use the data type to create the output
    TensorField(t,similar(Array{ElType,N}, axes(bc)))
end

"`A = find_tf(As)` returns the first TensorField among the arguments."
find_tf(bc::Base.Broadcast.Broadcasted) = find_tf(bc.args)
find_tf(bc::Base.Broadcast.Extruded) = find_tf(bc.x)
find_tf(args::Tuple) = find_tf(find_tf(args[1]), Base.tail(args))
find_tf(x) = x
find_tf(::Tuple{}) = nothing
find_tf(a::TensorField, rest) = a
find_tf(::Any, rest) = find_tf(rest)

"`A = find_gs(As)` returns the first GlobalSection among the arguments."
find_gs(bc::Base.Broadcast.Broadcasted) = find_gs(bc.args)
find_gs(bc::Base.Broadcast.Extruded) = find_gs(bc.x)
find_gs(args::Tuple) = find_gs(find_gs(args[1]), Base.tail(args))
find_gs(x) = x
find_gs(::Tuple{}) = nothing
find_gs(a::GlobalSection, rest) = a
find_gs(::Any, rest) = find_gs(rest)

(m::TensorField{B,F,N,<:SimplexFrameBundle} where {B,F,N})(i::ImmersedTopology) = TensorField(PointCloud(m)(i),fiber(m)[vertices(i)])
(m::TensorField{B,F,N,<:GridFrameBundle} where {B,F,N})(i::ImmersedTopology) = TensorField(base(m)(i),fiber(m))
for fun ∈ (:Open,:Mirror,:Clamped,:Torus,:Ribbon,:Wing,:Mobius,:Klein,:Cone,:Polar,:Sphere,:Geographic,:Hopf)
    for top ∈ (Symbol(fun,:Topology),)
        @eval begin
            $top(m::TensorField{B,F,N,<:GridFrameBundle} where {B,F,N}) = TensorField($top(base(m)),fiber(m))
            $top(m::GridFrameBundle) = m($top(size(m)))
            $top(p::PointArray) = TensorField(GridFrameBundle(p,$top(size(p))))
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

linterp(x,x1,x2,f1,f2) = f1 + (f2-f1)*(x-x1)/(x2-x1)
function bilinterp(x,y,x1,x2,y1,y2,f11,f21,f12,f22)
    f1 = linterp(x,x1,x2,f11,f21)
    f2 = linterp(x,x1,x2,f12,f22)
    linterp(y,y1,y2,f1,f2)
end
function trilinterp(x,y,z,x1,x2,y1,y2,z1,z2,f111,f211,f121,f221,f112,f212,f122,f222)
    f1 = bilinterp(x,y,x1,x2,y1,y2,f111,f211,f121,f221)
    f2 = bilinterp(x,y,x1,x2,y1,y2,f112,f212,f122,f222)
    linterp(z,z1,z2,f1,f2)
end
function quadlinterp(x,y,z,w,x1,x2,y1,y2,z1,z2,w1,w2,f1111,f2111,f1211,f2211,f1121,f2121,f1221,f2221,f1112,f2112,f1212,f2212,f1122,f2122,f1222,f2222)
    f1 = trilinterp(x,y,z,x1,x2,y1,y2,z1,z2,f1111,f2111,f1211,f2211,f1121,f2121,f1221,f2221)
    f2 = trilinterp(x,y,z,x1,x2,y1,y2,z1,z2,f1112,f2112,f1212,f2212,f1122,f2122,f1222,f2222)
    linterp(w,w1,w2,f1,f2)
end
function quintlinterp(x,y,z,w,v,x1,x2,y1,y2,z1,z2,w1,w2,v1,v2,f11111,f21111,f12111,f22111,f11211,f21211,f12211,f22211,f11121,f21121,f12121,f22121,f11221,f21221,f12221,f22221,f11112,f21112,f12112,f22112,f11212,f21212,f12212,f22212,f11122,f21122,f12122,f22122,f11222,f21222,f12222,f22222)
    f1 = quadlinterp(x,y,z,w,x1,x2,y1,y2,z1,z2,w1,w2,f11111,f21111,f12111,f22111,f11211,f21211,f12211,f22211,f11121,f21121,f12121,f22121,f11221,f21221,f12221,f22221)
    f2 = quadlinterp(x,y,z,w,x1,x2,y1,y2,z1,z2,w1,w2,f11112,f21112,f12112,f22112,f11212,f21212,f12212,f22212,f11122,f21122,f12122,f22122,f11222,f21222,f12222,f22222)
    linterp(v,v1,v2,f1,f2)
end

reposition_odd(p,x,t) = @inbounds (iseven(p) ? x[end]-x[1]+t : 2x[1]-t)
reposition_even(p,x,t) = @inbounds (isodd(p) ? x[1]-x[end]+t : 2x[end]-t)
@inline reposition(i1,i2,p1,p2,x,t) = i1 ? reposition_odd(p1,x,t) : i2 ? reposition_even(p2,x,t) : eltype(x)(t)

function searchpoints(p,t)
    i = searchsortedfirst(p,t)-1
    i01 = iszero(i)
    i01 && t==(@inbounds p[1]) ? (i+1,false) : (i,i01)
end

(m::TensorField)(s::Coordinate) = m(base(s))
(m::TensorField)(s::LocalTensor) = LocalTensor(base(s), m(fiber(s)))
(m::Grid{1})(t::Chain) = linterp(m,t)
(m::Grid{1})(t::AbstractFloat) = linterp(m,t)
(m::IntervalMap)(t::Chain) = linterp(m,t)
(m::IntervalMap)(t::AbstractFloat) = linterp(m,t)
function linterp(m,t)
    p,f,t1 = points(m),fiber(m),(@inbounds t[1])
    isnan(t1) && (return zero(fibertype(m))/0)
    i,i0 = searchpoints(p,t1)
    if !isopen(m)
        q = immersion(m)
        if iszero(i)
            if iszero(@inbounds q.r[1])
                return zero(fibertype(m))
            else
                return m(@inbounds reposition_odd(q.p[1],p,t1))
            end
        elseif i==length(p)
            if iszero(@inbounds q.r[2])
                return zero(fibertype(m))
            else
                return m(@inbounds reposition_even(q.p[2],p,t1))
            end
        end
    elseif iszero(i) || i==length(p)
        return zero(fibertype(m))
    end
    linterp(t1,p[i],p[i+1],f[i],f[i+1])
end
#=function (m::IntervalMap)(t::Vector,d=diff(m.cod)./diff(m.dom))
    [parametric(i,m,d) for i ∈ t]
end=#
function parametric(t,m,d=diff(codomain(m))./diff(domain(m)))
    p = points(m)
    i,i0 = searchpoints(p,t)
    codomain(m)[i]+(t-p[i])*d[i]
end

(m::RectangleMap)(t::Real) = leaf(m,t)
leaf(m::RectangleMap,i::Int,j::Int=2) = isone(j) ? m[i,:] : m[:,i]
function leaf(m::RectangleMap,t::AbstractFloat,j::Int=2)
    Q,p = isone(j),points(m).v[j]
    i,i0 = searchpoints(p,t)
    TensorField(points(m).v[Q ? 2 : 1],linterp(t,p[i],p[i+1],Q ? m.cod[i,:] : m.cod[:,i],Q ? m.cod[i+1,:] : m.cod[:,i+1]))
end

(m::HyperrectangleMap)(t::Real,j::Int=3) = leaf(m,t,j)
function leaf2(m::HyperrectangleMap,i::Int,j::Int,k::Int=3)
    isone(k) ? m[:,i,j] : k==2 ? m[i,:,j] : m[i,j,:]
end
function leaf(m::HyperrectangleMap,i::Int,j::Int=3)
    isone(j) ? m[i,:,:] : j==2 ? m[:,i,:] : m[:,:,i]
end

leaf(m::TensorField{B,F,2,<:FiberProductBundle} where {B,F},i::Int) = TensorField(base(base(m)),fiber(m)[:,i])
function (m::TensorField{B,F,2,<:FiberProductBundle} where {B,F})(t::Real)
    k = 2; p = base(m).cod.v[1]
    i,i0 = searchpoints(p,t)
    TensorField(base(base(m)),linterp(t,p[i],p[i+1],m.cod[:,i],m.cod[:,i+1]))
end

#(m::TensorField)(t::TensorField) = TensorField(base(t),m.(fiber(t)))
#(m::GridFrameBundle)(t::TensorField) = GridFrameBundle(PointArray(points(t),m.(fiber(t))),immersion(m))
(X::VectorField{B,F,N} where {B,F})(Y::VectorField{B,<:Chain{V,1,T,N} where {V,T},1} where B) where N = TensorField(base(Y),X.(fiber(Y)))
(m::GridFrameBundle{N})(t::VectorField{B,<:Chain{V,1,T,N} where {V,T},1} where B) where N = TensorField(GridFrameBundle(PointArray(points(t),m.(fiber(t))),immersion(t)),fiber(t))
#(m::GridFrameBundle{N})(t::VectorField{B,<:Chain{V,1,T,N} where {V,T},1} where B) where N = GridFrameBundle(PointArray(points(t),m.(fiber(t))),immersion(t))

(m::SimplexFrameBundle)(t::Chain) = sinterp(m,t)
(m::TensorField{B,F,N,<:SimplexFrameBundle} where {B,F,N})(t::Chain) = sinterp(m,t)
function sinterp(m,t::Chain{V}) where V
    j = findfirst(t,coordinates(m))
    iszero(j) && (return zero(fibertype(m)))
    i = immersion(m)[j]
    Chain{V}(fiber(m)[i])⋅(Chain{V}(points(m)[i])\t)
end

(m::Grid{2})(t::Chain) = bilinterp(m,t)
(m::Grid{2})(x::AbstractFloat,y::AbstractFloat) = bilinterp(m,Chain(x,y))
(m::TensorField{B,F,N,<:RealSpace{2}} where {B,F,N})(t::Chain) = bilinterp(m,t)
(m::TensorField{B,F,N,<:RealSpace{2}} where {B,F,N})(x,y) = bilinterp(m,Chain(x,y))
function bilinterp(m,t::Chain{V,G,T,2} where {G,T}) where V
    x,y,f,t1,t2 = @inbounds (points(m).v[1],points(m).v[2],fiber(m),t[1],t[2])
    (isnan(t1) || isnan(t2)) && (return zero(fibertype(m))/0)
    (i,i01),(j,j01) = searchpoints(x,t1),searchpoints(y,t2)
    if !isopen(m)
        q = immersion(m)
        i02,iq1,iq2,j02,jq1,jq2 = @inbounds (
            i==length(x),iszero(q.r[1]),iszero(q.r[2]),
            j==length(y),iszero(q.r[3]),iszero(q.r[4]))
        if (i01 && iq1) || (i02 && iq2) || (j01 && jq1) || (j02 && jq2)
            return zero(fibertype(m))
        else
            i1,i2,j1,j2 = (
                i01 && !iq1,i02 && !iq2,
                j01 && !jq1,j02 && !jq2)
            if i1 || i2 || j1 || j2
                return m(Chain{V}(
                    (@inbounds reposition(i1,i2,q.p[1],q.p[2],x,t1)),
                    (@inbounds reposition(j1,j2,q.p[3],q.p[4],y,t2))))
            end
        end
    elseif i01 || j01 || i==length(x) || j==length(y)
        # elseif condition creates 1 allocation, as opposed to 0 ???
        return zero(fibertype(m))
    end
    #f1 = linterp(t[1],x[i],x[i+1],f[i,j],f[i+1,j])
    #f2 = linterp(t[1],x[i],x[i+1],f[i,j+1],f[i+1,j+1])
    #linterp(t[2],y[j],y[j+1],f1,f2)
    bilinterp(t1,t2,x[i],x[i+1],y[j],y[j+1],
        f[i,j],f[i+1,j],f[i,j+1],f[i+1,j+1])
end

(m::Grid{3})(t::Chain) = trilinterp(m,t)
(m::Grid{3})(x::AbstractFloat,y::AbstractFloat,z::AbstractFloat) = trilinterp(m,Chain(x,y,z))
(m::TensorField{B,F,N,<:RealSpace{3}} where {B,F,N})(t::Chain) = trilinterp(m,t)
(m::TensorField{B,F,N,<:RealSpace{3}} where {B,F,N})(x,y,z) = trilinterp(m,Chain(x,y,z))
function trilinterp(m,t::Chain{V,G,T,3} where {G,T}) where V
    x,y,z,f,t1,t2,t3 = @inbounds (points(m).v[1],points(m).v[2],points(m).v[3],fiber(m),t[1],t[2],t[3])
    (isnan(t1) || isnan(t2) || isnan(t3)) && (return zero(fibertype(m))/0)
    (i,i01),(j,j01),(k,k01) = (searchpoints(x,t1),searchpoints(y,t2),searchpoints(z,t3))
    if !isopen(m)
        q = immersion(m)
        i02,iq1,iq2,j02,jq1,jq2,k02,kq1,kq2 = @inbounds (
            i==length(x),iszero(q.r[1]),iszero(q.r[2]),
            j==length(y),iszero(q.r[3]),iszero(q.r[4]),
            k==length(z),iszero(q.r[5]),iszero(q.r[6]))
        if (i01 && iq1) || (i02 && iq2) || (j01 && jq1) || (j02 && jq2) || (k01 && kq1) || (k02 && kq2)
            return zero(fibertype(m))
        else
            i1,i2,j1,j2,k1,k2 = (
                i01 && !iq1,i02 && !iq2,
                j01 && !jq1,j02 && !jq2,
                k01 && !kq1,k02 && !kq2)
            if i1 || i2 || j1 || j2 || k1 || k2
                return m(Chain{V}(
                    (@inbounds reposition(i1,i2,q.p[1],q.p[2],x,t1)),
                    (@inbounds reposition(j1,j2,q.p[3],q.p[4],y,t2)),
                    (@inbounds reposition(k1,k2,q.p[5],q.p[6],z,t3))))
            end
        end
    elseif i01 || j01 || k01 || i==length(x) || j==length(y) || k==length(z)
        # elseif condition creates 1 allocation, as opposed to 0 ???
        return zero(fibertype(m))
    end
    #f1 = linterp(t[1],x[i],x[i+1],f[i,j,k],f[i+1,j,k])
    #f2 = linterp(t[1],x[i],x[i+1],f[i,j+1,k],f[i+1,j+1,k])
    #g1 = linterp(t[2],y[j],y[j+1],f1,f2)
    #f3 = linterp(t[1],x[i],x[i+1],f[i,j,k+1],f[i+1,j,k+1])
    #f4 = linterp(t[1],x[i],x[i+1],f[i,j+1,k+1],f[i+1,j+1,k+1])
    #g2 = linterp(t[2],y[j],y[j+1],f3,f4)
    #linterp(t[3],z[k],z[k+1],g1,g2)
    trilinterp(t1,t2,t3,x[i],x[i+1],y[j],y[j+1],z[k],z[k+1],
        f[i,j,k],f[i+1,j,k],f[i,j+1,k],f[i+1,j+1,k],
        f[i,j,k+1],f[i+1,j,k+1],f[i,j+1,k+1],f[i+1,j+1,k+1])
end

(m::Grid{4})(t::Chain) = quadlinterp(m,t)
(m::Grid{4})(x::AbstractFloat,y::AbstractFloat,z::AbstractFloat,w::AbstractFloat) = quadlinterp(m,Chain(x,y,z,w))
(m::TensorField{B,F,N,<:RealSpace{4}} where {B,F,N})(t::Chain) = quadlinterp(m,t)
(m::TensorField{B,F,N,<:RealSpace{4}} where {B,F,N})(x,y,z,w) = m(Chain(x,y,z,w))
function (m)(t::Chain{V,G,T,4} where {G,T}) where V
    x,y,z,w,f,t1,t2,t3,t4 = @inbounds (points(m).v[1],points(m).v[2],points(m).v[3],points(m).v[4],fiber(m),t[1],t[2],t[3],t[4])
    (isnan(t1) || isnan(t2) || isnan(t3) ||isnan(t4)) && (return zero(fibertype(m))/0)
    (i,i01),(j,j01),(k,k01),(l,l01) = (searchpoints(x,t1),searchpoints(y,t2),searchpoints(z,t3),searchpoints(w,t4))
    if !isopen(m)
        q = immersion(m)
        i02,iq1,iq2,j02,jq1,jq2,k02,kq1,kq2,l02,lq1,lq2 = @inbounds (
            i==length(x),iszero(q.r[1]),iszero(q.r[2]),
            j==length(y),iszero(q.r[3]),iszero(q.r[4]),
            k==length(z),iszero(q.r[5]),iszero(q.r[6]),
            l==length(w),iszero(q.r[7]),iszero(q.r[8]))
        if (i01 && iq1) || (i02 && iq2) || (j01 && jq1) || (j02 && jq2) || (k01 && kq1) || (k02 && kq2) || (l01 && lq1) || (l02 && lq2)
            return zero(fibertype(m))
        else
            i1,i2,j1,j2,k1,k2,l1,l2 = (
                i01 && !iq1,i02 && !iq2,
                j01 && !jq1,j02 && !jq2,
                k01 && !kq1,k02 && !kq2,
                l01 && !lq1,l02 && !lq2)
            if i1 || i2 || j1 || j2 || k1 || k2 || l1 || l2
                return m(Chain{V}(
                    (@inbounds reposition(i1,i2,q.p[1],q.p[2],x,t1)),
                    (@inbounds reposition(j1,j2,q.p[3],q.p[4],y,t2)),
                    (@inbounds reposition(k1,k2,q.p[5],q.p[6],z,t3)),
                    (@inbounds reposition(l1,l2,q.p[7],q.p[8],w,t4))))
            end
        end
    elseif i01 || j01 || k01 || l01 || i==length(x) || j==length(y) || k==length(z) || l==length(w)
        # elseif condition creates 1 allocation, as opposed to 0 ???
        return zero(fibertype(m))
    end
    #f1 = linterp(t[1],x[i],x[i+1],f[i,j,k,l],f[i+1,j,k,l])
    #f2 = linterp(t[1],x[i],x[i+1],f[i,j+1,k,l],f[i+1,j+1,k,l])
    #g1 = linterp(t[2],y[j],y[j+1],f1,f2)
    #f3 = linterp(t[1],x[i],x[i+1],f[i,j,k+1,l],f[i+1,j,k+1,l])
    #f4 = linterp(t[1],x[i],x[i+1],f[i,j+1,k+1,l],f[i+1,j+1,k+1,l])
    #g2 = linterp(t[2],y[j],y[j+1],f3,f4)
    #h1 = linterp(t[3],z[k],z[k+1],g1,g2)
    #f5 = linterp(t[1],x[i],x[i+1],f[i,j,k,l+1],f[i+1,j,k,l+1])
    #f6 = linterp(t[1],x[i],x[i+1],f[i,j+1,k,l+1],f[i+1,j+1,k,l+1])
    #g3 = linterp(t[2],y[j],y[j+1],f5,f6)
    #f7 = linterp(t[1],x[i],x[i+1],f[i,j,k+1,l+1],f[i+1,j,k+1,l+1])
    #f8 = linterp(t[1],x[i],x[i+1],f[i,j+1,k+1,l+1],f[i+1,j+1,k+1,l+1])
    #g4 = linterp(t[2],y[j],y[j+1],f7,f8)
    #h2 = linterp(t[3],z[k],z[k+1],g3,g4)
    #linterp(t[4],w[l],w[l+1],h1,h2)
    quadlinterp(t1,t2,t3,t4,x[i],x[i+1],y[j],y[j+1],z[k],z[k+1],w[l],w[l+1],
        f[i,j,k,l],f[i+1,j,k,l],f[i,j+1,k,l],f[i+1,j+1,k,l],
        f[i,j,k+1,l],f[i+1,j,k+1,l],f[i,j+1,k+1,l],f[i+1,j+1,k+1,l],
        f[i,j,k,l+1],f[i+1,j,k,l+1],f[i,j+1,k,l+1],f[i+1,j+1,k,l+1],
        f[i,j,k+1,l+1],f[i+1,j,k+1,l+1],f[i,j+1,k+1,l+1],f[i+1,j+1,k+1,l+1])
end

(m::Grid{5})(t::Chain) = quintlinterp(m,t)
(m::Grid{5})(x::AbstractFloat,y::AbstractFloat,z::AbstractFloat,w::AbstractFloat,v::AbstractFloat) = quintlinterp(m,Chain(x,y,z,w,v))
(m::TensorField{B,F,N,<:RealSpace{5}} where {B,F,N})(t::Chain) = quintlinterp(m,t)
(m::TensorField{B,F,N,<:RealSpace{5}} where {B,F,N})(x,y,z,w,v) = m(Chain(x,y,z,w,v))
function quintlinterp(m,t::Chain{V,G,T,5} where {G,T}) where V
    x,y,z,w,v,f,t1,t2,t3,t4,t5 = @inbounds (points(m).v[1],points(m).v[2],points(m).v[3],points(m).v[4],points(m).v[5],fiber(m),t[1],t[2],t[3],t[4],t[5])
    (isnan(t1) || isnan(t2) || isnan(t3) || isnan(t4) || isnan(t5)) && (return zero(fibertype(m))/0)
    (i,i01),(j,j01),(k,k01),(l,l01),(o,o01) = (searchpoints(x,t1),searchpoints(y,t2),searchpoints(z,t3),searchpoints(w,t4),searchpoints(v,t5))
    if !isopen(m)
        q = immersion(m)
        i02,iq1,iq2,j02,jq1,jq2,k02,kq1,kq2,l02,lq1,lq2,o02,oq1,oq2 = @inbounds (
            i==length(x),iszero(q.r[1]),iszero(q.r[2]),
            j==length(y),iszero(q.r[3]),iszero(q.r[4]),
            k==length(z),iszero(q.r[5]),iszero(q.r[6]),
            l==length(w),iszero(q.r[7]),iszero(q.r[8]),
            o==length(v),iszero(q.r[9]),iszero(q.r[10]))
        if (i01 && iq1) || (i02 && iq2) || (j01 && jq1) || (j02 && jq2) || (k01 && kq1) || (k02 && kq2) || (l01 && lq1) || (l02 && lq2) || (o01 && oq1) || (o02 && oq2)
            return zero(fibertype(m))
        else
            i1,i2,j1,j2,k1,k2,l1,l2,o1,o2 = (
                i01 && !iq1,i02 && !iq2,
                j01 && !jq1,j02 && !jq2,
                k01 && !kq1,k02 && !kq2,
                l01 && !lq1,l02 && !lq2,
                o01 && !oq1,o02 && !oq2)
            if i1 || i2 || j1 || j2 || k1 || k2 || l1 || l2 || o1 || o2
                return m(Chain{V}(
                    (@inbounds reposition(i1,i2,q.p[1],q.p[2],x,t1)),
                    (@inbounds reposition(j1,j2,q.p[3],q.p[4],y,t2)),
                    (@inbounds reposition(k1,k2,q.p[5],q.p[6],z,t3)),
                    (@inbounds reposition(l1,l2,q.p[7],q.p[8],w,t4)),
                    (@inbounds reposition(o1,o2,q.p[9],q.p[10],v,t5))))
            end
        end
    elseif i01 || j01 || k01 || l01 || o01 || i==length(x) || j==length(y) || k==length(z) || l==length(w) || o==length(v)
        # elseif condition creates 1 allocation, as opposed to 0 ???
        return zero(fibertype(m))
    end
    quintlinterp(t1,t2,t3,t4,t5,x[i],x[i+1],y[j],y[j+1],z[k],z[k+1],w[l],w[l+1],v[o],v[o+1],
        f[i,j,k,l,o],f[i+1,j,k,l,o],f[i,j+1,k,l,o],f[i+1,j+1,k,l,o],
        f[i,j,k+1,l,o],f[i+1,j,k+1,l,o],f[i,j+1,k+1,l,o],f[i+1,j+1,k+1,l,o],
        f[i,j,k,l+1,o],f[i+1,j,k,l+1,o],f[i,j+1,k,l+1,o],f[i+1,j+1,k,l+1,o],
        f[i,j,k+1,l+1,o],f[i+1,j,k+1,l+1,o],f[i,j+1,k+1,l+1,o],f[i+1,j+1,k+1,l+1,o],
        f[i,j,k,l,o+1],f[i+1,j,k,l,o+1],f[i,j+1,k,l,o+1],f[i+1,j+1,k,l,o+1],
        f[i,j,k+1,l,o+1],f[i+1,j,k+1,l,o+1],f[i,j+1,k+1,l,o+1],f[i+1,j+1,k+1,l,o+1],
        f[i,j,k,l+1,o+1],f[i+1,j,k,l+1,o+1],f[i,j+1,k,l+1,o+1],f[i+1,j+1,k,l+1,o+1],
        f[i,j,k+1,l+1,o+1],f[i+1,j,k+1,l+1,o+1],f[i,j+1,k+1,l+1,o+1],f[i+1,j+1,k+1,l+1,o+1])
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
Base.:^(t::TensorField,n::Int) = TensorField(domain(t), .^(codomain(t),n,ref(metricextensor(base(t)))))
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
for type ∈ (:TensorField,)
    for (op,mop) ∈ ((:*,:wedgedot_metric),(:wedgedot,:wedgedot_metric),(:veedot,:veedot_metric),(:⋅,:contraction_metric),(:contraction,:contraction_metric),(:>,:contraction_metric),(:⊘,:⊘),(:>>>,:>>>),(:/,:/),(:^,:^))
        let bop = op ∈ (:*,:>,:>>>,:/,:^) ? :(Base.$op) : :(Grassmann.$op)
        @eval begin
            $bop(a::$type{R},b::$type{R}) where R = $type(base(a),Grassmann.$mop.(fiber(a),fiber(b),refmetric(base(a))))
            $bop(a::Number,b::$type) = $type(base(b), Grassmann.$op.(a,fiber(b)))
            $bop(a::$type,b::Number) = $type(base(a), Grassmann.$op.(fiber(a),b,$((op≠:^ ? () : (:(refmetric(base(a))),))...)))
        end end
    end
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

Variation(dom,cod::TensorField) = TensorField(dom,cod.(dom))
function Variation(cod::TensorField); p = points(cod).v[end]
    TensorField(p,leaf.(Ref(cod),1:length(p)))
end
function Variation(cod::TensorField{B,F,2,<:FiberProductBundle} where {B,F})
    p = base(cod).cod.v[1]
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

include("diffgeo.jl")
include("constants.jl")
include("element.jl")

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
    display(fun(v.cod[1],args...))
    for i ∈ 2:length(v)
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
    display(fun(leaf(v,1),args...))
    for i ∈ 2:length(points(v).v[end])
        fun!(leaf(v,i),args...)
        sleep(t)
    end
end
function variation(v::TensorField{B,F,2,<:FiberProductBundle} where {B,F},t,fun::Function,args...)
    for i ∈ 1:length(base(v).cod.v[1])
        display(fun(leaf(v,i),args...))
        sleep(t)
    end
end
function variation(v::TensorField{B,F,2,<:FiberProductBundle} where {B,F},t,fun::Function,fun!::Function,args...)
    display(fun(leaf(v,1),args...))
    for i ∈ 2:length(base(v).cod.v[1])
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
    display(fun(leaf(v,1,1),args...))
    for i ∈ 2:length(points(v).v[1])
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
    display(fun(leaf(v,1,2),args...))
    for i ∈ 2:length(points(v).v[2])
        fun!(leaf(v,i,2),args...)
        sleep(t)
    end
end

function __init__()
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        export linegraph, linegraph!
        funsym(sym) = String(sym)[end] == '!' ? sym : Symbol(sym,:!)
        for lines ∈ (:lines,:lines!,:linesegments,:linesegments!)
            @eval begin
                Makie.$lines(t::ScalarMap;args...) = Makie.$lines(getindex.(domain(t),2),codomain(t);args...)
                Makie.$lines(t::SpaceCurve,f::Function=speed;args...) = Makie.$lines(codomain(t);color=Real.(codomain(f(t))),args...)
                Makie.$lines(t::PlaneCurve,f::Function=speed;args...) = Makie.$lines(codomain(t);color=Real.(codomain(f(t))),args...)
                Makie.$lines(t::RealFunction,f::Function=speed;args...) = Makie.$lines(Real.(points(t)),Real.(codomain(t));color=Real.(codomain(f(t))),args...)
                Makie.$lines(t::ComplexMap{B,F,1},f::Function=speed;args...) where {B<:Coordinate{<:AbstractReal},F} = Makie.$lines(realvalue.(codomain(t)),imagvalue.(codomain(t));color=Real.(codomain(f(t))),args...)
            end
        end
        #Makie.lines(t::TensorField{B,F,1};args...) where {B<:Coordinate{<:AbstractReal},F<:AbstractReal} = linegraph(t;args...)
        #Makie.lines!(t::TensorField{B,F,1};args...) where {B<:Coordinate{<:AbstractReal},F<:AbstractReal} = linegraph(t;args...)
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
        function Makie.arrows(M::VectorField,t::TensorField{B,<:TensorOperator,N,<:GridFrameBundle} where B;args...) where N
            Makie.arrows(TensorField(fiber(M),fiber(t));args...)
        end
        function Makie.arrows!(M::VectorField,t::TensorField{B,<:TensorOperator,N,<:GridFrameBundle} where B;args...) where N
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
                function Makie.$fun(t::TensorField{<:Coordinate{<:Chain},<:TensorOperator,N,<:GridFrameBundle};args...) where N
                    display(Makie.$fun(getindex.(t,1);args...))
                    for i ∈ 2:mdims(eltype(codomain(t)))
                        Makie.$fun!(getindex.(t,i);args...)
                    end
                end
                function Makie.$fun!(t::TensorField{<:Coordinate{<:Chain},<:TensorOperator,N,<:GridFrameBundle};args...) where N
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
                function Makie.$fun(t::GradedField{G,B,F,2,<:RealSpace{2}} where G;args...) where {B,F}
                    x,y = points(t),value.(codomain(t))
                    display(Makie.$fun(x.v...,Real.(getindex.(y,1));args...))
                    for i ∈ 2:binomial(mdims(eltype(codomain(t))),G)
                        Makie.$(funsym(fun))(x.v...,Real.(getindex.(y,i));args...)
                    end
                end
            end
        end
        Makie.wireframe(t::SurfaceGrid;args...) = Makie.wireframe(graph(t);args...)
        Makie.wireframe!(t::SurfaceGrid;args...) = Makie.wireframe!(graph(t);args...)
        point2chain(x,V=Submanifold(2)) = Chain(x[1],x[2])
        chain3vec(x) = Makie.Vec3(x[1],x[2],x[3])
        for fun ∈ (:streamplot,:streamplot!)
            @eval begin
                Makie.$fun(f::Function,t::Rectangle;args...) = Makie.$fun(f,t.v...;args...)
                Makie.$fun(f::Function,t::Hyperrectangle;args...) = Makie.$fun(f,t.v...;args...)
                Makie.$fun(m::ScalarField{<:Coordinate{<:Chain},<:AbstractReal,N,<:RealSpace} where N;args...) = Makie.$fun(gradient_fast(m);args...)
                Makie.$fun(m::ScalarMap,dims...;args...) = Makie.$fun(gradient_fast(m),dims...;args...)
                Makie.$fun(m::VectorField{R,F,1,<:SimplexFrameBundle} where {R,F},dims...;args...) = Makie.$fun(p->Makie.Point(m(Chain(one(eltype(p)),p.data...))),dims...;args...)
                Makie.$fun(m::VectorField{<:Coordinate{<:Chain},F,N,<:RealSpace} where {F,N};args...) = Makie.$fun(p->Makie.Point(m(Chain(p.data...))),points(m).v...;args...)
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
                Makie.$fun(t::ScalarField{<:Coordinate{<:Chain},F,2,<:RealSpace{2}} where F;args...) = Makie.$fun(Makie.Point.(fiber(graph(Real(t))))[:],Makie.Point.(fiber(normal(Real(t))))[:];args...)
                Makie.$fun(t::VectorField{<:Coordinate{<:Chain{W,L,F,2} where {W,L,F}},<:Chain{V,G,T,2} where {V,G,T},2,<:AlignedRegion{2}};args...) = Makie.$fun(points(t).v...,getindex.(codomain(t),1),getindex.(codomain(t),2);args...)
                Makie.$fun(t::VectorField{<:Coordinate{<:Chain},<:Chain,2,<:RealSpace{2}};args...) = Makie.$fun(Makie.Point.(points(t))[:],Makie.Point.(codomain(t))[:];args...)
                Makie.$fun(t::VectorField{<:Coordinate{<:Chain},F,N,<:GridFrameBundle} where {F,N};args...) = Makie.$fun(Makie.Point.(points(t))[:],Makie.Point.(codomain(t))[:];args...)
                Makie.$fun(t::VectorField,f::VectorField;args...) = Makie.$fun(Makie.Point.(fiber(t))[:],Makie.Point.(fiber(f))[:];args...)
                Makie.$fun(t::Rectangle,f::Function;args...) = Makie.$fun(t.v...,f;args...)
                Makie.$fun(t::Hyperrectangle,f::Function;args...) = Makie.$fun(t.v...,f;args...)
                Makie.$fun(t::ScalarMap;args...) = Makie.$fun(points(points(t)),Real.(codomain(t));args...)
            end
        end
        Makie.mesh(t::ScalarMap;args...) = Makie.mesh(domain(t);color=Real.(codomain(t)),args...)
        Makie.mesh!(t::ScalarMap;args...) = Makie.mesh!(domain(t);color=Real.(codomain(t)),args...)
        #Makie.wireframe(t::ElementFunction;args...) = Makie.wireframe(value(domain(t));color=Real.(codomain(t)),args...)
        #Makie.wireframe!(t::ElementFunction;args...) = Makie.wireframe!(value(domain(t));color=Real.(codomain(t)),args...)
        Makie.convert_arguments(P::Makie.PointBased, a::SimplexFrameBundle) = Makie.convert_arguments(P, points(a))
        Makie.convert_single_argument(a::LocalFiber) = convert_arguments(P,Point(a))
        Makie.arrows(p::SimplexFrameBundle,v;args...) = Makie.arrows(GeometryBasics.Point.(↓(V).(points(p))),GeometryBasics.Point.(value(v));args...)
        Makie.arrows!(p::SimplexFrameBundle,v;args...) = Makie.arrows!(GeometryBasics.Point.(↓(V).(points(p))),GeometryBasics.Point.(value(v));args...)
        Makie.scatter(p::SimplexFrameBundle,x;args...) = Makie.scatter(submesh(p)[:,1],x;args...)
        Makie.scatter!(p::SimplexFrameBundle,x;args...) = Makie.scatter!(submesh(p)[:,1],x;args...)
        Makie.scatter(p::SimplexFrameBundle;args...) = Makie.scatter(submesh(p);args...)
        Makie.scatter!(p::SimplexFrameBundle;args...) = Makie.scatter!(submesh(p);args...)
        Makie.lines(p::SimplexFrameBundle;args...) = Makie.lines(points(p);args...)
        Makie.lines!(p::SimplexFrameBundle;args...) = Makie.lines!(points(p);args...)
        #Makie.lines(p::Vector{<:TensorAlgebra};args...) = Makie.lines(GeometryBasics.Point.(p);args...)
        #Makie.lines!(p::Vector{<:TensorAlgebra};args...) = Makie.lines!(GeometryBasics.Point.(p);args...)
        #Makie.lines(p::Vector{<:TensorTerm};args...) = Makie.lines(value.(p);args...)
        #Makie.lines!(p::Vector{<:TensorTerm};args...) = Makie.lines!(value.(p);args...)
        #Makie.lines(p::Vector{<:Chain{V,G,T,1} where {V,G,T}};args...) = Makie.lines(getindex.(p,1);args...)
        #Makie.lines!(p::Vector{<:Chain{V,G,T,1} where {V,G,T}};args...) = Makie.lines!(getindex.(p,1);args...)
        function Makie.linesegments(e::SimplexFrameBundle;args...)
            mdims(immersion(e)) ≠ 2 && (return Makie.linesegments(PointCloud(e)(edges(e))))
            p=points(e)
            Makie.linesegments(Grassmann.pointpair.(e[ImmersedTopology(e)],↓(Manifold(p)));args...)
        end
        function Makie.linesegments!(e::SimplexFrameBundle;args...)
            mdims(immersion(e)) ≠ 2 && (return Makie.linesegments!(PointCloud(e)(edges(e))))
            p=points(e)
            Makie.linesegments!(Grassmann.pointpair.(e[ImmersedTopology(e)],↓(Manifold(p)));args...)
        end
        Makie.wireframe(t::SimplexFrameBundle;args...) = Makie.linesegments(t(edges(t));args...)
        Makie.wireframe!(t::SimplexFrameBundle;args...) = Makie.linesegments!(t(edges(t));args...)
        for fun ∈ (:mesh,:mesh!,:wireframe,:wireframe!)
            @eval Makie.$fun(M::GridFrameBundle;args...) = Makie.$fun(GeometryBasics.Mesh(M);args...)
        end
        function linegraph(M::TensorField{B,<:Chain,2,<:GridFrameBundle} where B,f::Function=speed;args...)
            variation(M,Makie.lines,Makie.lines!,f;args...)
            alteration(M,Makie.lines!,Makie.lines!,f;args...)
        end
        function linegraph!(M::TensorField{B,<:Chain,2,<:GridFrameBundle} where B,f::Function=speed;args...)
            variation(M,Makie.lines!,Makie.lines!,f;args...)
            alteration(M,Makie.lines!,Makie.lines!,f;args...)
        end
        function linegraph(v::TensorField{B,<:Chain,3,<:GridFrameBundle} where B,f::Function=speed;args...)
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
        function Makie.mesh(M::TensorField{B,<:Chain,2,<:GridFrameBundle} where B;args...)
            Makie.mesh(GridFrameBundle(fiber(M));args...)
        end
        function Makie.mesh!(M::TensorField{B,<:Chain,2,<:GridFrameBundle} where B;args...)
            Makie.mesh!(GridFrameBundle(fiber(M));args...)
        end
        function Makie.mesh(M::TensorField{B,<:AbstractReal,2,<:GridFrameBundle} where B;args...)
            Makie.mesh(GeometryBasics.Mesh(base(M));color=fiber(Real(M))[:],args...)
        end
        function Makie.mesh!(M::TensorField{B,<:AbstractReal,2,<:GridFrameBundle} where B;args...)
            Makie.mesh!(GeometryBasics.Mesh(base(M));color=fiber(Real(M))[:],args...)
        end
        function Makie.mesh(M::TensorField{B,<:Chain,2,<:GridFrameBundle} where B,f::TensorField;args...)
            Makie.mesh(GridFrameBundle(fiber(M));color=fiber(Real(f))[:],args...)
        end
        function Makie.mesh!(M::TensorField{B,<:Chain,2,<:GridFrameBundle} where B,f::TensorField;args...)
            Makie.mesh!(GridFrameBundle(fiber(M));color=fiber(Real(f))[:],args...)
        end
        Makie.wireframe(M::TensorField{B,<:Chain,N,<:GridFrameBundle} where {B,N};args...) = Makie.wireframe(GridFrameBundle(fiber(M));args...)
        Makie.wireframe!(M::TensorField{B,<:Chain,N,<:GridFrameBundle} where {B,N};args...) = Makie.wireframe!(GridFrameBundle(fiber(M));args...)
        function Makie.mesh(M::SimplexFrameBundle;args...)
            if mdims(points(M)) == 2
                sm = submesh(M)[:,1]
                Makie.lines(sm,args[:color])
                Makie.plot!(sm,args[:color])
            else
                Makie.mesh(submesh(M),array(ImmersedTopology(M));args...)
            end
        end
        function Makie.mesh!(M::SimplexFrameBundle;args...)
            if mdims(points(M)) == 2
                sm = submesh(M)[:,1]
                Makie.lines!(sm,args[:color])
                Makie.plot!(sm,args[:color])
            else
                Makie.mesh!(submesh(M),array(ImmersedTopology(M));args...)
            end
        end
    end
    @require UnicodePlots="b8865327-cd53-5732-bb35-84acbb429228" begin
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
        function initmesh(m::Meshes.SimpleMesh{N}) where N
            c,f = Meshes.vertices(m),m.topology.connec
            s = N+1; V = Submanifold(ℝ^s) # s
            n = length(f[1].indices)
            p = PointCloud([Chain{V,1}(Values{s,Float64}(1.0,k.coords...)) for k ∈ c])
            t = SimplexTopology([Values{n,Int}(k.indices) for k ∈ f])
            return (p,∂(t),t)
        end
    end
    @require GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326" begin
        Base.convert(::Type{GeometryBasics.Point},t::T) where T<:LocalFiber = GeometryBasics.Point(base(t))
        #GeometryBasics.Point(t::T) where T<:TensorAlgebra = convert(GeometryBasics.Point,t)
        function GeometryBasics.Mesh(m::GridFrameBundle{2})
            nm = size(points(m))
            faces = GeometryBasics.Tesselation(GeometryBasics.Rect(0, 0, 1, 1), nm)
            uv = Chain(0.0,0.0):map(inv,Chain((nm.-1)...)):Chain(1.0,1.0)
            GeometryBasics.Mesh(GeometryBasics.meta(GeometryBasics.Point.(points(m)[:]); uv=GeometryBasics.Vec{2}.(value.(uv[:]))), GeometryBasics.decompose(GeometryBasics.QuadFace{GeometryBasics.GLIndex}, faces))
        end
        function initmesh(m::GeometryBasics.Mesh)
            c,f = GeometryBasics.coordinates(m),GeometryBasics.faces(m)
            s = size(eltype(c))[1]+1; V = varmanifold(s) # s
            n = size(eltype(f))[1]
            p = [Chain{V,1}(Values{s,Float64}(1.0,k...)) for k ∈ c]
            t = SimplexTopology([Values{n,Int}(k) for k ∈ f])
            return (p,∂(t),t)
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
            p(SimplexTopology([Values(getindex.(Ref(n),k)) for k ∈ T.simplices]))
        end
        function initmesh(t::Chull)
            p = PointCloud(initpoints(t.points'))
            p(SimplexTopology(Values.(t.simplices)))
        end
    end
    @require MiniQhull="978d7f02-9e05-4691-894f-ae31a51d76ca" begin
        MiniQhull.delaunay(p::Vector{<:Chain},args...) = MiniQhull.delaunay(PointCloud(p),1:length(p),args...)
        MiniQhull.delaunay(p::Vector{<:Chain},n::AbstractVector,args...) = MiniQhull.delaunay(PointCloud(p),n,args...)
        MiniQhull.delaunay(p::PointCloud,args...) = MiniQhull.delaunay(p,1:length(p),args...)
        function MiniQhull.delaunay(p::PointCloud,n::AbstractVector,args...)
            N,m = mdims(p),length(n)
            l = list(1,N)
            T = MiniQhull.delaunay(Matrix(submesh(m==length(p) ? p : p[n])'),args...)
            SimplexTopology([Values{N,Int}(getindex.(Ref(n),Int.(T[l,k]))) for k ∈ 1:size(T,2)],m)
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
            B = p.id
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
        function Triangulate.TriangulateIO(e::SimplexFrameBundle,h=nothing)
            triin=Triangulate.TriangulateIO()
            triin.pointlist=triangle(points(e))
            triin.segmentlist=triangle(immersion(e))
            !isnothing(h) && (triin.holelist=triangle(h))
            return triin
        end
        function Triangulate.triangulate(i,e::SimplexFrameBundle;holes=nothing)
            initmesh(Triangulate.triangulate(i,Triangulate.TriangulateIO(e,holes))[1])
        end
        initmesh(t::Triangulate.TriangulateIO) = initmeshdata(t.pointlist,t.segmentlist,t.trianglelist,Val(2))
        #aran(area=0.001,angle=20) = "pa$(Printf.@sprintf("%.15f",area))q$(Printf.@sprintf("%.15f",angle))Q"
    end
    @require TetGen="c5d3f3f7-f850-59f6-8a2e-ffc6dc1317ea" begin
        function TetGen.JLTetGenIO(mesh::SimplexFrameBundle;
                marker = :markers, holes = TetGen.Point{3, Float64}[])
            f = TetGen.TriangleFace{Cint}.(immersion(mesh))
            kw_args = Any[:facets => TetGen.metafree(f),:holes => holes]
            if hasproperty(f, marker)
                push!(kw_args, :facetmarkers => getproperty(f, marker))
            end
            pm = points(mesh); V = Manifold(pm)
            TetGen.JLTetGenIO(TetGen.Point.(↓(V).(pm)); kw_args...)
        end
        function initmesh(tio::TetGen.JLTetGenIO, command = "Qp")
            r = TetGen.tetrahedralize(tio, command); V = Submanifold(ℝ^4)
            p = [Chain{V,1}(Values{4,Float64}(1.0,k...)) for k ∈ r.points]
            t = Values{4,Int}.(r.tetrahedra)
            e = Values{3,Int}.(r.trifaces) # Values{2,Int}.(r.edges)
            return PointCloud(p),SimplexTopology(e),SimplexTopology(t)
        end
        function TetGen.tetrahedralize(mesh::SimplexFrameBundle, command = "Qp";
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
        function matlab(p::SimplexFrameBundle)
            B = bundle(p)
            if length(matlab_cache)<B || isempty(matlab_cache[B])
                ap = array(p)'
                matlab(islocal(p) ? vcat(ap,ones(length(p))') : ap[2:end,:],B)
            else
                return matlab_cache[B]
            end
        end
        function matlab(p::SimplexTopology)
            B = bundle(p)
            if length(matlab_top_cache)<B || isempty(matlab_top_cache[B])
                ap = array(p)'
                matlab_top(islocal(p) ? vcat(ap,ones(length(p))') : ap[2:end,:],B)
            else
                return matlab_top_cache[B]
            end
        end
        initmesh(g,args...) = initmeshall(g,args...)[list(1,3)]
        initmeshall(g::Matrix{Int},args...) = initmeshall(Matrix{Float64}(g),args...)
        function initmeshall(g,args...)
            P,E,T = MATLAB.mxcall(:initmesh,3,g,args...)
            p,e,t = initmeshdata(P,E,T,Val(2))
            return (p,e,t,T,E,P)
        end
        function initmeshes(g,args...)
            p,e,t,T = initmeshall(g,args...)
            p,e,t,[Int(T[end,k]) for k ∈ 1:size(T,2)]
        end
        export initmeshes
        function refinemesh(g,args...)
            p,e,t,T,E,P = initmeshall(g,args...)
            matlab(P,bundle(p)); matlab(E,bundle(e)); matlab(T,bundle(t))
            return (g,p,e,t)
        end
        #=refinemesh3(g,p::ChainBundle,e,t,s...) = MATLAB.mxcall(:refinemesh,3,g,matlab(p),matlab(e),matlab(t),s...)
        refinemesh4(g,p::ChainBundle,e,t,s...) = MATLAB.mxcall(:refinemesh,4,g,matlab(p),matlab(e),matlab(t),s...)
        refinemesh(g,p::ChainBundle,e,t) = refinemesh3(g,p,e,t)
        refinemesh(g,p::ChainBundle,e,t,s::String) = refinemesh3(g,p,e,t,s)
        refinemesh(g,p::ChainBundle,e,t,η::Vector{Int}) = refinemesh3(g,p,e,t,float.(η))
        refinemesh(g,p::ChainBundle,e,t,η::Vector{Int},s::String) = refinemesh3(g,p,e,t,float.(η),s)
        refinemes(g,p::ChainBundle,e,t,u) = refinemesh4(g,p,e,t,u)
        refinemesh(g,p::ChainBundle,e,t,u,s::String) = refinemesh4(g,p,e,t,u,s)
        refinemesh(g,p::ChainBundle,e,t,u,η) = refinemesh4(g,p,e,t,u,float.(η))
        refinemesh(g,p::ChainBundle,e,t,u,η,s) = refinemesh4(g,p,e,t,u,float.(η),s)
        refinemesh!(g::Matrix{Int},p::ChainBundle,args...) = refinemesh!(Matrix{Float64}(g),p,args...)
        function refinemesh!(g,p::ChainBundle{V},e,t,s...) where V
            P,E,T = refinemesh(g,p,e,t,s...); l = size(P,1)+1
            matlab(P,bundle(p)); matlab(E,bundle(e)); matlab(T,bundle(t))
            submesh!(p); array!(t); el,tl = list(1,l-1),list(1,l)
            bundle_cache[bundle(p)] = [Chain{V,1,Float64}(vcat(1,P[:,k])) for k ∈ 1:size(P,2)]
            bundle_cache[bundle(e)] = [Chain{↓(p),1,Int}(Int.(E[el,k])) for k ∈ 1:size(E,2)]
            bundle_cache[bundle(t)] = [Chain{p,1,Int}(Int.(T[tl,k])) for k ∈ 1:size(T,2)]
            return (p,e,t)
        end=#
    end
end

end # module Cartan
