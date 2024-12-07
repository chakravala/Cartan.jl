
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

export bound, boundabove, boundbelow, boundlog, isclosed, updatetopology
export centraldiff, centraldiff_slow, centraldiff_fast
export gradient, gradient_slow, gradient_fast, unitgradient
export integral, integrate, ∫

# analysis

boundabove(s::LocalTensor{B,T},lim=10) where {B,T<:Real} = fiber(s)≤lim ? s : LocalTensor(base(s),T(lim))
boundabove(x::T,lim=10) where T<:Real = x≤lim ? x : T(lim)
boundabove(t::TensorField,lim=10) = TensorField(base(t), boundabove.(codomain(t),lim))
boundbelow(s::LocalTensor{B,T},lim=-10) where {B,T<:Real} = fiber(s)≥lim ? s : LocalTensor(base(s),T(lim))
boundbelow(x::T,lim=-10) where T<:Real = x≥lim ? x : T(lim)
boundbelow(t::TensorField,lim=-10) = TensorField(base(t), boundbelow.(codomain(t),lim))
bound(s::LocalTensor{B,T},lim=10) where {B,T<:Real} = abs(fiber(s))≤lim ? s : LocalTensor(base(s),T(sign(fiber(s)*lim)))
bound(s::LocalTensor{B,T},lim=10) where {B,T} = (x=abs(fiber(s)); x≤lim ? s : ((lim/x)*s))
bound(x::T,lim=10) where T<:Real = abs(x)≤lim ? x : T(sign(x)*lim)
bound(z,lim=10) = (x=abs(z); x≤lim ? z : (lim/x)*z)
bound(t::TensorField,lim=10) = TensorField(base(t), bound.(codomain(t),lim))
boundlog(s::LocalTensor{B,T},lim=10) where {B,T<:Real} = (x=abs(fiber(s)); x≤lim ? s : LocalTensor(base(s),T(sign(fiber(s))*(lim+log(x+1-lim)))))
boundlog(s::LocalTensor{B,T},lim=10) where {B,T} = (x=abs(fiber(s)); x≤lim ? s : LocalTensor(base(s),T(fiber(s)*((lim+log(x+1-lim))/x))))
boundlog(z::T,lim=10) where T<:Real = (x=abs(z); x≤lim ? z : T(sign(z)*(lim+log(x+1-lim))))
boundlog(z,lim=10) = (x=abs(fiber(s)); x≤lim ? s : LocalTensor(base(s),T(fiber(s)*((lim+log(x+1-lim))/x))))
boundlog(t::TensorField,lim=10) = TensorField(base(t), boundlog.(codomain(t),lim))

isclosed(t::IntervalMap) = norm(codomain(t)[end]-codomain(t)[1]) ≈ 0
updatetopology(t::IntervalMap) = isclosed(t) ? TorusTopology(t) : t

export Grid

const Grid{N,C,PA,TA} = GridFrameBundle{C,N,PA,TA}
Grid(v::A,t::I=OpenTopology(size(v))) where {N,T,A<:AbstractArray{T,N},I} = GridFrameBundle(0,PointArray(0,v),t)

#=struct Grid{N,T,A<:AbstractArray{T,N},I<:ImmersedTopology}
    v::A
    t::I
    Grid(v::A,t::I=OpenTopology(size(v))) where {N,T,A<:AbstractArray{T,N},I} = new{N,T,A,I}(v,t)
end=#

#Grid(v::A) where {N,T,A<:AbstractArray{T,N}} = Grid(v,Global{N}(InducedMetric()))
#Grid(v::GridFrameBundle{<:Real}) = Grid(points(v))
#Grid(v::GridFrameBundle) = Grid(points(v),fiber(metricextensor(v)))

#=Base.size(m::Grid) = size(m.v)

@generated function Base.getindex(g::Grid{M,T,A,<:OpenTopology} where {T,A},j::Int,::Val{N},i::Vararg{Int}) where {M,N}
    :(Base.getindex(g.v,$([k≠N ? :(i[$k]) : :(i[$k]+j) for k ∈ 1:M]...)))
end
@generated function Base.getindex(g::Grid{M},j::Int,::Val{N},i::Vararg{Int}) where {M,N}
    :(Base.getindex(g.v,Base.getindex(g.t,$([k≠N ? :(i[$k]) : :(i[$k]+j) for k ∈ 1:M]...))...))
end=#
Base.getindex(g::GridFrameBundle{C,M,<:FiberBundle,<:OpenTopology} where C,j::Int,n::Val,i::Vararg{Int}) where M = getpoint(g,j,n,i...)
@generated function getpoint(g::GridFrameBundle{C,M,<:FiberBundle} where C,j::Int,::Val{N},i::Vararg{Int}) where {M,N}
    :(Base.getindex(points(g),$([k≠N ? :(@inbounds i[$k]) : :(@inbounds i[$k]+j) for k ∈ 1:M]...)))
end
@generated function Base.getindex(g::GridFrameBundle{C,M} where C,j::Int,n::Val{N},i::Vararg{Int}) where {M,N}
    :(Base.getindex(points(g),Base.getindex(immersion(g),n,$([k≠N ? :(@inbounds i[$k]) : :(@inbounds i[$k]+j) for k ∈ 1:M]...))...))
end

# centraldiff

centraldiffdiff(f,dt,l) = centraldiff(centraldiff(f,dt,l),dt,l)
centraldiffdiff(f,dt) = centraldiffdiff(f,dt,size(f))
centraldiff(f::AbstractVector,args...) = centraldiff_slow(f,args...)
centraldiff(f::AbstractArray,args...) = centraldiff_fast(f,args...)
centraldifffiber(f::AbstractVector,args...) = centraldiff_slow_fiber(f,args...)
centraldifffiber(f::AbstractArray,args...) = centraldiff_fast_fiber(f,args...)
centraldiffpoints(f::AbstractVector,args...) = centraldiff_slow_points(f,args...)
centraldiffpoints(f::AbstractArray,args...) = centraldiff_fast_points(f,args...)

gradient(f::IntervalMap,args...) = gradient_slow(f,args...)
gradient(f::TensorField{B,F,N,<:AbstractArray} where {B,F,N},args...) = gradient_fast(f,args...)
function unitgradient(f::TensorField,args...)
    t = gradient(f,args...)
    return t/abs(t)
end

(::Derivation)(t::TensorField) = getnabla(t)
function getnabla(t::TensorField)
    n = ndims(t)
    V = Submanifold(tangent(S"0",1,n))
    Chain(Values{n,Any}(Λ(V).b[2:n+1]...))
end

export invd, cartan, firststructure, secondstructure

cartan(ξ) = invd(ξ)⋅ξ
firststructure(θ,ω) = d(θ)+ω∧θ
secondstructure(ω) = d(ω)+ω∧ω

Grassmann.curl(t::TensorField) = ⋆d(t)
Grassmann.d(t::TensorField) = TensorField(fromany(∇(t)∧Chain(t)))
Grassmann.d(t::GlobalSection) = gradient(t)
Grassmann.∂(t::TensorField) = TensorField(fromany(Chain(t)⋅∇(t)))
Grassmann.d(t::ScalarField{B,<:AbstractReal,N,<:AbstractFrameBundle} where {B,N}) = gradient(t)
#Grassmann.∂(t::ScalarField) = gradient(t)
#Grassmann.∂(t::VectorField) = TensorField(domain(t), sum.(value.(codomain(t))))
#=function Grassmann.∂(t::VectorField{G,B,<:Chain{V},N,T} where {B,N,T}) where {G,V}
    n = mdims(V)
    TensorField(domain(t), Real.(Chain{V,G}(ones(Values{binomial(n,G),Int})).⋅codomain(t)))
end
function Grassmann.∂(t::GradedField{G,B,<:Chain{V}} where B) where {G,V}
    n = mdims(V)
    TensorField(domain(t), (Chain{V,G}(ones(Values{binomial(n,G),Int})).⋅codomain(t)))
end=#

@generated function dvec(t::TensorField{B,<:Chain{V,G} where {V,G}} where B)
    V = Manifold(fibertype(t)); N = mdims(V)
    Expr(:.,:(Chain{$V,1}),Expr(:tuple,[:(gradient(getindex.(t,$i))) for i ∈ list(1,N)]...))
end

@generated function Grassmann.d(t::TensorField{B,<:Chain{V,G,<:Chain} where {V,G}} where B)
    V = Manifold(fibertype(t)); N = mdims(V)
    Expr(:.,:(Chain{$V,1}),Expr(:tuple,[:(d(getindex.(t,$i))) for i ∈ list(1,N)]...))
end
@generated function Grassmann.d(t::EndomorphismField{B,<:Endomorphism,N,<:AbstractFrameBundle} where {B,N})
    fibertype(t) <: DiagonalOperator && (return :(DiagonalOperator.(dvec(value.(t)))))
    V = Manifold(fibertype(t)); N = mdims(V)
    syms = Symbol.(:x,list(1,N))
    Expr(:block,:(V = $V),
        :($(Expr(:tuple,syms...)) = $(Expr(:tuple,[:(getindex.(t,$i)) for i ∈ list(1,N)]...))),
        Expr(:call,:TensorField,:(base(t)),Expr(:.,:TensorOperator,Expr(:tuple,Expr(:.,:(Chain{V,1}),Expr(:tuple,[Expr(:.,:(Chain{V,1}),Expr(:tuple,
        [:(fiber(d(getindex.($(syms[j]),$i)))) for i ∈ list(1,N)]...)) for j ∈ list(1,N)]...))))))
end
@generated function invd(t::EndomorphismField)
    fibertype(t) <: DiagonalOperator && (return :(DiagonalOperator.(dvec(.-value.(t)))))
    V = Manifold(fibertype(t)); N = mdims(V)
    syms = Symbol.(:x,list(1,N))
    Expr(:block,:(V = $V),
        :($(Expr(:tuple,syms...)) = $(Expr(:tuple,[:(-getindex.(t,$i)) for i ∈ list(1,N)]...))),
        Expr(:.,:TensorOperator,Expr(:tuple,Expr(:.,:(Chain{V,1}),Expr(:tuple,[Expr(:.,:(Chain{V,1}),Expr(:tuple,
        [:(d(getindex.($(syms[i]),$j))) for i ∈ list(1,N)]...)) for j ∈ list(1,N)]...)))))
end

for op ∈ (:(Base.:*),:(Base.:/),:(Grassmann.:∧),:(Grassmann.:∨))
    @eval begin
        $op(::Derivation,t::TensorField) = TensorField(fromany($op(∇(t),Chain(t))))
        $op(t::TensorField,::Derivation) = TensorField(fromany($op(Chain(t),∇(t))))
    end
end
LinearAlgebra.dot(::Derivation,t::TensorField) = TensorField(fromany(Grassmann.contraction(∇(t),Chain(t))))
LinearAlgebra.dot(t::TensorField,::Derivation) = TensorField(fromany(Grassmann.contraction(Chain(t),∇(t))))

function Base.:*(n::Submanifold,t::TensorField)
    if istangent(n)
        gradient(t,Val(indices(n)[1]))
    else
        TensorField(domain(t), (Ref(n).*codomain(t)))
    end
end
function Base.:*(t::TensorField,n::Submanifold)
    if istangent(n)
        gradient(t,Val(indices(n)[1]))
    else
        TensorField(domain(t), (codomain(t).*Ref(n)))
    end
end
function LinearAlgebra.dot(n::Submanifold,t::TensorField)
    if istangent(n)
        gradient(t,Val(indices(n)[1]))
    else
        TensorField(domain(t), dot.(Ref(n),codomain(t)))
    end
end
function LinearAlgebra.dot(t::TensorField,n::Submanifold)
    if istangent(n)
        gradient(t,Val(indices(n)[1]))
    else
        TensorField(domain(t), dot.(codomain(t),Ref(n)))
    end
end

for fun ∈ (:_slow,:_fast)
    cd,grad = Symbol(:centraldiff,fun),Symbol(:gradient,fun)
    cdg,cdp,cdf = Symbol(cd,:_calc),Symbol(cd,:_points),Symbol(cd,:_fiber)
    @eval begin
        $cdf(f,args...) = $cdg(Grid(codomain(f),immersion(f)),args...)
        $cdp(f,args...) = $cdg(Grid(points(f),immersion(f)),args...)
        $cdp(f::TensorField{B,F,Nf,<:RealSpace{Nf,P,<:InducedMetric} where P} where {B,F,Nf},n::Val{N},args...) where N = $cd(points(f).v[N],args...)
        function $grad(f::IntervalMap,d::AbstractVector=$cdp(f))
            TensorField(domain(f), $cdf(f,d))
        end
        function $grad(f::TensorField{B,F,N,<:AbstractArray} where {B,F,N},d::AbstractArray=$cd(base(f)))
            TensorField(domain(f), $cdf(f,d))
        end
        function $grad(f::IntervalMap,::Val{1},d::AbstractVector=$cdp(f))
            TensorField(domain(f), $cdf(f,d))
        end
        function $grad(f::TensorField{B,F,Nf,<:RealSpace{Nf,P,<:InducedMetric} where P} where {B,F,Nf},n::Val{N},d::AbstractArray=$cd(points(f).v[N])) where N
            TensorField(domain(f), $cdf(f,n,d))
        end
        function $grad(f::TensorField{B,F,Nf,<:RealSpace} where {B,F,Nf},n::Val{N},d::AbstractArray=$cd(points(f).v[N])) where N
            l = size(points(f))
            dg = sqrt.(getindex.(metricextensor(f),N+1,N+1))
            @threads for i ∈ l[1]; for j ∈ l[2]
                dg[i,j] *= d[isone(N) ? i : j]
            end end
            TensorField(domain(f), $cdf(f,n,dg))
        end
        function $grad(f::TensorField,n::Val,d::AbstractArray=$cdp(f,n))
            TensorField(domain(f), $cdf(f,n,d))
        end
        $grad(f::TensorField,n::Int,args...) = $grad(f,Val(n),args...)
        $cd(f::AbstractArray,args...) = $cdg(Grid(f),args...)
        function $cdg(f::Grid{1},dt::Real,s::Tuple=size(f))
            d = similar(points(f))
            @threads for i ∈ OneTo(s[1])
                d[i] = $cdg(f,s,i)/$cdg(i,dt,l)
            end
            return d
        end
        function $cdg(f::Grid{1},dt::Vector,s::Tuple=size(f))
            d = similar(points(f))
            @threads for i ∈ OneTo(s[1])
                d[i] = $cdg(f,s,i)/dt[i]
            end
            return d
        end
        function $cdg(f::Grid{1},s::Tuple=size(f))
            d = similar(points(f))
            @threads for i ∈ OneTo(s[1])
                d[i] = $cdg(f,s,i)
            end
            return d
        end
        function $cdg(f::Grid{2},dt::AbstractMatrix,s::Tuple=size(f))
            d = Array{Chain{Submanifold(2),1,pointtype(f),2},2}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2])
                d[i,j] = Chain($cdg(f,s,i,j).v./dt[i,j].v)
            end end
            return d
        end
        function $cdg(f::Grid{2},s::Tuple=size(f))
            d = Array{Chain{Submanifold(2),1,pointtype(f),2},2}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2])
                d[i,j] = $cdg(f,s,i,j)
            end end
            return d
        end
        function $cdg(f::Grid{3},dt::AbstractArray{T,3} where T,s::Tuple=size(f))
            d = Array{Chain{Submanifold(3),1,pointtype(f),3},3}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3])
                d[i,j,k] = Chain($cdg(f,s,i,j,k).v./dt[i,j,k].v)
            end end end
            return d
        end
        function $cdg(f::Grid{3},s::Tuple=size(f))
            d = Array{Chain{Submanifold(3),1,pointtype(f),3},3}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3])
                d[i,j,k] = $cdg(f,s,i,j,k)
            end end end
            return d
        end
        function $cdg(f::Grid{4},dt::AbstractArray{T,4} where T,s::Tuple=size(f))
            d = Array{Chain{Submanifold(4),1,pointtype(f),4},4}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3]); for l ∈ OneTo(s[4])
                d[i,j,k,l] = Chain($cdg(f,s,i,j,k,l).v./dt[i,j,k,l].v)
            end end end end
            return d
        end
        function $cdg(f::Grid{4},s::Tuple=size(f))
            d = Array{Chain{Submanifold(4),1,pointtype(f),4},4}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3]); for l ∈ OneTo(s[4])
                d[i,j,k,l] = $cdg(f,s,i,j,k,l)
            end end end end
            return d
        end
        function $cdg(f::Grid{5},dt::AbstractArray{T,5} where T,s::Tuple=size(f))
            d = Array{Chain{Submanifold(5),1,pointtype(f),5},5}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3]); for l ∈ OneTo(s[4]); for o ∈ OneTo(s[5])
                d[i,j,k,l,o] = Chain($cdg(f,s,i,j,k,l,o).v./dt[i,j,k,l,o].v)
            end end end end end
            return d
        end
        function $cdg(f::Grid{5},s::Tuple=size(f))
            d = Array{Chain{Submanifold(5),1,pointtype(f),5},5}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3]); for l ∈ OneTo(s[4]); for o ∈ OneTo(s[5])
                d[i,j,k,l,o] = $cdg(f,s,i,j,k,l,o)
            end end end end end
            return d
        end
        function $cdg(f::Grid{2},n::Val{1},dt::AbstractVector,s::Tuple=size(f))
            d = Array{pointtype(f),2}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2])
                d[i,j] = $cdg(f,(@inbounds s[1]),n,i,j)/dt[i]
            end end
            return d
        end
        function $cdg(f::Grid{2},n::Val{2},dt::AbstractVector,s::Tuple=size(f))
            d = Array{pointtype(f),2}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2])
                d[i,j] = $cdg(f,(@inbounds s[2]),n,i,j)/dt[j]
            end end
            return d
        end
        function $cdg(f::Grid{2},n::Val{N},dt::AbstractMatrix,s::Tuple=size(f)) where N
            d = Array{pointtype(f),2}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2])
                d[i,j] = $cdg(f,s[N],n,i,j)/dt[i,j]
            end end
            return d
        end
        function $cdg(f::Grid{2},n::Val{N},s::Tuple=size(f)) where N
            d = Array{pointtype(f),2}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2])
                d[i,j] = $cdg(f,s[N],n,i,j)
            end end
            return d
        end
        function $cdg(f::Grid{3},n::Val{1},dt::AbstractVector,s::Tuple=size(f))
            d = Array{pointtype(f),3}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3])
                d[i,j,k] = $cdg(f,(@inbounds s[1]),n,i,j,k)/dt[i]
            end end end
            return d
        end
        function $cdg(f::Grid{3},n::Val{2},dt::AbstractVector,s::Tuple=size(f))
            d = Array{pointtype(f),3}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3])
                d[i,j,k] = $cdg(f,(@inbounds s[2]),n,i,j,k)/dt[j]
            end end end
            return d
        end
        function $cdg(f::Grid{3},n::Val{3},dt::AbstractVector,s::Tuple=size(f))
            d = Array{pointtype(f),3}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3])
                d[i,j,k] = $cdg(f,(@inbounds s[3]),n,i,j,k)/dt[k]
            end end end
            return d
        end
        function $cdg(f::Grid{3},n::Val{N},dt::AbstractArray,s::Tuple=size(f)) where N
            d = Array{pointtype(f),3}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3])
                d[i,j,k] = $cdg(f,s[N],n,i,j,k)/dt[i,j,k]
            end end end
            return d
        end
        function $cdg(f::Grid{3},n::Val{N},s::Tuple=size(f)) where N
            d = Array{pointtype(f),3}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3])
                d[i,j,k] = $cdg(f,s[N],n,i,j,k)
            end end end
            return d
        end
        function $cdg(f::Grid{4},n::Val{1},dt::AbstractVector,s::Tuple=size(f))
            d = Array{pointtype(f),4}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3]); for l ∈ OneTo(s[4])
                d[i,j,k,l] = $cdg(f,(@inbounds s[1]),n,i,j,k,l)/dt[i]
            end end end end
            return d
        end
        function $cdg(f::Grid{4},n::Val{2},dt::AbstractVector,s::Tuple=size(f))
            d = Array{pointtype(f),4}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3]); for l ∈ OneTo(s[4])
                d[i,j,k,l] = $cdg(f,(@inbounds s[2]),n,i,j,k,l)/dt[j]
            end end end end
            return d
        end
        function $cdg(f::Grid{4},n::Val{3},dt::AbstractVector,s::Tuple=size(f))
            d = Array{pointtype(f),4}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3]); for l ∈ OneTo(s[4])
                d[i,j,k,l] = $cdg(f,(@inbounds s[3]),n,i,j,k,l)/dt[k]
            end end end end
            return d
        end
        function $cdg(f::Grid{4},n::Val{4},dt::AbstractVector,s::Tuple=size(f))
            d = Array{pointtype(f),4}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3]); for l ∈ OneTo(s[4])
                d[i,j,k,l] = $cdg(f,(@inbounds s[4]),n,i,j,k,l)/dt[l]
            end end end end
            return d
        end
        function $cdg(f::Grid{4},n::Val{N},dt::AbstractArray,s::Tuple=size(f)) where N
            d = Array{pointtype(f),4}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3]); for l ∈ OneTo(s[4])
                d[i,j,k,l] = $cdg(f,s[N],n,i,j,k,l)/dt[i,j,k,l]
            end end end end
            return d
        end
        function $cdg(f::Grid{4},n::Val{N},s::Tuple=size(f)) where N
            d = Array{pointtype(f),4}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3]); for l ∈ OneTo(s[4])
                d[i,j,k,l] = $cdg(f,s[N],n,i,j,k,l)
            end end end end
            return d
        end
        function $cdg(f::Grid{5},n::Val{1},dt::AbstractVector,s::Tuple=size(f))
            d = Array{pointtype(f),5}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3]); for l ∈ OneTo(s[4]); for o ∈ OneTo(s[5])
                d[i,j,k,l,o] = $cdg(f,(@inbounds s[1]),n,i,j,k,l,o)/dt[i]
            end end end end end
            return d
        end
        function $cdg(f::Grid{5},n::Val{2},dt::AbstractVector,s::Tuple=size(f))
            d = Array{pointtype(f),5}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3]); for l ∈ OneTo(s[4]); for o ∈ OneTo(s[5])
                d[i,j,k,l,o] = $cdg(f,(@inbounds s[2]),n,i,j,k,l,o)/dt[j]
            end end end end end
            return d
        end
        function $cdg(f::Grid{5},n::Val{3},dt::AbstractVector,s::Tuple=size(f))
            d = Array{pointtype(f),5}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3]); for l ∈ OneTo(s[4]); for o ∈ OneTo(s[5])
                d[i,j,k,l,o] = $cdg(f,(@inbounds s[3]),n,i,j,k,l,o)/dt[k]
            end end end end end
            return d
        end
        function $cdg(f::Grid{5},n::Val{4},dt::AbstractVector,s::Tuple=size(f))
            d = Array{pointtype(f),5}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3]); for l ∈ OneTo(s[4]); for o ∈ OneTo(s[5])
                d[i,j,k,l,o] = $cdg(f,(@inbounds s[4]),n,i,j,k,l,o)/dt[l]
            end end end end end
            return d
        end
        function $cdg(f::Grid{5},n::Val{5},dt::AbstractVector,s::Tuple=size(f))
            d = Array{pointtype(f),5}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3]); for l ∈ OneTo(s[4]); for o ∈ OneTo(s[5])
                d[i,j,k,l,o] = $cdg(f,(@inbounds s[5]),n,i,j,k,l,o)/dt[o]
            end end end end end
            return d
        end
        function $cdg(f::Grid{5},n::Val{N},dt::AbstractArray,s::Tuple=size(f)) where N
            d = Array{pointtype(f),5}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3]); for l ∈ OneTo(s[4]); for o ∈ OneTo(s[5])
                d[i,j,k,l,o] = $cdg(f,s[N],n,i,j,k,l,o)/dt[i,j,k,l,o]
            end end end end end
            return d
        end
        function $cdg(f::Grid{5},n::Val{N},s::Tuple=size(f)) where N
            d = Array{pointtype(f),5}(undef,s...)
            @threads for i ∈ OneTo(s[1]); for j ∈ OneTo(s[2]); for k ∈ OneTo(s[3]); for l ∈ OneTo(s[4]); for o ∈ OneTo(s[5])
                d[i,j,k,l,o] = $cdg(f,s[N],n,i,j,k,l,o)
            end end end end end
            return d
        end
        $cdg(f::Grid{1},s::Tuple,i::Int) = $cdg(f,s[1],Val(1),i)
        @generated function $cdg(f::Grid{N},s::Tuple,i::Vararg{Int}) where N
            :(Chain($([:($$cdg(f,s[$n],Val($n),i...)) for n ∈ list(1,N)]...)))
        end
        $cd(f::RealRegion) = ProductSpace($cd.(f.v))
        $cd(f::GridFrameBundle{Coordinate{P,G},N,<:PointArray{P,G,N,<:RealRegion,<:Global{N,<:InducedMetric}},<:ProductTopology}) where {P,G,N} = ProductSpace($cd.(base(base(f)).v))
        $cd(f::GridFrameBundle{Coordinate{P,G},N,<:PointArray{P,G,N,<:RealRegion,<:Global{N,<:InducedMetric}},<:OpenTopology}) where {P,G,N} = ProductSpace($cd.(base(base(f)).v))
        $cd(f::GridFrameBundle{Coordinate{P,G},N,<:PointArray{P,G,N,<:RealRegion,<:Global{N,<:InducedMetric}}}) where {P,G,N} = sum.(value.($cdg(f)))
        #$cd(f::GridFrameBundle{Coordinate{P,G},N,<:PointArray{P,G,N,<:RealRegion},<:ProductTopology}) where {P,G,N} = applymetric.($cd(base(base(f))),metricextensor(f))
        $cd(f::GridFrameBundle{Coordinate{P,G},N,<:PointArray{P,G,N,<:RealRegion},<:OpenTopology}) where {P,G,N} = applymetric.($cd(base(base(f))),metricextensor(f))
        $cd(f::GridFrameBundle{Coordinate{P,G},N,<:PointArray{P,G,N,<:RealRegion}}) where {P,G,N} = applymetric.(sum.(value.($cdg(f))),metricextensor(f))
        function $cd(f::AbstractRange,s::Tuple=size(f))
            d = Vector{eltype(f)}(undef,s[1])
            @threads for i ∈ OneTo(s[1])
                d[i] = $cdg(i,step(f),s[1])
            end
            return d
        end
        function $cd(dt::Real,s::Tuple)
            d = Vector{Float64}(undef,s[1])
            @threads for i ∈ OneTo(s[1])
                d[i] = $cdg(i,dt,s[1])
            end
            return d
        end
    end
end

applymetric(f::Chain{V,G},g::DiagonalOperator{W,<:Multivector} where W) where {V,G} = Chain{V,G}(value(f)./sqrt.(value(value(g)(Val(G)))))
applymetric(f::Chain{V,G},g::DiagonalOperator{W,<:Chain} where W) where {V,G} = Chain{V,G}(value(f)./sqrt.(value(value(g))))
applymetric(f::Chain{V,G},g::Outermorphism) where {V,G} = applymetric(f,(@inbounds value(g)[1]))
applymetric(f::Chain{V,G},g::Endomorphism{W,<:Simplex} where W) where {V,G} = applymetric(f,value(g))
@generated function applymetric(x::Chain{V,G,T,N} where {G,T},g::Simplex) where {V,N}
    Expr(:call,:(Chain{V}),[:(x[$k]/sqrt(g[$k,$k])) for k ∈ list(1,N)]...)
end


function centraldiff_slow_calc(f::Grid{M,T,PA,<:OpenTopology} where {M,T,PA},l::Int,n::Val{N},i::Vararg{Int}) where N #l=size(f)[N]
    if isone(i[N])
        18f[1,n,i...]-9f[2,n,i...]+2f[3,n,i...]-11points(f)[i...]
    elseif i[N]==l
        11points(f)[i...]-18f[-1,n,i...]+9f[-2,n,i...]-2f[-3,n,i...]
    elseif i[N]==2
        6f[1,n,i...]-f[2,n,i...]-3points(f)[i...]-2f[-1,n,i...]
    elseif i[N]==l-1
        3points(f)[i...]-6f[-1,n,i...]+f[-2,n,i...]+2f[1,n,i...]
    else
        f[-2,n,i...]+8(f[1,n,i...]-f[-1,n,i...])-f[2,n,i...]
    end
end
function Cartan.centraldiff_slow_calc(f::Grid,l::Int,n::Val{N},i::Vararg{Int}) where N #l=size(f)[N]
    if isone(i[N])
        if iszero(f.cod.r[2N-1])
            18f[1,n,i...]-9f[2,n,i...]+2f[3,n,i...]-11points(f)[i...]
        elseif f.cod.p[2N-1]≠2N-1
            f[-2,n,i...]+7(f[0,n,i...]-points(f)[i...])+8(f[1,n,i...]-f[-1,n,i...])-f[2,n,i...]
        else
            (-f[-2,n,i...])+7(-f[0,n,i...]-points(f)[i...])+8(f[1,n,i...]+f[-1,n,i...])-f[2,n,i...]
        end
    elseif i[N]==l
        if iszero(f.cod.r[2N])
            11points(f)[i...]-18f[-1,n,i...]+9f[-2,n,i...]-2f[-3,n,i...]
        elseif f.cod.p[2N]≠2N
            f[-2,n,i...]+8(f[1,n,i...]-f[-1,n,i...])+7(points(f)[i...]-f[0,n,i...])-f[2,n,i...]
        else
            f[-2,n,i...]+8(-f[1,n,i...]-f[-1,n,i...])+7(points(f)[i...]-f[0,n,i...])+f[2,n,i...]
        end
    elseif i[N]==2
        if iszero(f.cod.r[2N-1])
            6f[1,n,i...]-f[2,n,i...]-3points(f)[i...]-2f[-1,n,i...]
        elseif f.cod.p[2N-1]≠2N-1
            f[-2,n,i...]-f[-1,n,i...]+8f[1,n,i...]-7getpoint(f,-1,n,i...)-f[2,n,i...]
        else
            (-f[-2,n,i...])+f[-1,n,i...]+8f[1,n,i...]-7getpoint(f,-1,n,i...)-f[2,n,i...]
        end
    elseif i[N]==l-1
        if iszero(f.cod.r[2N])
            3points(f)[i...]-6f[-1,n,i...]+f[-2,n,i...]+2f[1,n,i...]
        elseif f.cod.p[2N]≠2N
            f[-2,n,i...]+7getpoint(f,1,n,i...)-8f[-1,n,i...]+f[1,n,i...]-f[2,n,i...]
        else
            f[-2,n,i...]+7getpoint(f,1,n,i...)-8f[-1,n,i...]-f[1,n,i...]+f[2,n,i...]
        end
    else
        getpoint(f,-2,n,i...)+8(getpoint(f,1,n,i...)-getpoint(f,-1,n,i...))-getpoint(f,2,n,i...)
    end
end

function centraldiff_slow_calc(i::Int,dt::Real,d1::Real,d2::Real,l::Int)
    if isone(i) # 8*(d1+d2)-(d0+d1+d2+d3)
        6(dt+d1) # (8-2)*(d1+d2)
    elseif i==l
        6(dt+d2) # (8-2)*(d1+d2)
    elseif i==2
        13dt-d1 # 8*2d1-3d1-d2
    elseif i==l-1
        13dt-d2 # 8*2d1-3d1-d2
    else
        12dt # (8-2)*2dt
    end
end
function centraldiff_slow_calc(::Val{1},i::Int,dt::Real,d1::Real,l::Int)
    if isone(i)
        6(dt+d1) # (8-2)*(dt+d1)
    elseif i==l
        6dt
    elseif i==2
        13dt-d1 # 8*2dt-3dt-d1
    elseif i==l-1
        6dt
    else
        12dt # (8-2)*2dt
    end
end
function centraldiff_slow_calc(::Val{2},i::Int,dt::Real,d2::Real,l::Int)
    if isone(i)
        6dt
    elseif i==l
        6(dt+d2) # (8-2)*(dt+d2)
    elseif i==2
        6dt
    elseif i==l-1
        13dt-d2 # 8*2dt-3dt-d2
    else
        12dt # (8-2)*2dt
    end
end
function centraldiff_slow_calc(i::Int,dt::Real,l::Int)
    if i∈(1,2,l-1,l)
        6dt # (8-2)*dt
    else
        12dt # (8-2)*2dt
    end
end

function centraldiff_fast_calc(f::Grid{M,T,PA,<:OpenTopology} where {M,T,PA},l::Int,n::Val{N},i::Vararg{Int}) where N
    if isone(i[N]) # 4f[1,k,i...]-f[2,k,i...]-3f.v[i...]
        18f[1,n,i...]-9f[2,n,i...]+2f[3,n,i...]-11points(f)[i...]
    elseif i[N]==l # 3f.v[i...]-4f[-1,k,i...]+f[-2,k,i...]
        11points(f)[i...]-18f[-1,n,i...]+9f[-2,n,i...]-2f[-3,n,i...]
    else
        f[1,n,i...]-f[-1,n,i...]
    end
end
function Cartan.centraldiff_fast_calc(f::Grid,l::Int,n::Val{N},i::Vararg{Int}) where N
    if isone(i[N])
        if iszero(f.cod.r[2N-1])
            18f[1,n,i...]-9f[2,n,i...]+2f[3,n,i...]-11points(f)[i...]
        elseif f.cod.p[2N-1]≠2N-1
            (f[1,n,i...]-points(f)[i...])+(f[0,n,i...]-f[-1,n,i...])
        else # mirror
            (f[1,n,i...]-points(f)[i...])-(f[0,n,i...]-f[-1,n,i...])
        end
    elseif i[N]==l
        if iszero(f.cod.r[2N])
            11points(f)[i...]-18f[-1,n,i...]+9f[-2,n,i...]-2f[-3,n,i...]
        elseif f.cod.p[2N]≠2N
            (f[1,n,i...]-f[0,n,i...])+(points(f)[i...]-f[-1,n,i...])
        else # mirror
            (f[0,n,i...]-f[1,n,i...])+(points(f)[i...]-f[-1,n,i...])
        end
    else
        getpoint(f,1,n,i...)-getpoint(f,-1,n,i...)
    end
end

centraldiff_fast_calc(i::Int,dt::Real,d1::Real,d2::Real,l::Int) = isone(i) ? dt+d1 : i==l ? dt+d2 : 2dt
centraldiff_fast_calc(::Val{1},i::Int,d1::Real,d2::Real,l::Int) = isone(i) ? d1+d2 : 2dt
centraldiff_fast_calc(::Val{2},i::Int,d1::Real,d2::Real,l::Int) = i==l ? d1+d2 : 2dt
centraldiff_fast_calc(i::Int,dt::Real,l::Int) = i∈(1,l) ? 6dt : 2dt
#centraldiff_fast_calc(i::Int,dt::Real,l::Int) = 2dt

qnumber(n,q) = (q^n-1)/(q-1)
qfactorial(n,q) = prod(cumsum([q^k for k ∈ 0:n-1]))
qbinomial(n,k,q) = qfactorial(n,q)/(qfactorial(n-k,q)*qfactorial(k,q))

richardson(k) = [(-1)^(j+k)*2^(j*(1+j))*qbinomial(k,j,4)/prod([4^n-1 for n ∈ 1:k]) for j ∈ k:-1:0]

# parallelization

select1(n,j,k=:k,f=:f) = :($f[$([i≠j ? :(:) : k for i ∈ 1:n]...)])
select2(n,j,k=:k,f=:f) = :($f[$([i≠j ? :(:) : k for i ∈ 1:n if i≠j]...)])
psum(A,j) = psum(A,Val(j))
pcumsum(A,j) = pcumsum(A,Val(j))
for N ∈ 2:5
    for J ∈ 1:N
        @eval function psum(A::AbstractArray{T,$N} where T,::Val{$J})
            S = similar(A);
            @threads for k in axes(A,$J)
                @views sum!($(select1(N,J,:k,:S)),$(select1(N,J,:k,:A)),dims=$J)
            end
            return S
        end
    end
end
for N ∈ 2:5
    for J ∈ 1:N
        @eval function pcumsum(A::AbstractArray{T,$N} where T,::Val{$J})
            S = similar(A);
            @threads for k in axes(A,$J)
                @views cumsum!($(select1(N,J,:k,:S)),$(select1(N,J,:k,:A)),dims=$J)
            end
            return S
        end
    end
end

# trapezoid # ⎎, ∇

integrate(args...) = trapz(args...)

arclength(f::Vector) = sum(value.(abs.(diff(f))))
trapz(f::IntervalMap,d::AbstractVector=diff(points(f))) = sum((d/2).*(f.cod[2:end]+f.cod[1:end-1]))
trapz1(f::Vector,h::Real) = h*((f[1]+f[end])/2+sum(f[2:end-1]))
trapz(f::IntervalMap,j::Int) = trapz(f,Val(j))
trapz(f::IntervalMap,j::Val{1}) = trapz(f)
trapz(f::ParametricMap,j::Int) = trapz(f,Val(j))
trapz(f::ParametricMap,j::Val{J}) where J = remove(domain(f),j) → trapz2(codomain(f),j,diff(points(f).v[J]))
trapz(f::ParametricMap{B,F,N,<:AlignedSpace} where {B,F,N},j::Val{J}) where J = remove(domain(f),j) → trapz1(codomain(f),j,step(points(f).v[J]))
gentrapz1(n,j,h=:h,f=:f) = :($h*(($(select1(n,j,1))+$(select1(n,j,:(size(f)[$j]))))/2+$(select1(n,j,1,:(sum($(select1(n,j,:(2:$(:end)-1),f)),dims=$j))))))
@generated function trapz1(f::Array{T,N} where T,::Val{J},h::Real,s::Tuple=size(f)) where {N,J}
    gentrapz1(N,J)
end
@generated function trapz1(f::Array{T,N} where T,h::D...) where {N,D<:Real}
    Expr(:block,:(s=size(f)),
         [:(i = $(gentrapz1(j,j,:(h[$j]),j≠N ? :i : :f,))) for j ∈ N:-1:1]...)
end
function gentrapz2(n,j,f=:f,d=:(d[$j]))
    z = n≠1 ? :zeros : :zero
    quote
        for k ∈ 1:s[$j]-1
            $(select1(n,j,:k)) = $d[k]*($(select1(n,j,:k,f))+$(select1(n,j,:(k+1),f)))
        end
        $(select1(n,j,:(s[$j]))) = $z(eltype(f),$((:(s[$i]) for i ∈ 1:n if i≠j)...))
        f = $(select1(n,j,1,:(sum(f,dims=$j))))
    end
end
function trapz(f::IntervalMap{B,F,<:IntervalRange} where {B<:AbstractReal,F})
    trapz1(codomain(f),step(points(f)))
end
for N ∈ 2:5
    @eval function trapz(f::ParametricMap{B,F,$N,<:AlignedSpace{$N}} where {B,F})
        trapz1(codomain(f),$([:(step(points(f).v[$j])) for j ∈ 1:N]...))
    end
    @eval function trapz(m::ParametricMap{B,F,$N,T} where {B,F,T},D::AbstractVector=diff.(points(m).v))
        c = codomain(m)
        f,s,d = similar(c),size(c),D./2
        $(Expr(:block,vcat([gentrapz2(j,j,j≠N ? :f : :c).args for j ∈ N:-1:1]...)...))
    end
    for J ∈ 1:N
        @eval function trapz2(c::Array{T,$N} where T,j::Val{$J},D)
            f,s,d = similar(c),size(c),D/2
            $(gentrapz2(N,J,:c,:d))
        end
    end
end

integral(args...) = cumtrapz(args...)
const ∫ = integral

refdiff(x::Global) = ref(x)
refdiff(x) = x[2:end]

isregular(f::IntervalMap) = prod(.!iszero.(fiber(speed(f))))

Grassmann.metric(f::TensorField,g::TensorField) = maximum(fiber(abs(f-g)))
LinearAlgebra.norm(f::IntervalMap,g::IntervalMap) = arclength(f-g)

arctime(f) = inv(arclength(f))
totalarclength(f::IntervalMap) = sum(abs.(diff(codomain(f)),refdiff(metricextensor(f))))
function arclength(f::IntervalMap)
    int = cumsum(abs.(diff(codomain(f)),refdiff(metricextensor(f))))
    pushfirst!(int,zero(eltype(int)))
    TensorField(domain(f), int)
end # cumtrapz(speed(f))
function cumtrapz(f::IntervalMap,d::AbstractVector=diff(points(f)))
    i = (d/2).*cumsum(f.cod[2:end]+f.cod[1:end-1])
    pushfirst!(i,zero(eltype(i)))
    TensorField(domain(f), i)
end
function cumtrapz1(f::Vector,h::Real)
    i = (h/2)*cumsum(f[2:end]+f[1:end-1])
    pushfirst!(i,zero(eltype(i)))
    return i
end
cumtrapz(f::IntervalMap,j::Int) = cumtrapz(f,Val(j))
cumtrapz(f::IntervalMap,j::Val{1}) = cumtrapz(f)
cumtrapz(f::ParametricMap,j::Int) = cumtrapz(f,Val(j))
cumtrapz(f::ParametricMap,j::Val{J}) where J = TensorField(domain(f), cumtrapz2(codomain(f),j,diff(points(f).v[J])))
cumtrapz(f::ParametricMap{B,F,N,<:AlignedSpace{N}} where {B,F,N},j::Val{J}) where J = TensorField(domain(f), cumtrapz1(codomain(f),j,step(points(f).v[J])))
selectzeros(n,j) = :(zeros($([i≠j ? :(s[$i]) : 1 for i ∈ 1:n]...)))
selectzeros2(n,j) = :(zeros($([i≠j ? i<j ? :(s[$i]) : :(s[$i]-1) : 1 for i ∈ 1:n]...)))
gencat(n,j=n,cat=n≠2 ? :cat : j≠2 ? :vcat : :hcat) = :($cat($(selectzeros2(n,j)),$(j≠1 ? gencat(n,j-1) : :i);$((cat≠:cat ? () : (Expr(:kw,:dims,j),))...)))
gencumtrapz1(n,j,h=:h,f=:f) = :(($h/2)*cumsum($(select1(n,j,:(2:$(:end)),f)).+$(select1(n,j,:(1:$(:end)-1),f)),dims=$j))
@generated function cumtrapz1(f::Array{T,N} where T,::Val{J},h::Real,s::Tuple=size(f)) where {N,J}
    :(cat($(selectzeros(N,J)),$(gencumtrapz1(N,J)),dims=$J))
end
@generated function cumtrapz1(f::Array{T,N} where T,h::D...) where {N,D<:Real}
    Expr(:block,:(s=size(f)),
         [:(i = $(gencumtrapz1(N,j,:(h[$j]),j≠1 ? :i : :f,))) for j ∈ 1:N]...,
        gencat(N))
end
function gencumtrapz2(n,j,d=:(d[$j]),f=j≠1 ? :i : :f)
    quote
        i = cumsum($(select1(n,j,:(2:$(:end)),f)).+$(select1(n,j,:(1:$(:end)-1),f)),dims=$j)
        @threads for k ∈ 1:s[$j]-1
            $(select1(n,j,:k,:i)) .*= $d[k]
        end
    end
end
function cumtrapz(f::IntervalMap{B,F,<:IntervalRange} where {B<:AbstractReal,F})
    TensorField(domain(f), cumtrapz1(codomain(f),step(points(f))))
end
for N ∈ 2:5
    @eval function cumtrapz(f::ParametricMap{B,F,$N,<:AlignedSpace{$N}} where {B,F})
        TensorField(domain(f), cumtrapz1(codomain(f),$([:(step(points(f).v[$j])) for j ∈ 1:N]...)))
    end
    @eval function cumtrapz(m::ParametricMap{B,F,$N,T} where {B,F,T},D::AbstractVector=diff.(points(m).v))
        f = codomain(m)
        s,d = size(f),D./2
        $(Expr(:block,vcat([gencumtrapz2(N,j,:(d[$j])).args for j ∈ 1:N]...)...))
        TensorField(domain(m), $(gencat(N)))
    end
    for J ∈ 1:N
        @eval function cumtrapz2(c::Array{T,$N} where T,::Val{$J})
            s,d = size(f),D/2
            $(gencumtrapz2(N,J,:d,:f))
            cat($(selectzeros(N,J)),i,dims=$J)
        end
    end
end
function linecumtrapz(γ::IntervalMap,f::Function)
    cumtrapz(TensorField(domain(γ),f.(codomain(γ)).⋅codomain(gradient(γ))))
end

# differential geometry

export 𝓛, bracket

struct LieBracket end
struct LieDerivative{X<:VectorField}
    v::X
end
const 𝓛 = LieBracket()

Base.getindex(::LieBracket,X::VectorField) = LieDerivative(X)
Base.getindex(::LieBracket,X::VectorField,Y::VectorField...) = bracket(X,Y...)
(::LieBracket)(X::VectorField,Y::VectorField...) = bracket(X,Y...)
(::LieBracket)(X::VectorField) = LieDerivative(X)
(X::LieDerivative)(Y::VectorField...) = bracket(X.v,Y...)
(X::VectorField)(f::ScalarField) = X⋅gradient(f)
(X::GradedVector)(f::ScalarField) = X⋅gradient(f)
bracket(X,Y) = X(Y) - Y(X)
bracket(X,Y,Z) = X(bracket(Y,Z)) + Y(bracket(Z,X)) + Z(bracket(X,Y))
bracket(W,X,Y,Z) = W(bracket(X,Y,Z)) + X(bracket(W,Z,Y)) + Y(bracket(W,X,Z)) + Z(bracket(W,Y,X))
bracket(V,W,X,Y,Z) = V(bracket(W,X,Y,Z)) + W(bracket(V,X,Z,Y)) + X(bracket(V,W,Y,Z)) + Y(bracket(V,W,Z,X)) + Z(bracket(V,W,X,Y))
𝓛dot(x::Chain,y::Simplex{V}) where V = Chain{V}(Real.(x.⋅value(y)))
function (X::VectorField)(Y::VectorField)
    TensorField(base(X),𝓛dot.(fiber(X),fiber(gradient(Y))))
end

export arclength, arctime, totalarclength, trapz, cumtrapz, linecumtrapz, psum, pcumsum
export centraldiff, tangent, tangent_fast, unittangent, speed, normal, unitnormal
export curvenormal, unitcurvenormal, ribbon, tangentsurface, planecurve
export normalnorm, area, surfacearea, weingarten, gauss
export normalnorm_slow, area_slow, surfacearea_slow, weingarten_slow, gauss_slow
export tangent_slow, normal_slow, unittangent_slow, unitnormal_slow

# use graph for IntervalMap? or RealFunction!
tangent(f::IntervalMap) = gradient(f)
tangent(f::ScalarField) = tangent(graph(f))
tangent(f::VectorField) = det(gradient(f))
normal(f::ScalarField) = ⋆tangent(f)
normal(f::VectorField) = ⋆tangent(f)
unittangent(f::ScalarField,n=tangent(f)) = TensorField(domain(f), codomain(n)./norm.(codomain(n)))
unittangent(f::VectorField,n=tangent(f)) = TensorField(domain(f), codomain(n)./norm.(codomain(n)))
unittangent(f::IntervalMap) = unitgradient(f)
unitnormal(f) = ⋆unittangent(f)
normalnorm(f) = Real(abs(normal(f)))

tangent_slow(f::IntervalMap) = gradient_slow(f)
tangent_slow(f::ScalarField) = tangent_slow(graph(f))
tangent_slow(f::VectorField) = det(gradient_slow(f))
normal_slow(f::ScalarField) = ⋆tangent_slow(f)
normal_slow(f::VectorField) = ⋆tangent_slow(f)
unittangent_slow(f::ScalarField,n=tangent_slow(f)) = TensorField(domain(f), codomain(n)./norm.(codomain(n)))
unittangent_slow(f::VectorField,n=tangent_slow(f)) = TensorField(domain(f), codomain(n)./norm.(codomain(n)))
unittangent_slow(f::IntervalMap) = unitgradient_slow(f)
unitnormal_slow(f) = ⋆unittangent_slow(f)
normalnorm_slow(f) = Real(abs(normal_slow(f)))

ribbon(f::AbstractCurve,g::Vector{<:AbstractCurve}) = TensorField(points(f)⊕LinRange(0,1,length(g)+1),hcat(fiber(f),fiber.(g)...))
ribbon(f::AbstractCurve,g::AbstractCurve,n::Int=100) = tangentsurface(f,g-f,n)
tangentsurface(f::AbstractCurve,g::AbstractCurve,n::Int=100) = ribbon(f,Ref(f).+(LinRange(inv(n),1,n).*Ref(g)))
tangentsurface(f::AbstractCurve,v::Real=1,n::Int=100) = tangentsurface(f,v*tangent(f),n)

area(f::VectorField) = integral(normalnorm(f))
surfacearea(f::VectorField) = integrate(normalnorm(f))
weingarten(f::VectorField) = gradient(unitnormal(f))
gauss(f::VectorField) = Real(abs(det(weingarten(f))))

area_slow(f::VectorField) = integral(normalnorm_slow(f))
surfacearea_slow(f::VectorField) = integrate(normalnorm_slow(f))
weingarten_slow(f::VectorField) = gradient(unitnormal_slow(f))
gauss_slow(f::VectorField) = Real(abs(det(weingarten_slow(f))))

area(f::PlaneCurve) = integral(f∧gradient(f))/2
#volume(f::VectorField) = integral(f∧det(gradient(f)))/3
sectorarea(f::PlaneCurve) = integrate(f∧gradient(f))/2
sectorvolume(f::VectorField) = integrate(f∧det(gradient(f)))/3

function speed(f::IntervalMap,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    TensorField(domain(f), abs.(t))
end
function curvenormal(f::IntervalMap,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    TensorField(domain(f), centraldiff(t,d))
end
function unitcurvenormal(f::IntervalMap,d=centraldiffpoints(f),t=centraldifffiiber(f,d),n=centraldiff(t,d))
    TensorField(domain(f), (n./abs.(n)))
end

export curvature, radius, osculatingplane, unitosculatingplane, binormal, unitbinormal, curvaturebivector, torsion, curvaturetrivector, frame, unitframe, frenet, wronskian, bishoppolar, bishop, curvaturetorsion

function curvature(f::PlaneCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),a=abs.(t))
    TensorField(domain(f), abs.(centraldiff(t./a,d))./a)
end

function curvature(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),a=abs.(t))
    TensorField(domain(f), abs.(centraldiff(t./a,d))./a)
end
function radius(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),a=abs.(t))
    TensorField(domain(f), a./abs.(centraldiff(t./a,d)))
end
function osculatingplane(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d))
    TensorField(domain(f), t.∧n)
end
function unitosculatingplane(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d))
    TensorField(domain(f), (t./abs.(t)).∧(n./abs.(n)))
end
function binormal(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d))
    TensorField(domain(f), .⋆(t.∧n))
end
function unitbinormal(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d))
    a = t./abs.(t)
    n = centraldiff(t./abs.(t),d)
    TensorField(domain(f), .⋆(a.∧(n./abs.(n))))
end
function curvaturebivector(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d))
    a=abs.(t); ut=t./a
    TensorField(domain(f), abs.(centraldiff(ut,d))./a.*(ut.∧(n./abs.(n))))
end
function torsion(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d),b=t.∧n)
    TensorField(domain(f), (b.∧centraldiff(n,d))./abs.(.⋆b).^2)
end
function curvaturetrivector(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d),b=t.∧n)
    a=abs.(t); ut=t./a
    TensorField(domain(f), (abs.(centraldiff(ut,d)./a).^2).*(b.∧centraldiff(n,d))./abs.(.⋆b).^2)
end
#torsion(f::TensorField,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d),a=abs.(t)) = TensorField(domain(f), abs.(centraldiff(.⋆((t./a).∧(n./abs.(n))),d))./a)
function frame(f::PlaneCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d))
    TensorField(domain(f), TensorOperator.(Chain.(t,n)))
end
function frame(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d))
    TensorField(domain(f), TensorOperator.(Chain.(t,n,.⋆(t.∧n))))
end
function unitframe(f::PlaneCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d))
    TensorField(domain(f), TensorOperator.(Chain.(t./abs.(t),n./abs.(n))))
end
function unitframe(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d))
    b = .⋆(t.∧n)
    TensorField(domain(f), TensorOperator.(Chain.(t./abs.(t),n./abs.(n),b./abs.(b))))
end
function frenet(f::PlaneCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d))
    TensorField(domain(f), TensorOperator.(centraldiff(Chain.(t./abs.(t),n./abs.(n)),d)))
end
function frenet(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d))
    (ut,un)=(t./abs.(t),n./abs.(n))
    TensorField(domain(f), TensorOperator.(centraldiff(Chain.(ut,un,.⋆(ut.∧un)),d)))
end
function wronskian(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d))
    TensorField(domain(f), f.cod.∧t.∧n)
end

function frame(f::VectorField{<:Coordinate{<:Chain},<:Chain,2,<:RealSpace{2}})#,d=centraldiff(points(f)),t=centraldiff(codomain(f),d),n=centraldiff(t,d))
    Ψ = gradient(f)
    Ψu,Ψv = getindex.(Ψ,1),getindex.(Ψ,2)
    ξ3 = ⋆(Ψu∧Ψv)
    TensorField(domain(f), TensorOperator.(Chain.(fiber(Ψu),fiber(ξ3),fiber(⋆(ξ3∧Ψu)))))
end
function unitframe(f::VectorField{<:Coordinate{<:Chain},<:Chain,2,<:RealSpace{2}})#,d=centraldiff(points(f)),t=centraldiff(codomain(f),d),n=centraldiff(t,d))
    Ψ = gradient(f)
    Ψu,Ψv = getindex.(Ψ,1),getindex.(Ψ,2)
    ξ3 = ⋆(Ψu∧Ψv)
    ξ2 = ⋆(ξ3∧Ψu)
    TensorField(domain(f), TensorOperator.(Chain.(fiber(Ψu/abs(Ψu)),fiber(ξ3/abs(ξ3)),fiber(ξ2/abs(ξ2)))))
end

#???
function compare(f::TensorField)#???
    d = centraldiffpoints(f)
    t = centraldifffiber(f,d)
    n = centraldiff(t,d)
    TensorField(domain(f), centraldiff(t./abs.(t)).-n./abs.(t))
end #????

function curvaturetorsion(f::SpaceCurve,d=centraldiffpoints(f),t=centraldifffiber(f,d),n=centraldiff(t,d),b=t.∧n)
    a = abs.(t)
    TensorField(domain(f), Chain.(value.(abs.(centraldiff(t./a,d))./a),getindex.((b.∧centraldiff(n,d))./abs.(.⋆b).^2,1)))
end

function bishoppolar(f::SpaceCurve,κ=value.(codomain(curvature(f))))
    d = diff(points(f))
    τs = getindex.(codomain(torsion(f)).*codomain(speed(f)),1)
    θ = (d/2).*cumsum(τs[2:end]+τs[1:end-1])
    pushfirst!(θ,zero(eltype(θ)))
    TensorField(domain(f), Chain.(κ,θ))
end
function bishop(f::SpaceCurve,κ=value.(codomain(curvature(f))))
    d = diff(points(f))
    τs = getindex.(codomain(torsion(f)).*codomain(speed(f)),1)
    θ = (d/2).*cumsum(τs[2:end]+τs[1:end-1])
    pushfirst!(θ,zero(eltype(θ)))
    TensorField(domain(f), Chain.(κ.*cos.(θ),κ.*sin.(θ)))
end
#bishoppolar(f::TensorField) = TensorField(domain(f), Chain.(value.(codomain(curvature(f))),getindex.(codomain(cumtrapz(torsion(f))),1)))
#bishop(f::TensorField,κ=value.(codomain(curvature(f))),θ=getindex.(codomain(cumtrapz(torsion(f))),1)) = TensorField(domain(f), Chain.(κ.*cos.(θ),κ.*sin.(θ)))

function planecurve(κ::RealFunction,φ::Real=0.0)
    int = iszero(φ) ? integral(κ) : integral(κ)+φ
    integral(Chain.(cos(int),sin(int)))
end

export surfacemetric, surfaceframe, firstkind, secondkind, geodesicsystem, applymetric

surfacemetric(dom::ScalarField,f::Function) = surfacemetric(TensorField(dom,f))
function surfacemetric(t::ScalarField)
    g = gradient(t); V = Submanifold(MetricTensor([1 1; 1 1]))
    dfdx,dfdy = getindex.(g,1),getindex.(g,2)
    g1 = (1+dfdx*dfdx)*Λ(V).v1+(dfdx*dfdy)*Λ(V).v2
    g2 = (1+dfdy*dfdy)*Λ(V).v2+(dfdx*dfdy)*Λ(V).v1;
    GridFrameBundle(points(t), Outermorphism.(Chain{V}.(fiber(g1),fiber(g2))))
end

surfacemetric(dom,f::Function) = surfacemetric(TensorField(dom,f))
function surfacemetric(t) # TensorField(M, torus.(points(M)))
    g = gradient(t); V = Submanifold(MetricTensor([1 1; 1 1]))
    dfdx,dfdy = getindex.(g,1),getindex.(g,2)
    E,F,G = Real(dfdx⋅dfdx),Real(dfdx⋅dfdy),Real(dfdy⋅dfdy)
    g1 = E*Λ(V).v1+F*Λ(V).v2
    g2 = G*Λ(V).v2+F*Λ(V).v1;
    GridFrameBundle(points(t), Outermorphism.(Chain{V}.(fiber(g1),fiber(g2))))
end

function surfaceframe(t)
    g = getindex.(metricextensor(t),1); V = Submanifold(MetricTensor([1 1; 1 1]))
    E,F,G = getindex.(g,1,1),getindex.(g,1,2),getindex.(g,2,2)
    F2 = F.*F; mag,sig = sqrt.((E.*E).+F2), sign.(F2.-(E.*G))
    TensorOperator.(Chain.(Chain.(E,F)./mag,Chain.(F,.-E)./(sig.*mag)))
end

_firstkind(dg,k,i,j) = dg[k,j][i] + dg[i,k][j] - dg[i,j][k]
firstkind(g::TensorField) = TensorField(base(g),TensorOperator.(firstkind.(d(g)/2)))
@generated function firstkind(dg,i,j,k)
    Expr(:call,:+,[:(_firstkind(dg,$l,i,j)) for l ∈ list(1,mdims(fibertype(dg)))]...)
end
@generated function firstkind(dg,i,j)
    Expr(:call,:Chain,[:(firstkind(dg,i,j,$k)) for k ∈ list(1,mdims(fibertype(dg)))]...)
end
@generated function firstkind(dg,j)
    Expr(:call,:Chain,[:(firstkind(dg,$i,j)) for i ∈ list(1,mdims(fibertype(dg)))]...)
end
@generated function firstkind(dg)
    Expr(:call,:Chain,[:(firstkind(dg,$j)) for j ∈ list(1,mdims(fibertype(dg)))]...)
end

secondkind(g) = TensorField(base(g),TensorOperator.(secondkind.(inv.(g),d(g)/2)))
@generated function secondkind(ig,dg,i,j,k)
    Expr(:call,:+,[:(ig[k,$l]*_firstkind(dg,$l,i,j)) for l ∈ list(1,mdims(fibertype(dg)))]...)
end
@generated function secondkind(ig,dg,i,j)
    Expr(:call,:Chain,[:(secondkind(ig,dg,i,j,$k)) for k ∈ list(1,mdims(fibertype(dg)))]...)
end
@generated function secondkind(ig,dg,j)
    Expr(:call,:Chain,[:(secondkind(ig,dg,$i,j)) for i ∈ list(1,mdims(fibertype(dg)))]...)
end
@generated function secondkind(ig,dg)
    Expr(:call,:Chain,[:(secondkind(ig,dg,$j)) for j ∈ list(1,mdims(fibertype(dg)))]...)
end

geodesicsystem(x::Chain,Γ) = Chain(x,-geodesic(x,Γ))
geodesicsystem(x::Chain,Γ,g) = Chain(applymetric(x,g),applymetric(-geodesic(x,Γ),g))
geodesicsystem(x::Chain,Γ::TensorField) = geodesicsystem(x,Γ(x))
geodesicsystem(x::Chain,Γ::TensorField,g::TensorField) = geodesicsystem(x,Γ(x),g(x))
@generated function geodesic(x::Chain{V,G,T,N} where {V,G,T},Γ) where N
    Expr(:call,:+,vcat([[:(Γ[$i,$j]*(x[$i]*x[$j])) for i ∈ list(1,N)] for j ∈ list(1,N)]...)...)
end
@generated function metricscale(x::Chain{V,G,T,N} where {G,T},g::Simplex) where {V,N}
    Expr(:call,:(Chain{V}),[:(x[$k]*sqrt(g[$k,$k])) for k ∈ list(1,N)]...)
end

#export beta, betafunction

function beta(a,b,n=30000)
    x = OpenParameter(n)
    integrate(x^(a-1)*(1-x)^(b-1))
end
function betafunction(a,b,n=100)
    x = OpenParameter(n)
    integral(x^(a-1)*(1-x)^(b-1))
end

