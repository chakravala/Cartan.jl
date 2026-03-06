
#   This file is part of Cartan.jl
#   It is licensed under the AGPL license
#   Cartan Copyright (C) 2025 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com

export FourierSpace
struct FourierSpace{T,F<:AbstractVector{T},G} <: AbstractVector{T}
    f::F
    v::G
end

Base.size(f::FourierSpace) = size(f.f)
Base.getindex(f::FourierSpace,i::Int) = f.f[i]
invdim(f::FourierSpace,dims=1) = length(f.v)
invdim(f::ProductSpace,dims=1) = invdim(f.v[dims])
invdim(f::AbstractVector,dims=1) = length(f)
isfourier(x::FourierSpace) = true
isfourier(x::ProductSpace) = prod(isfourier.(split(x)))
isfourier(x::FiberBundle) = isfourier(points(x))
isfourier(x) = false

export fftspace, rfftspace, r2rspace
import AbstractFFTs: fftfreq, rfftfreq, fftshift, ifftshift
for fun ∈ (:fftspace,:r2rspace)
    @eval begin
        $fun(t::TensorField) = TensorField($fun(base(t)))
        $fun(x::GridBundle) = GridBundle($fun(points(x)))
        $fun(x::ProductSpace{V}) where V = ProductSpace{V}($fun.(x.v))
    end
end
rfftspace(t::TensorField) = TensorField(rfftspace(base(t)))
rfftspace(x::GridBundle) = GridBundle(rfftspace(points(x)))
rfftspace(x::ProductSpace{V}) where V = ProductSpace{V}(rfftspace(x.v[2]),fftspace.(x.v[2:end])...)
r2rspace(t::TensorField,kind) = TensorField(r2rspace(base(t),kind))
r2rspace(x::GridBundle,kind) = GridBundle(r2rspace(points(x),kind))
r2rspace(x::ProductSpace{V},kind) where V = ProductSpace{V}(r2rspace.(x.v,kind))

GridBundle(x::FourierSpace) = GridBundle(x,ClampedTopology(size(x)))
rfftspace(N::Real,ω=1/N) = rfftfreq(N,N*ω)
rfftspace(x::AbstractRange) = FourierSpace(rfftspace(length(x),2π/(x[end]-x[1])),x)
rfftspace(x::FourierSpace) = x.v
rfftspace(x::Frequencies) = Base.OneTo(length(x))
fftspace(x::AbstractRange) = FourierSpace(fftspace(length(x),2π/(x[end]-x[1])),x)
function fftspace(N::Real,ω=1/N)
    n=(2(N-1)+iseven(N))
    (n/N)*rfftfreq(n,N*ω)
end
fftspace(x::FourierSpace) = x.v
fftspace(x::Frequencies) = Base.OneTo(length(x))

r2rspace(N::Real,ω::Float64=1/N) = fftspace(N,ω)
r2rspace(x::AbstractRange) = FourierSpace(r2rspace(length(x),π/(x[end]-x[1])),x)
r2rspace(x::FourierSpace) = x.v
r2rspace(x::Frequencies) = Base.OneTo(length(x))

function r2rspace(N::Real,kind::Int,fs=1)
    out = r2rspace(N,1/fs)
    kind ∈ (9,6,10) ? out .+ out[2] : out
end
r2rspace(x::AbstractRange,kind) = FourierSpace(r2rspace(length(x),kind,(x[end]-x[1])/π),x)
r2rspace(x::FourierSpace,kind) = x.v
r2rspace(x::Frequencies,kind) = Base.OneTo(length(x))

fftshiftalias(x) = fftshift(x).-x[Int(ceil(length(x)/2))]
fftshift(x::FourierSpace) = FourierSpace(fftshiftalias(x.f),x.v)
ifftshift(x::FourierSpace) = fftspace(x.v)
for fun ∈ (:fftshift,:ifftshift)
    @eval begin
        $fun(t::TensorField) = TensorField($fun(base(t)),$fun(fiber(t)))
        $fun(x::GridBundle) = GridBundle($fun(points(x)))
        $fun(x::ProductSpace{V}) where V = ProductSpace{V}($fun.(x.v))
    end
end
for fun ∈ (:fft,:fft!,:ifft,:ifft!,:bfft,:bfft!)
    @eval AbstractFFTs.$fun(t::TensorField,args...) = TensorField(fftspace(base(t)), $fun(fiber(t),args...))
end
for fun ∈ (:rfft,)
    @eval AbstractFFTs.$fun(t::TensorField,args...) = TensorField(rfftspace(base(t)), $fun(fiber(t),args...))
end
for fun ∈ (:irfft,:brfft)
    @eval begin
        AbstractFFTs.$fun(t::TensorField) = TensorField(rfftspace(base(t)), $fun(fiber(t),invdim(points(t))))
        AbstractFFTs.$fun(t::TensorField,dims) = TensorField(rfftspace(base(t)), $fun(fiber(t),invdim(points(t),dims[1]),dims))
    end
end

flt(f::TensorField,σ::Number) = fft(exp((-σ)*TensorField(base(f)))*f)
bflt(f::TensorField,σ::Number) = bfft(exp((-σ)*TensorField(base(f)))*f)
rflt(f::TensorField,σ::Number) = rfft(exp((-σ)*TensorField(base(f)))*f)
brflt(f::TensorField,σ::Number) = brfft(exp((-σ)*TensorField(base(f)))*f)
iflt(f::TensorField,σ::Number) = ifft(exp(σ*TensorField(base(f)))*f)
irflt(f::TensorField,σ::Number) = irfft(exp(σ*TensorField(base(f)))*f)

export flt, bflt, rflt, brflt, iflt, irflt
function flt(f::TensorField,σ::AbstractVector)
    t = TensorField(base(f))
    out = Matrix{Complex{Float64}}(undef,length(σ),length(t))
    for i ∈ 1:length(σ)
        out[i,:] = fiber(fft(exp((-fiber(σ)[i])*t)*f))
    end
    return TensorField(σ⊕fftspace(base(f)),out)
end
function bflt(f::TensorField,σ::AbstractVector)
    t = TensorField(base(f))
    out = Matrix{Complex{Float64}}(undef,length(σ),length(t))
    for i ∈ 1:length(σ)
        out[i,:] = fiber(bfft(exp((-fiber(σ)[i])*t)*f))
    end
    return TensorField(σ⊕fftspace(base(f)),out)
end
function rflt(f::TensorField,σ::AbstractVector)
    t = TensorField(base(f))
    out = Matrix{Complex{Float64}}(undef,length(σ),length(t))
    for i ∈ 1:length(σ)
        out[i,:] = fiber(rfft(exp((-fiber(σ)[i])*t)*f))
    end
    return TensorField(σ⊕rfftspace(base(f)),out)
end
function brflt(f::TensorField,σ::AbstractVector)
    t = TensorField(base(f))
    id = length(t)
    out = Matrix{Complex{Float64}}(undef,length(σ),id)
    for i ∈ 1:length(σ)
        out[i,:] = fiber(brfft(exp((-fiber(σ)[i])*t)*f,id))
    end
    return TensorField(σ⊕rfftspace(base(f)),out)
end

function iflt(f::TensorField)
    σ = TensorField(points(f).v[1])
    t = TensorField(fftspace(points(f).v[2]))
    out = Matrix{Complex{Float64}}(undef,length(σ),length(t))
    for i ∈ 1:length(σ)
        out[i,:] = exp.(fiber(σ)[i]*fiber(t)).*ifft(view(fiber(f),i,:))
    end
    return TensorField(base(t),vec(sum(out;dims=1))/length(σ))
end
function irflt(f::TensorField)
    σ = TensorField(points(f).v[1])
    t = TensorField(rfftspace(points(f).v[2]))
    id = length(t)
    out = Matrix{Complex{Float64}}(undef,length(σ),id)
    for i ∈ 1:length(σ)
        out[i,:] = exp.(fiber(σ)[i]*fiber(t)).*irfft(view(fiber(f),i,:),id)
    end
    return TensorField(base(t),vec(sum(out;dims=1))/length(σ))
end

export fgt, bfgt, rfgt, brfgt
for fun ∈ (:fgt, :bfgt, :rfgt, :brfgt)
    @eval $fun(f::TensorField,σ::Int,g) = $fun(f,resample(points(f),σ),g)
end
function fgt(f::TensorField,σ::AbstractVector,g)
    t = TensorField(base(f))
    out = Matrix{Complex{Float64}}(undef,length(σ),length(t))
    for i ∈ 1:length(σ)
        out[i,:] = fiber(fft(g(t-fiber(σ)[i])*f))
    end
    return TensorField(σ⊕fftspace(base(f)),out)
end
function bfgt(f::TensorField,σ::AbstractVector,g)
    t = TensorField(base(f))
    out = Matrix{Complex{Float64}}(undef,length(σ),length(t))
    for i ∈ 1:length(σ)
        out[i,:] = fiber(bfft(g(t-fiber(σ)[i])*f))
    end
    return TensorField(σ⊕fftspace(base(f)),out)
end
function rfgt(f::TensorField,σ::AbstractVector,g)
    t = TensorField(base(f))
    out = Matrix{Complex{Float64}}(undef,length(σ),length(t))
    for i ∈ 1:length(σ)
        out[i,:] = fiber(rfft(g(t-fiber(σ)[i])*f))
    end
    return TensorField(σ⊕rfftspace(base(f)),out)
end
function brfgt(f::TensorField,σ::AbstractVector,g)
    t = TensorField(base(f))
    id = length(t)
    out = Matrix{Complex{Float64}}(undef,length(σ),id)
    for i ∈ 1:length(σ)
        out[i,:] = fiber(brfft(g(t-fiber(σ)[i])*f,id))
    end
    return TensorField(σ⊕rfftspace(base(f)),out)
end

export OrthogonalTransform, seriestransform
export FourierCosine, FourierSine, ChebyshevFirst, ChebyshevSecond

struct OrthogonalTransform{F,T}
    f::F
    a::T
    b::T
end

const FourierCosine = OrthogonalTransform((n,x)->cos(n*x),0.0,float(π))
const FourierSine = OrthogonalTransform((n,x)->sin((n+1)*x),0.0,float(π))
const ChebyshevFirst = OrthogonalTransform((n,x)->cos(n*acos(x)),-1.0,1.0)
const ChebyshevSecond = OrthogonalTransform((n,x)->(θ=acos(x);iszero(θ) ? one(θ) : sin((n+1)*θ)/sin(θ)),-1.0,1.0)

(ot::OrthogonalTransform)(n::Int,x) = ot.f(n,x)
(ot::OrthogonalTransform)(n::Int,x::TensorField) = TensorField(base(x),ot.f.(n,fiber(x)))

function (f::OrthogonalTransform)(g::AbstractVector,N=length(g)) # first N coefficients of g
    a,b = f.a,f.b
    if isfourier(points(g)) # restore
        x = TensorField(fftspace(base(g)))
        ωx = ((b-a)/interval_scale(x))*x + a
        out = fiber(g)[1]*f(0,ωx)
        for n ∈ 2:N
            out += fiber(g)[n]*f(n-1,ωx)
        end
        return out
    else # transform to series coefficients
        L = interval_scale(g) # T/2
        ω = FourierSpace(f,points(g),N)
        ωx = ((b-a)/L)*TensorField(base(g)) + a
        TensorField(ω,[(2/L)*integrate(g*f(n,ωx)) for n in 0:N-1])
    end
end

function (f::OrthogonalTransform)(g::AbstractMatrix,N=size(g)[1],M=size(g)[2])
    a,b = f.a,f.b
    isf = isfourier(points(g))
    XY = TensorField(isf ? fftspace(base(g)) : base(g))
    L = value(interval_scale(XY))
    ωx,ωy = ((b-a)./L).*split(XY).+a
    if isf # restore
        out = 0*ωx
        for i ∈ 1:N, j ∈ 1:M
            out += fiber(g)[i,j]*f(i-1,ωx)*f(j-1,ωy)
        end
        return out
    else # transform to series coefficients
        TensorField(FourierSpace(f,points(g),N,M),[(4/prod(L))*
        integrate(g*f(i,ωx)*f(j,ωy)) for i in 0:N-1, j in 0:M-1])
    end
end

function (f::OrthogonalTransform)(g::AbstractArray{T,3} where T,N=size(g)[1],M=size(g)[2],A=size(g)[3])
    a,b = f.a,f.b
    isf = isfourier(points(g))
    XYZ = TensorField(isf ? fftspace(base(g)) : base(g))
    L = value(interval_scale(XYZ))
    ωx,ωy,ωz = ((b-a)./L).*split(XYZ).+a
    if isf # restore
        out = 0*ωx
        for i ∈ 1:N, j ∈ 1:M, k ∈ 1:A
            out += fiber(g)[i,j,k]*f(i-1,ωx)*f(j-1,ωy)*f(k-1,ωz)
        end
        return out
    else # transform to series coefficients
        TensorField(FourierSpace(f,points(g),N,M,A),[(8/prod(L))*
        integrate(g*f(i,ωx)*f(j,ωy)*f(k,ωz)) for i in 0:N-1, j in 0:M-1, k ∈ 0:A-1])
    end
end

function (f::OrthogonalTransform)(g::AbstractArray{T,4} where T,N=size(g)[1],M=size(g)[2],A=size(g)[3],B=size(g)[4])
    a,b = f.a,f.b
    isf = isfourier(points(g))
    XYZ = TensorField(isf ? fftspace(base(g)) : base(g))
    L = value(interval_scale(XYZ))
    ωx,ωy,ωz,ωu = ((b-a)./L).*split(XYZ).+a
    if isf # restore
        out = 0*ωx
        for i ∈ 1:N, j ∈ 1:M, k ∈ 1:A, l ∈ 1:B
            out += fiber(g)[i,j,k]*f(i-1,ωx)*f(j-1,ωy)*f(k-1,ωz)*f(l-1,ωu)
        end
        return out
    else # transform to series coefficients
        TensorField(FourierSpace(f,points(g),N,M,A,B),[(16/prod(L))*
        integrate(g*f(i,ωx)*f(j,ωy)*f(k,ωz)*f(l,ωu)) for i in 0:N-1, j in 0:M-1, k ∈ 0:A-1, l ∈ 0:B-1])
    end
end

function (f::OrthogonalTransform)(g::AbstractArray{T,5} where T,N=size(g)[1],M=size(g)[2],A=size(g)[3],B=size(g)[4],C=size(g)[5])
    a,b = f.a,f.b
    isf = isfourier(points(g))
    XYZ = TensorField(isf ? fftspace(base(g)) : base(g))
    L = value(interval_scale(XYZ))
    ωx,ωy,ωz,ωu,ωw = ((b-a)./L).*split(XYZ).+a
    if isf # restore
        out = 0*ωx
        for i ∈ 1:N, j ∈ 1:M, k ∈ 1:A, l ∈ 1:B, o ∈ 1:C
            out += fiber(g)[i,j,k]*f(i-1,ωx)*f(j-1,ωy)*f(k-1,ωz)*f(l-1,ωu)*f(o-1,ωw)
        end
        return out
    else # transform to series coefficients
        TensorField(FourierSpace(f,points(g),N,M,A,B,C),[(32/prod(L))*
        integrate(g*f(i,ωx)*f(j,ωy)*f(k,ωz)*f(l,ωu)*f(o,ωw)) for i in 0:N-1, j in 0:M-1, k ∈ 0:A-1, l ∈ 0:B-1, o ∈ 0:C-1])
    end
end

FourierSpace(f::OrthogonalTransform,x::ProductSpace) = FourierSpace(f,x,size(x)...)
function FourierSpace(f::OrthogonalTransform,x::ProductSpace{V},args...) where V
    ProductSpace{V}(FourierSpace.(Ref(f),split(x),args))
end
function FourierSpace(f::OrthogonalTransform,x::AbstractVector,N=length(x))
    FourierSpace(((f.b-f.a)/interval_scale(x))*(0:N-1),x)
end

export Chebyshev, ChebyshevMatrix, ChebyshevVector, chebyshevfft, chebyshevifft, unitpoints

struct Chebyshev{T,A<:AbstractVector} <: DenseVector{T}
    v::Vector{T}
    a::A
end

function Chebyshev(N::Int)
    θ = (π/(N-1))*(0:N-1)
    x = .-cos.(θ)
    Chebyshev(x,θ)
end
function Chebyshev(x::AbstractVector)
    c = Chebyshev(length(x))
    p = (points(c).+1)*((x[end]-x[1])/2).+x[1]
    Chebyshev(p,angle(c))
end

points(t::Chebyshev) = t.v
function unitpoints(t::Chebyshev)
    x = points(t)
    (x.-x[1])*(2/(x[end]-x[1])).-1
end
Base.angle(t::Chebyshev) = t.a
Base.getindex(t::Chebyshev,i::Integer) = getindex(points(t),i)
Base.size(t::Chebyshev) = size(points(t))

resample(m::Chebyshev,i::NTuple{1,Int}) = resample(m,i...)
resample(m::Chebyshev,i::Int=length(m)) = LinRange(m[1],m[end],i)

#ChebyshevVector(x::TensorField) = ChebyshevVector(points(x))
#ChebyshevVector(x::Chebyshev,N=length(x)) = vcat(0,reverse(inv(ChebyshevMatrix(-unitpoints(x))[1:N-1,1:N-1])[1,:]))*(interval_scale(x)/2)
#ChebyshevVector(N::Int) = vcat(0,reverse(inv(ChebyshevMatrix(N)[1:N-1,1:N-1])[1,:]))

ChebyshevVector(x,N=length(x)) = vcat(0,reverse(inv(ChebyshevMatrix(x)[1:N-1,1:N-1])[1,:]))
ChebyshevVector(N::Int) = vcat(0,reverse(inv(ChebyshevMatrix(N)[1:N-1,1:N-1])[1,:]))
ChebyshevMatrix(x::TensorField) = ChebyshevMatrix(points(x))
ChebyshevMatrix(x::Chebyshev) = ChebyshevMatrix(-points(x))
ChebyshevMatrix(N::Int) = iszero(N) ? [0;;] : ChebyshevMatrix(Chebyshev(N))
function ChebyshevMatrix(x) # differentiation matrix
    N = length(x)-1
    c = vcat(2,ones(N-1),2).*(-1).^(0:N)
    X = repeat(x,1,N+1)
    D = (c*inv.(c)')./((X-X')+I) # off-diagonal entries
    D-Diagonal(vec(sum(D,dims=2))) # diagonal entries
end

chebyshevfft(v::TensorField) = chebyshevfft(fiber(v))
chebyshevfft(v::TensorField,i::Int) = chebyshevfft(fiber(v),i)
chebyshevfft(v::AbstractVector) = fft(vcat(v,reverse(v[2:length(v)-1])))
function chebyshevfft(v::AbstractMatrix,i::Int)
    N,M = size(v)
    if isone(i)
        fft(vcat(v,reverse(v[2:N-1,:],dims=1)),i)
    else
        fft(hcat(v,reverse(v[:,2:M-1],dims=2)),i)
    end
end
function chebyshevfft(v::AbstractArray{T,3} where T,i::Int)
    N,M,R = size(v)
    if isone(i)
        fft(vcat(v,reverse(v[2:N-1,:,:],dims=1)),i)
    elseif i==2
        fft(hcat(v,reverse(v[:,2:M-1,:],dims=2)),i)
    else
        fft(cat(v,reverse(v[:,:,2:R-1],dims=2),dims=3),i)
    end
end

function chebyshevifft(V::AbstractVector,U::AbstractVector,N)
    ii = 0:N-2
    W = real.(ifft(V))
    w = zeros(N)
    w[2:N-1] = -W[2:N-1]./sqrt.(1.0.-cos.(π*(1:N-2)/(N-1)).^2)
    w[1] = sum((ii.^2).*U[ii.+1])/(N-1) .+ (0.5(N-1))*U[N]
    w[N] = sum((-1).^(ii.+1).*(ii.^2).*U[ii.+1])/(N-1) .+ 0.5(N-1)*(-1)^N*U[N]
    return w
end

function chebyshevifft(V::AbstractMatrix,U::AbstractMatrix,i,N,M)
    W = real.(ifft(V,i))
    w = zeros(N,M)
    if isone(i)
        ii = 0:N-2
        x2 = sqrt.(1.0.-cos.(π*(1:N-2)/(N-1)).^2)
        for i ∈ 1:M
            w[2:N-1,i] = -W[2:N-1,i]./x2
            w[1,i] = sum((ii.^2).*U[ii.+1,i])/(N-1) .+ (0.5(N-1))*U[N,i]
            w[N,i] = sum((-1).^(ii.+1).*(ii.^2).*U[ii.+1,i])/(N-1) .+ 0.5(N-1)*(-1)^N*U[N,i]
        end
    else
        ii = 0:M-2
        y2 = sqrt.(1.0.-cos.(π*(1:M-2)/(M-1)).^2)
        for i ∈ 1:N
            w[i,2:M-1] = -W[i,2:M-1]./y2
            w[i,1] = sum((ii.^2).*U[i,ii.+1])/(M-1) .+ (0.5(M-1))*U[i,M]
            w[i,M] = sum((-1).^(ii.+1).*(ii.^2).*U[i,ii.+1])/(M-1) .+ 0.5(M-1)*(-1)^M*U[i,M]
        end
    end
    return w
end

function chebyshevifft(V::AbstractArray{T,3} where T,U::AbstractArray{T,3} where T,i,N,M,R)
    W = real.(ifft(V,i))
    w = zeros(N,M,R)
    if isone(i)
        ii = 0:N-2
        x2 = sqrt.(1.0.-cos.(π*(1:N-2)/(N-1)).^2)
        for i ∈ 1:M
            for j ∈ 1:R
                w[2:N-1,i,j] = -W[2:N-1,i,j]./x2
                w[1,i,j] = sum((ii.^2).*U[ii.+1,i,j])/(N-1) .+ (0.5(N-1))*U[N,i,j]
                w[N,i,j] = sum((-1).^(ii.+1).*(ii.^2).*U[ii.+1,i,j])/(N-1) .+ 0.5(N-1)*(-1)^N*U[N,i,j]
            end
        end
    elseif i==2
        ii = 0:M-2
        y2 = sqrt.(1.0.-cos.(π*(1:M-2)/(M-1)).^2)
        for i ∈ 1:N
            for j ∈ 1:R
                w[i,2:M-1,j] = -W[i,2:M-1,j]./y2
                w[i,1,j] = sum((ii.^2).*U[i,ii.+1,j])/(M-1) .+ (0.5(M-1))*U[i,M,j]
                w[i,M,j] = sum((-1).^(ii.+1).*(ii.^2).*U[i,ii.+1,j])/(M-1) .+ 0.5(M-1)*(-1)^M*U[i,M,j]
            end
        end
    else
        ii = 0:R-2
        y2 = sqrt.(1.0.-cos.(π*(1:R-2)/(R-1)).^2)
        for i ∈ 1:N
            for j ∈ 1:M
                w[i,j,2:R-1] = -W[i,j,2:R-1]./y2
                w[i,j,1] = sum((ii.^2).*U[i,j,ii.+1])/(R-1) .+ (0.5(R-1))*U[i,j,R]
                w[i,j,R] = sum((-1).^(ii.+1).*(ii.^2).*U[i,j,ii.+1])/(R-1) .+ 0.5(R-1)*(-1)^R*U[i,j,R]
            end
        end
    end
    return w
end

function chebyshevifft2(V1::AbstractVector,V2::AbstractVector,U::AbstractVector,N)
    W1 = real.(ifft(V1))
    W2 = real.(ifft(V2))
    u = zeros(N)
    ii = 2:N-1
    x = cos.(π*(1:N-2)/(N-1))
    x2 = sqrt.(1.0.-x.^2)
    u[ii] = W2[ii]./x2 - x.*W1[ii]./x2.^(3/2)
    return u
end

function chebyshevifft2(V1::AbstractMatrix,V2::AbstractMatrix,U::AbstractMatrix,i,N,M)
    W1 = real.(ifft(V1,i))
    W2 = real.(ifft(V2,i))
    u = zeros(N,M)
    if isone(i)
        ii = 2:N-1
        x = cos.(π*(1:N-2)/(N-1))
        x2 = sqrt.(1.0.-x.^2)
        for i ∈ 1:M
            u[ii,i] = W2[ii,i]./x2 - x.*W1[ii,i]./x2.^(3/2)
        end
    else
        ii = 2:M-1
        y = cos.(π*(1:M-2)/(M-1))
        y2 = sqrt.(1.0.-y.^2)
        for i ∈ 1:N
            u[i,ii] = W2[i,ii]./y2 - y.*W1[i,ii]./y2.^(3/2)
        end
    end
    return u
end

function chebyshevifft2(V1::AbstractArray{T,3} where T,V2::AbstractArray{T,3} where T,U::AbstractArray{T,3} where T,i,N,M,R)
    W1 = real.(ifft(V1,i))
    W2 = real.(ifft(V2,i))
    u = zeros(N,M,R)
    if isone(i)
        ii = 2:N-1
        x = cos.(π*(1:N-2)/(N-1))
        x2 = sqrt.(1.0.-x.^2)
        for i ∈ 1:M
            for j ∈ 1:R
                u[ii,i,j] = W2[ii,i,j]./x2 - x.*W1[ii,i,j]./x2.^(3/2)
            end
        end
    elseif i==2
        ii = 2:M-1
        y = cos.(π*(1:M-2)/(M-1))
        y2 = sqrt.(1.0.-y.^2)
        for i ∈ 1:N
            for j ∈ 1:R
                u[i,ii,j] = W2[i,ii,j]./y2 - y.*W1[i,ii,j]./y2.^(3/2)
            end
        end
    else
        ii = 2:R-1
        z = cos.(π*(1:R-2)/(R-1))
        z2 = sqrt.(1.0.-z.^2)
        for i ∈ 1:N
            for j ∈ 1:M
                u[i,j,ii] = W2[i,j,ii]./z2 - z.*W1[i,j,ii]./z2.^(3/2)
            end
        end
    end
    return u
end

function resample_sinc(v::AbstractVector,n)
    N = length(v)
    x = points(v)
    h = step(x)
    xx = resample(x,n)
    xh,xxh = x/h,xx/h
    p = zeros(length(xx))
    for i ∈ 1:N
        p += fiber(v)[i]*sinc.((xxh.-xh[i]))
    end
    return TensorField(xx,p)
end

resample_sinc(v::AbstractMatrix,n,m::Int) = resample_sinc(resample_sinc(v,n,Val(1)),m,Val(2))
resample_sinc(v::AbstractArray{T,3} where T,n,m,o) = resample_sinc(resample_sinc(resample_sinc(v,n,Val(1)),m,Val(2)),o,Val(3))
resample_sinc(v::AbstractArray{T,4} where T,n,m,o,p) = resample_sinc(resample_sinc(resample_sinc(resample_sinc(v,n,Val(1)),m,Val(2)),o,Val(3)),p,Val(4))
resample_sinc(v::AbstractArray{T,5} where T,n,m,o,p,q) = resample_sinc(resample_sinc(resample_sinc(resample_sinc(resample_sinc(v,n,Val(1)),m,Val(2)),o,Val(3)),p,Val(4)),q,Val(5))
function resample_sinc(v::AbstractArray{T,N} where T,n,Q::Val{q}) where {N,q}
    M = size(v)
    M[q] == n && (return v)
    M2 = size_new(Q,n,M...)
    x = split(points(v))[q]
    h = step(x)
    xy = resample(points(v),M2)
    xh,xxh = x/h,split(xy)[q]/h
    p = zeros(M2...)
    for i ∈ 1:M[q]
        sx = sinc.(xxh.-xh[i])
        for j ∈ 1:M[isone(q) ? 2 : 1]
            if N==2
                if isone(q)
                    p[:,j] .+= fiber(v)[i,j].*sx
                else
                    p[j,:] .+= fiber(v)[j,i].*sx
                end
            else for k ∈ 1:M[q==3 ? 2 : 3]
                if N==3
                    if isone(q)
                        p[:,j,k] .+= fiber(v)[i,j,k].*sx
                    elseif q==2
                        p[j,:,k] .+= fiber(v)[j,i,k].*sx
                    else
                        p[j,k,:] .+= fiber(v)[j,k,i].*sx
                    end
                else for l ∈ 1:M[q==4 ? 3 : 4]
                    if N==4
                        if isone(q)
                            p[:,j,k,l] .+= fiber(v)[i,j,k,l].*sx
                        elseif q==2
                            p[j,:,k,l] .+= fiber(v)[j,i,k,l].*sx
                        elseif q==3
                            p[j,k,:,l] .+= fiber(v)[j,k,i,l].*sx
                        else
                            p[j,k,l,:] .+= fiber(v)[j,k,l,i].*sx
                        end
                    else for m ∈ 1:M[q==5 ? 4 : 5]
                        if isone(q)
                            p[:,j,k,l,m] .+= fiber(v)[i,j,k,l,m].*sx
                        elseif q==2
                            p[j,:,k,l,m] .+= fiber(v)[j,i,k,l,m].*sx
                        elseif q==3
                            p[j,k,:,l,m] .+= fiber(v)[j,k,i,l,m].*sx
                        elseif q==4
                            p[j,k,l,:,m] .+= fiber(v)[j,k,l,i,m].*sx
                        else
                            p[j,k,l,m,:] .+= fiber(v)[j,k,l.m,i].*sx
                        end
                    end end
                end end
            end end
        end
    end
    return TensorField(xy,p)
end

export resample_sinc, resample_lagrange, resample_roots
export LagrangeWeights, lagrangepoints, lagrangeweights
export lagrangepolynomial, rootspolynomial

struct LagrangeWeights{T,W,V<:AbstractVector{T}} <: AbstractVector{T}
    v::V
    w::Vector{W}
end

LagrangeWeights(v::LagrangeWeights) = v
LagrangeWeights(v::AbstractVector) = LagrangeWeights(v,lagrangeweights(v))
LagrangeWeights(v::ProductSpace) = ProductSpace(LagrangeWeights.(split(v)))
LagrangeWeights(t::PointArray) = PointArray(LagrangeWeights(points(t)),metricextensor(t))
LagrangeWeights(t::GridBundle) = GridBundle(LagrangeWeights(fullcoordinates(t)),immersion(t))
LagrangeWeights(t::TensorField) = TensorField(LagrangeWeights(base(t)),fiber(t))

lagrangepoints(v) = v
lagrangepoints(v::LagrangeWeights) = v.v
lagrangepoints(v::FiberBundle) = lagrangepoints(points(v))

lagrangeweights(v::LagrangeWeights) = v.w
lagrangeweights(v::LagrangeWeights,j) = lagrangeweights(v)[j]
lagrangeweights(v::FiberBundle) = lagrangeweights(points(v))
lagrangeweights(v::FiberBundle,j) = lagrangeweights(points(v),j)
lagrangeweights(v) = [lagrangeweights(v,j) for j ∈ 1:length(v)]
function lagrangeweights(v,j)
    out = fiber(v)[j].-v
    inv(prod(out[1:j-1])*prod(out[j+1:end]))
end

#=function lagrangeweights2(v)
    N,h = length(v),step(v)
    [inv(Float64(factorial(big(j-1))*factorial(big(N-j)))*(h^(j-1))*((-h)^(N-j))) for  j ∈ 1:N]
end=#

resample(v::LagrangeWeights,n::Int) = length(v)==n ? v : resample(lagrangepoints(v),n)

function lagrangepolynomial(t::TensorField,x::AbstractVector)
    out = lagrangepolynomial(LagrangeWeights(points(t)),fiber(t),x)
    TensorField(x,[isnan(out[i]) ? t(x[i]) : out[i] for i ∈ 1:length(out)])
end
function lagrangepolynomial(t::TensorField,x::AbstractMatrix)
    out = lagrangepolynomial(LagrangeWeights(points(t)),fiber(t),fiber(x))
    n,m = size(x)
    TensorField(x,[iszero(j) && isnan(out[i,j]) ? t(real(x[i,j])) : out[i] for i ∈ 1:n, j ∈ 1:m])
end
function lagrangepolynomial(t::TensorField,x::Number)
    out = lagrangepolynomial(LagrangeWeights(points(t)),fiber(t),x)
    isnan(out) ? t(x) : out
end
lagrangepolynomial(L::LagrangeWeights,y,x::AbstractArray) = lagrangepolynomial.(Ref(L.v),Ref(L.w.*y),x)
lagrangepolynomial(L::LagrangeWeights,y,x::Number) = lagrangepolynomial(L.v,L.w.*y,x)
function lagrangepolynomial(v,wy,x)
    xv = x.-v
    prod(xv)*sum(wy./xv)
end

#=lagrangepolynomial(L::LagrangeWeights,y,x::Number) = lagrangepolynomial(L.v,L.w,x)⋅y
function lagrangepolynomial(v,w,x)
    xv = x.-v
    prod(xv)*(w./xv)
end=#

size_new(v::Val{q},n,N) where q = (isone(q) ? n : N,)
size_new(v::Val{q},n,args...) where q = (size_new(v,n,args[1:end-1]...)...,q==length(args) ? n : args[end])

resample_lagrange(t::AbstractVector,n) = lagrangepolynomial(LagrangeWeights(t),resample(lagrangepoints(t),n))
resample_lagrange(v::AbstractMatrix,n,m::Int) = resample_lagrange(resample_lagrange(LagrangeWeights(v),n,Val(1)),m,Val(2))
resample_lagrange(v::AbstractArray{T,3} where T,n,m,o) = resample_lagrange(resample_lagrange(resample_lagrange(LagrangeWeights(v),n,Val(1)),m,Val(2)),o,Val(3))
resample_lagrange(v::AbstractArray{T,4} where T,n,m,o,p) = resample_lagrange(resample_lagrange(resample_lagrange(resample_lagrange(LagrangeWeights(v),n,Val(1)),m,Val(2)),o,Val(3)),p,Val(4))
resample_lagrange(v::AbstractArray{T,5} where T,n,m,o,p,q) = resample_lagrange(resample_lagrange(resample_lagrange(resample_lagrange(resample_lagrange(LagrangeWeights(v),n,Val(1)),m,Val(2)),o,Val(3)),p,Val(4)),q,Val(5))
function resample_lagrange(v::AbstractArray{T,N} where T,n,Q::Val{q}) where {N,q}
    M = size(v)
    M[q] == n && (return v)
    X = split(points(v))
    xyz = resample(points(v),size_new(Q,n,M...))
    xx,p = split(xyz)[q],zeros(size(xyz)...)
    xh = split(points(v))[q]
    w = lagrangeweights(xh)
    for i ∈ 1:n
        xi = xx[i]
        xv = xi.-xh
        wxv = prod(xv).*(w./xv)
        for j ∈ 1:M[isone(q) ? 2 : 1]
            if N==2
                if isone(q)
                    u = wxv⋅view(fiber(v),:,j)
                    p[i,j] = isnan(u) ? v(xi,X[2][j]) : u
                else
                    u = wxv⋅view(fiber(v),j,:)
                    p[j,i] = isnan(u) ? v(X[1][j],xi) : u
                end
            else for k ∈ 1:M[q==3 ? 2 : 3]
                if N==3
                    if isone(q)
                        u = wxv⋅view(fiber(v),:,j,k)
                        p[i,j,k] = isnan(u) ? v(xi,X[2][j],X[3][k]) : u
                    elseif q==2
                        u = wxv⋅view(fiber(v),j,:,k)
                        p[j,i,k] = isnan(u) ? v(X[1][j],xi,X[3][k]) : u
                    else
                        u = wxv⋅view(fiber(v),j,k,:)
                        p[j,k,i] = isnan(u) ? v(X[1][j],X[2][k],xi) : u
                    end
                else for l ∈ 1:M[q==4 ? 3 : 4]
                    if N==4
                        if isone(q)
                            u = wxv⋅view(fiber(v),:,j,k,l)
                            p[i,j,k,l] = isnan(u) ? v(xi,X[2][j],X[3][k],X[4][l]) : u
                        elseif q==2
                            u = wxv⋅view(fiber(v),j,:,k,l)
                            p[j,i,k,l] = isnan(u) ? v(X[1][j],xi,X[3][k],X[4][l]) : u
                        elseif q==3
                            u = wxv⋅view(fiber(v),j,k,:,l)
                            p[j,k,i,l] = isnan(u) ? v(X[1][j],X[2][k],xi,X[4][l]) : u
                        else
                            u = wxv⋅view(fiber(v),j,k,l,:)
                            p[j,k,l,i] = isnan(u) ? v(X[1][j],X[2][k],X[3][l],xi) : u
                        end
                    else for m ∈ 1:M[q==5 ? 4 : 5]
                        if isone(q)
                            u = wxv⋅view(fiber(v),:,j,k,l,m)
                            p[i,j,k,l,m] = isnan(u) ? v(xi,X[2][j],X[3][k],X[4][l],X[5][m]) : u
                        elseif q==2
                            u = wxv⋅view(fiber(v),j,:,k,l,m)
                            p[j,i,k,l,m] = isnan(u) ? v(X[1][j],xi,X[3][k],X[4][l],X[5][m]) : u
                        elseif q==3
                            u = wxv⋅view(fiber(v),j,k,:,l,m)
                            p[j,k,i,l,m] = isnan(u) ? v(X[1][j],X[2][k],xi,X[4][l],X[5][m]) : u
                        elseif q==4
                            u = wxv⋅view(fiber(v),j,k,l,:,m)
                            p[j,k,l,i,m] = isnan(u) ? v(X[1][j],X[2][k],X[3][l],xi,X[5][m]) : u
                        else
                            u = wxv⋅view(fiber(v),j,k,l,m,:)
                            p[j,k,l,m,i] = isnan(u) ? v(X[1][j],X[2][k],X[3][l],X[4][m],xi) : u
                        end
                    end end
                end end
            end end
        end
    end
    return TensorField(xyz,p)
end

resample_roots(t,n) = rootspolynomial(t,resample(lagrangepoints(t),n))
rootspolynomial(t::TensorField,x::AbstractArray) = TensorField(x,rootspolynomial(lagrangepoints(t),fiber(x)))
rootspolynomial(t::TensorField,x::Number) = rootspolynomial(lagrangepoints(t),x)
rootspolynomial(v,x::AbstractArray) = rootspolynomial.(Ref(v),x)
rootspolynomial(v,x::Number) = prod(x.-v)

Base.size(m::LagrangeWeights) = (length(m.v),)
Base.getindex(m::LagrangeWeights,i::Int) = getindex(m.v,i)

# spectral

convolve(f::ScalarField...) = irfft(*(rfft.(f)...))

export fftwavenumber, rfftwavenumber, r2rwavenumber
fftwavenumber(N::AbstractArray) = fftwavenumber(size(N)...)
rfftwavenumber(N::AbstractArray) = rfftwavenumber(size(N)...)
r2rwavenumber(N::AbstractArray) = ProductSpace(r2rwavenumber.(size(N))...)
r2rwavenumber(N::AbstractArray,kind) = ProductSpace(r2rwavenumber.(size(N),kind)...)
fftwavenumber(N...) = ProductSpace(fftwavenumber.(N)...)
rfftwavenumber(N...) = ProductSpace(rfftwavenumber.(N)...)
fftwavenumber(N::Int) = vcat(0:Int((N-isodd(N))/2)-1,-Int((N+isodd(N))/2):-1)
rfftwavenumber(N::Int) = 0:Int((N-isodd(N))/2)
r2rwavenumber(N::Int) = 0:N-1
r2rwavenumber(N::Int,kind) = kind ∈ (9,6,10) ? (1:N) : (0:N-1)

spectral_diff_fft(N::Int) = im*vcat(0:Int((N-isodd(N))/2)-1,-Int((N+isodd(N))/2):-1)
spectral_diff_rfft(N::Int) = im*(0:Int((N-isodd(N))/2))
spectral_sum_fft(N::Int) = -im*vcat(0,inv.(1:Int((N-isodd(N))/2)-1),inv.(-Int((N+isodd(N))/2):-1))
spectral_sum_rfft(N::Int) = -im*vcat(0,inv.(1:Int((N-isodd(N))/2)))
for fun ∈ (:spectral_diff_fft,:spectral_diff_rfft,:spectral_sum_fft,:spectral_sum_rfft)
    @eval begin
        $fun(t,i,N) = TensorField(base(t),$fun(N).*fiber(t))
        function $fun(t,i,N,M)
            ω = $fun((N,M)[i])
            TensorField(base(t),[fiber(t)[n,m]*ω[(n,m)[i]] for n ∈ OneTo(N), m ∈ OneTo(M)])
        end
        function $fun(t,i,N,M,O)
            ω = $fun((N,M,O)[i])
            TensorField(base(t),[fiber(t)[n,m,o]*ω[(n,m,o)[i]] for n ∈ OneTo(N), m ∈ OneTo(M), o ∈ OneTo(O)])
        end
        function $fun(t,i,N,M,O,P)
            ω = $fun((N,M,O,P)[i])
            TensorField(base(t),[fiber(t)[n,m,o,p]*ω[(n,m,o,p)[i]] for n ∈ OneTo(N), m ∈ OneTo(M), o ∈ OneTo(O), p ∈ OneTo(P)])
        end
        function $fun(t,i,N,M,O,P,Q)
            ω = $fun((N,M,O,P,Q)[i])
            TensorField(base(t),[fiber(t)[n,m,o,p,q]*ω[(n,m,o,p,q)[i]] for n ∈ OneTo(N), m ∈ OneTo(M), o ∈ OneTo(O), p ∈ OneTo(P), q ∈ OneTo(Q)])
        end
    end
end

gradient_fft(t::RealFunction,d=spectral_diff_fft(length(t))) = real(ifft(d.*fft(t)))
gradient_fft(t::AbstractCurve,d=spectral_diff_fft(length(t))) = Chain.(gradient_fft.(value(fiber(Chain(t))),Ref(d))...)
function gradient_fft(t::AbstractMatrix,i::Int)
    V = fft(t,i)
    N,M = size(V)
    if isone(i)
        d = spectral_diff_fft(N)
        for i ∈ 1:M
            V[:,i] .*= d
        end
    else
        d = spectral_diff_fft(M)
        for i ∈ 1:N
            V[i,:] .*= d
        end
    end
    TensorField(t,ifft(V,i))
end
function gradient_fft(t::AbstractArray{T,3} where T,i::Int)
    V = fft(t,i)
    N,M,R = size(V)
    if isone(i)
        d = spectral_diff_fft(N)
        for i ∈ 1:M
            for j ∈ 1:R
                V[:,i,j] .*= d
            end
        end
    elseif i==2
        d = spectral_diff_fft(M)
        for i ∈ 1:N
            for j ∈ 1:R
                V[i,:,j] .*= d
            end
        end
    else
        d = spectral_diff_fft(R)
        for i ∈ 1:N
            for j ∈ 1:M
                V[i,j,:] .*= d
            end
        end
    end
    TensorField(t,ifft(V,i))
end
gradient_fft(t::TensorField) = Chain.(gradient_fft.(Ref(t),list(1,mdims(pointtype(t))))...)
gradient_fft(t::VectorField,i::Int) = gradient_fft.(value(fiber(Chain(t))),i)
gradient_rfft(t::RealFunction,d=spectral_diff_rfft(length(t))) = real(irfft(d.*rfft(t)))
gradient_rfft(t::AbstractCurve,d=spectral_diff_rfft(length(t))) = Chain.(gradient_rfft.(value(fiber(Chain(t))),Ref(d))...)
function gradient_rfft(t::AbstractMatrix,i::Int)
    V = rfft(t,i)
    N,M = size(V)
    if isone(i)
        d = spectral_diff_fft(N)
        for i ∈ 1:M
            V[:,i] .*= d
        end
    else
        d = spectral_diff_fft(M)
        for i ∈ 1:N
            V[i,:] .*= d
        end
    end
    TensorField(t,irfft(V,i))
end
function gradient_rfft(t::AbstractArray{3,T} where T,i::Int)
    V = rfft(t,i)
    N,M,R = size(V)
    if isone(i)
        d = spectral_diff_fft(N)
        for i ∈ 1:M
            for j ∈ 1:R
                V[:,i,j] .*= d
            end
        end
    elseif i==2
        d = spectral_diff_fft(M)
        for i ∈ 1:N
            for j ∈ 1:R
                V[i,:,j] .*= d
            end
        end
    else
        d = spectral_diff_fft(R)
        for i ∈ 1:N
            for j ∈ 1:M
                V[i,j,:] .*= d
            end
        end
    end
    TensorField(t,irfft(V,i))
end
gradient_rfft(t::TensorField) = Chain.(gradient_rfft.(Ref(t),list(1,mdims(pointtype(t))))...)
gradient_rfft(t::VectorField,i::Int) = gradient_rfft.(value(fiber(Chain(t))),i)
gradient_impulse(t::TensorField) = real(irfft(gradient_impulse_rfft(t)))
gradient_impulse_fft(N::Int) = TensorField(fftspace(N),spectral_diff_fft(N))
gradient_impulse_fft(t::TensorField) = TensorField(fftspace(t),spectral_diff_fft(length(t)))
gradient_impulse_rfft(N::Int) = TensorField(rfftspace(N),spectral_diff_rfft(N))
gradient_impulse_rfft(t::TensorField) = TensorField(rfftspace(t),spectral_diff_rfft(length(t)))

integral_fft(t::AbstractCurve,d=spectral_diff_fft(length(t))) = Chain.(integral_fft.(value(fiber(Chain(t))),Ref(d))...)
function integral_fft(t::RealFunction,d=spectral_sum_fft(length(t)))
    m = mean(fiber(t))
    out = real(ifft(d.*fft(t-m)))
    out+m*(TensorField(base(t))-(points(t)[1]+fiber(out[1])/m))
end
integral_rfft(t::AbstractCurve,d=spectral_diff_rfft(length(t))) = Chain.(integral_rfft.(value(fiber(Chain(t))),Ref(d))...)
function integral_rfft(t::RealFunction,d=spectral_sum_rfft(length(t)))
    m = mean(fiber(t))
    out = real(irfft(d.*rfft(t-m)))
    out+m*(TensorField(base(t))-(points(t)[1]+fiber(out[1])/m))
end
integral_impulse(t::TensorField) = real(irfft(integral_impulse_rfft(t)))
integral_impulse_fft(N::Int) = TensorField(fftspace(N),spectral_sum_fft(N))
integral_impulse_fft(t::TensorField) = TensorField(fftspace(t),spectral_sum_fft(length(t)))
integral_impulse_rfft(N::Int) = TensorField(rfftspace(N),spectral_sum_rfft(N))
integral_impulse_rfft(t::TensorField) = TensorField(rfftspace(t),spectral_sum_rfft(length(t)))

integrate_fft(t::AbstractCurve,d=spectral_diff_fft(length(t))) = Chain(integrate_fft.(value(fiber(Chain(t))),Ref(d))...)
integrate_fft(t::RealFunction,d=spectral_sum_fft(length(t))) = fiber(integral_fft(t,d)[end])
integrate_rfft(t::AbstractCurve,d=spectral_diff_rfft(length(t))) = Chain(integrate_rfft.(value(fiber(Chain(t))),Ref(d))...)
integrate_rfft(t::RealFunction,d=spectral_sum_rfft(length(t))) = fiber(integral_rfft(t,d)[end])

function spectral_sum_impulse(N::Int)
    x = N/2 # midpoint
    b = (π/2)/x # amplitude
    (b/-x).*(0:N-1).+b # slope
end
integral_impulse_line(t::TensorField) = TensorField(base(t),spectral_sum_impulse(length(t)))

gradient_chebyshev(v,D=ChebyshevMatrix(v)) = D*v

function gradient_chebyshevfft(v::AbstractVector,d=spectral_diff_chebfft(v))
    U = -real.(chebyshevfft(v))
    TensorField(v,chebyshevifft(d.*U,U,length(v)))
end

function gradient_chebyshevfft(v::AbstractMatrix,i)
    U = -real.(chebyshevfft(v,i))
    V = complex.(U)
    N,M = size(v)
    if isone(i)
        d = spectral_diff_chebfft(N)
        for i ∈ 1:M
            V[:,i] .*= d
        end
    else
        d = spectral_diff_chebfft(M)
        for i ∈ 1:N
            V[i,:] .*= d
        end
    end
    TensorField(v,chebyshevifft(V,U,i,size(v)...))
end

function gradient_chebyshevfft(v::AbstractArray{T,3} where T,i)
    U = -real.(chebyshevfft(v,i))
    V = complex.(U)
    N,M,R = size(v)
    if isone(i)
        d = spectral_diff_chebfft(N)
        for i ∈ 1:M
            for j ∈ 1:R
                V[:,i,j] .*= d
            end
        end
    elseif i==2
        d = spectral_diff_chebfft(M)
        for i ∈ 1:N
            for j ∈ 1:R
                V[i,:,j] .*= d
            end
        end
    else
        d = spectral_diff_chebfft(R)
        for i ∈ 1:N
            for j ∈ 1:M
                V[i,j,:] .*= d
            end
        end
    end
    TensorField(v,chebyshevifft(V,U,i,size(v)...))
end

function gradient2_chebyshevfft(v::AbstractVector,d=spectral_diff_chebfft(v),d2=spectral_diff_chebfft2(v))
    U = real.(chebyshevfft(v))
    TensorField(v,chebyshevifft2(d.*U,d2.*U,U,length(v)))
end

function gradient2_chebyshevfft(v::AbstractMatrix,i)
    U = real.(chebyshevfft(v,i))
    V1,V2 = complex.(copy(U)),complex.(copy(U))
    N,M = size(v)
    if isone(i)
        d = spectral_diff_chebfft(N)
        d2 = spectral_diff_chebfft2(N)
        for i ∈ 1:M
            V1[:,i] .*= d
            V2[:,i] .*= d2
        end
    else
        d = spectral_diff_chebfft(M)
        d2 = spectral_diff_chebfft2(M)
        for i ∈ 1:N
            V1[i,:] .*= d
            V2[i,:] .*= d2
        end
    end
    TensorField(v,chebyshevifft2(V1,V2,U,i,size(v)...))
end

function gradient2_chebyshevfft(v::AbstractArray{T,3} where T,i)
    U = real.(chebyshevfft(v,i))
    V1,V2 = complex.(copy(U)),complex.(copy(U))
    N,M,R = size(v)
    if isone(i)
        d = spectral_diff_chebfft(N)
        d2 = spectral_diff_chebfft2(N)
        for i ∈ 1:M
            for j ∈ 1:R
                V1[:,i,j] .*= d
                V2[:,i,j] .*= d2
            end
        end
    elseif i==2
        d = spectral_diff_chebfft(M)
        d2 = spectral_diff_chebfft2(M)
        for i ∈ 1:N
            for j ∈ 1:R
                V1[i,:,j] .*= d
                V2[i,:,j] .*= d2
            end
        end
    else
        d = spectral_diff_chebfft(R)
        d2 = spectral_diff_chebfft2(R)
        for i ∈ 1:N
            for j ∈ 1:M
                V1[i,j,:] .*= d
                V2[i,j,:] .*= d2
            end
        end
    end
    TensorField(v,chebyshevifft2(V1,V2,U,i,size(v)...))
end

for fun ∈ (:laplacian_chebyshevfft,:gradient,:gradient_fft,:gradient_rfft,:gradient_chebyshevfft,:gradient2_chebyshevfft)
    @eval $fun(v::LocalTensor) = $fun(fiber(v))
end
laplacian_chebyshevfft(v::AbstractVector) = gradient2_chebyshevfft(v)
laplacian_chebyshevfft(v::AbstractMatrix) = gradient2_chebyshevfft(v,1)+gradient2_chebyshevfft(v,2)
laplacian_chebyshevfft(v::AbstractArray{T,3} where T) = gradient2_chebyshevfft(v,1)+gradient2_chebyshevfft(v,2)+gradient2_chebyshevfft(v,3)

spectral_diff_chebfft(v::AbstractVector) = spectral_diff_chebfft(length(v))
spectral_diff_chebfft(N::Int) = im*vcat(0:N-2,0,2-N:-1)

spectral_diff_chebfft2(v::AbstractVector) = spectral_diff_chebfft2(length(v))
spectral_diff_chebfft2(N::Int) = -vcat(0:N-2,0,2-N:-1).^2

gradient_toeplitz(v,D=derivetoeplitz(v)) = D*v
gradient2_toeplitz(v,D=derivetoeplitz2(v)) = D*v

toeplitz1(N,h=2π/N) = vcat(0,0.5*(-1).^(1:N-1).*cot.((1:N-1)*h/2))
toeplitz2(N,h=2π/N) = vcat(-π^2/(3*h^2)-1/6,0.5*(-1).^(2:N)./sin.((1:N-1)*h/2).^2)
derivetoeplitz(v) = derivetoeplitz(length(v))
derivetoeplitz2(v) = derivetoeplitz2(length(v))
export derivetoeplitz, derivetoeplitz2

export integrate_haar, GaussLegendre, laplacian_chebyshevfft
export integrate_chebyshev, integrate_clenshawcurtis, integrate_gausslegendre
export integral_chebyshev, integral_clenshawcurtis, integral_gausslegendre
export gradient_chebyshev, gradient_chebyshevfft, clenshawcurtis, gausslegendre
export gradient2_chebyshevfft, gradient2_toeplitz, gradient_toeplitz

integral_weight(t,w) = TensorField(t,cumsum(w.*fiber(t)))
integrate_weight(t,w) = w⋅fiber(t)

integral_chebyshev(t,w=ChebyshevVector(t)) = integral_weight(t,w)
integral_clenshawcurtis(t,w=clenshawcurtis(t)) = integral_weight(t,w)
integral_gausslegendre(t,w=gausslegendre(t)) = integral_weight(t,w)
integrate_chebyshev(t,w=ChebyshevVector(t)) = integrate_weight(t,w)
integrate_clenshawcurtis(t,w=clenshawcurtis(t)) = integrate_weight(t,w)
integrate_gausslegendre(t,w=gausslegendre(t)) = integrate_weight(t,w)
clenshawcurtis(t) = clenshawcurtis(length(t))*(interval_scale(t)/2)
function clenshawcurtis(n::Int)
    N = n-1
    θ = π*(0:N)/N
    #x = cos.(θ)
    w = zeros(N+1)
    ii = 2:N
    v = ones(N-1)
    if iszero(N%2)
        w[1] = 1/(N^2+1)
        w[N+1] = 0
        for k ∈ 1:Int(N/2)-1
            v .-= 2cos.(2k*θ[ii])/(4k^2-1)
        end
        v .-= cos.(N*θ[ii])/(N^2-1)
    else
        w[1] = 1/N^2
        w[N+1] = 0
        for k ∈ 1:Int((N-1)/2)
            v .-= 2cos.(2k*θ[ii])/(4k^2-1)
        end
    end
    w[ii] .= (2/N).*v
    return reverse(w)
end

