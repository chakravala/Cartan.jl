
#   This file is part of TensorFields.jl
#   It is licensed under the GPL license
#   TensorFields Copyright (C) 2023 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com

export Grid

struct Grid{N,T,A<:AbstractArray{T,N}}
    v::A
end

Base.size(m::Grid) = size(m.v)

@generated function Base.getindex(g::Grid{M},j::Int,::Val{N},i::Vararg{Int}) where {M,N}
    :(Base.getindex(g.v,$([k≠N ? :(i[$k]) : :(i[$k]+j) for k ∈ 1:M]...)))
end

# centraldiff

centraldiffdiff(f,dt,l) = centraldiff(centraldiff(f,dt,l),dt,l)
centraldiffdiff(f,dt) = centraldiffdiff(f,dt,size(f))

centraldiff(f::AbstractArray,args...) = centraldiff(Grid(f),args...)

centraldiff(f::Grid{1},dt::Float64,l=size(f.v)) = [centraldiff(f,l,i)/centraldiff(i,dt,l) for i ∈ 1:l]
centraldiff(f::Grid{1},dt::Vector,l=size(f.v)) = [centraldiff(f,l,i)/dt[i] for i ∈ 1:l[1]]
centraldiff(f::Grid{1},l=size(f.v)) = [centraldiff(f,l,i) for i ∈ 1:l[1]]

centraldiff(f::Grid{2},dt::AbstractMatrix,l::Tuple=size(f.v)) = [Chain(centraldiff(f,l,i,j).v./(dt[i,j].v)) for i ∈ 1:l[1], j ∈ 1:l[2]]
centraldiff(f::Grid{2},l::Tuple=size(f.v)) = [centraldiff(f,l,i,j) for i ∈ 1:l[1], j ∈ 1:l[2]]

centraldiff(f::Grid{3},dt::AbstractArray{T,3} where T,l::Tuple=size(f.v)) = [Chain(centraldiff(f,l,i,j,k).v./(dt[i,j,k].v)) for i ∈ 1:l[1], j ∈ 1:l[2], k ∈ 1:l[3]]
centraldiff(f::Grid{3},l::Tuple=size(f.v)) = [centraldiff(f,l,i,j,k) for i ∈ 1:l[1], j ∈ 1:l[2], k ∈ 1:l[3]]

centraldiff(f::Grid{1},l,i::Int) = centraldiff(f,l[1],Val(1),i)
@generated function centraldiff(f::Grid{M},l,i::Vararg{Int}) where M
    :(Chain($([:(centraldiff(f,l[$k],Val($k),i...)) for k ∈ 1:M]...)))
end
function centraldiff(f::Grid,l,k::Val{N},i::Vararg{Int}) where N #l=size(f)[N]
    if isone(i[N])
        18f[1,k,i...]-9f[2,k,i...]+2f[3,k,i...]-11f.v[i...]
    elseif i[N]==l
        11f.v[i...]-18f[-1,k,i...]+9f[-2,k,i...]-2f[-3,k,i...]
    elseif i[N]==2
        6f[1,k,i...]-f[2,k,i...]-3f.v[i...]-2f[-1,k,i...]
    elseif i[N]==l-1
        3f.v[i...]-6f[-1,k,i...]+f[-2,k,i...]+2f[1,k,i...]
    else
        f[-2,k,i...]+8f[1,k,i...]-8f[-1,k,i...]-f[2,k,i...]
    end
end

centraldiff(f::RealRegion) = ProductSpace(centraldiff.(f.v))
centraldiff(f::StepRangeLen,l=length(f)) = [centraldiff(i,Float64(f.step),l) for i ∈ 1:l]
centraldiff(dt::Float64,l::Int) = [centraldiff(i,dt,l) for i ∈ 1:l]
function centraldiff(i::Int,dt::Float64,l::Int)
    if i∈(1,2,l-1,l)
        6dt
    else
        12dt
    end
end

centraldiff_fast(f::AbstractArray,args...) = centraldiff_fast(Grid(f),args...)

centraldiff_fast(f::Grid{1},dt::Float64,l=size(f.v)) = [centraldiff_fast(f,l,i)/centraldiff_fast(i,dt,l) for i ∈ 1:l]
centraldiff_fast(f::Grid{1},dt::Vector,l=size(f.v)) = [centraldiff_fast(f,l,i)/dt[i] for i ∈ 1:l[1]]
centraldiff_fast(f::Grid{1},l=size(f.v)) = [centraldiff_fast(f,l,i) for i ∈ 1:l[1]]

centraldiff_fast(f::Grid{2},dt::AbstractMatrix,l::Tuple=size(f.v)) = [Chain(centraldiff(f,l,i,j).v./(dt[i,j].v)) for i ∈ 1:l[1], j ∈ 1:l[2]]
centraldiff_fast(f::Grid{2},l::Tuple=size(f.v)) = [centraldiff(f,l,i,j) for i ∈ 1:l[1], j ∈ 1:l[2]]

centraldiff_fast(f::Grid{3},dt::AbstractArray{T,3} where T,l::Tuple=size(f.v)) = [Chain(centraldiff_fast(f,l,i,j,k).v./(dt[i,j,k].v)) for i ∈ 1:l[1], j ∈ 1:l[2], k ∈ 1:l[3]]
centraldiff_fast(f::Grid{3},l::Tuple=size(f.v)) = [centraldiff_fast(f,l,i,j,k) for i ∈ 1:l[1], j ∈ 1:l[2], k ∈ 1:l[3]]

centraldiff_fast(f::Grid{1},l,i::Int) = centraldiff_fast(f,l[1],Val(1),i)
@generated function centraldiff_fast(f::Grid{M},l,i::Vararg{Int}) where M
    :(Chain($([:(centraldiff_fast(f,l[$k],Val($k),i...)) for k ∈ 1:M]...)))
end
function centraldiff_fast(f::Grid,l,k::Val{N},i::Vararg{Int}) where N
    if isone(i[N]) # 4f[1,k,i...]-f[2,k,i...]-3f.v[i...]
        18f[1,k,i...]-9f[2,k,i...]+2f[3,k,i...]-11f.v[i...]
    elseif i[N]==l # 3f.v[i...]-4f[-1,k,i...]+f[-2,k,i...]
        11f.v[i...]-18f[-1,k,i...]+9f[-2,k,i...]-2f[-3,k,i...]
    else
        f[1,k,i...]-f[-1,k,i...]
    end
end

centraldiff_fast(f::RealRegion) = ProductSpace(centraldiff_fast.(f.v))
centraldiff_fast(f::StepRangeLen,l=length(f)) = [centraldiff_fast(i,Float64(f.step),l) for i ∈ 1:l]
centraldiff_fast(dt::Float64,l::Int) = [centraldiff_fast(i,dt,l) for i ∈ 1:l]
centraldiff_fast(i::Int,dt::Float64,l::Int) = i∈(1,l) ? 6dt : 2dt
#centraldiff_fast(i::Int,dt::Float64,l::Int) = 2dt

qnumber(n,q) = (q^n-1)/(q-1)
qfactorial(n,q) = prod(cumsum([q^k for k ∈ 0:n-1]))
qbinomial(n,k,q) = qfactorial(n,q)/(qfactorial(n-k,q)*qfactorial(k,q))

richardson(k) = [(-1)^(j+k)*2^(j*(1+j))*qbinomial(k,j,4)/prod([4^n-1 for n ∈ 1:k]) for j ∈ k:-1:0]

# differential geometry

export arclength, arctime, trapz, linetrapz
export centraldiff, tangent, tangent_fast, unittangent, speed, normal, unitnormal

function comp(f)
    at,al = arctime(f),arclength(f)
    domain(at)[2:end-1] → [domain(at)[j]-value(al(value(at.cod[j]))) for j ∈ 2:length(f)-1]
end

arctime(f) = inv(arclength(f))
arclength(f::Vector) = sum(value.(abs.(diff(f))))
function arclength(f::IntervalMap)
    int = cumsum(abs.(diff(codomain(f))))
    pushfirst!(int,zero(eltype(int)))
    domain(f) → int
end # trapz(speed(f))
function trapz(f::IntervalMap,d=diff(domain(f)))
    int = (d/2).*cumsum(f.cod[2:end]+f.cod[1:end-1])
    pushfirst!(int,zero(eltype(int)))
    domain(f) → int
end
function trapz(f::Vector,h::Float64)
    int = (h/2)*cumsum(f[2:end]+f[1:end-1])
    pushfirst!(int,zero(eltype(int)))
    return int
end
function linetrapz(γ::IntervalMap,f::Function)
    trapz(domain(γ)→(f.(codomain(γ)).⋅codomain(tangent(γ))))
end
function tangent(f::IntervalMap,d=centraldiff(domain(f)))
    domain(f) → centraldiff(codomain(f),d)
end
function tangent_fast(f::IntervalMap,d=centraldiff_fast(domain(f)))
    domain(f) → centraldiff_fast(codomain(f),d)
end
function tangent(f::ScalarGrid,d=centraldiff(domain(f)))
    domain(f) → centraldiff(Grid(codomain(f)),d)
end
function tangent(f::MeshFunction)
    domain(f) → interp(domain(f),gradient(domain(f),codomain(f)))
end
function unittangent(f::IntervalMap,d=centraldiff(domain(f)),t=centraldiff(codomain(f),d))
    domain(f) → (t./abs.(t))
end
function unittangent(f::MeshFunction)
    t = interp(domain(f),gradient(domain(f),codomain(f)))
    domain(f) → (t./abs.(t))
end
function speed(f::IntervalMap,d=centraldiff(domain(f)),t=centraldiff(codomain(f),d))
    domain(f) → abs.(t)
end
function normal(f::IntervalMap,d=centraldiff(domain(f)),t=centraldiff(codomain(f),d))
    domain(f) → centraldiff(t,d)
end
function unitnormal(f::IntervalMap,d=centraldiff(domain(f)),t=centraldiff(codomain(f),d),n=centraldiff(t,d))
    domain(f) → (n./abs.(n))
end

export curvature, radius, osculatingplane, unitosculatingplane, binormal, unitbinormal, curvaturebivector, torsion, curvaturetrivector, trihedron, frenet, wronskian, bishoppolar, bishop, curvaturetorsion

function curvature(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),a=abs.(t))
    domain(f) → (abs.(centraldiff(t./a,d))./a)
end
function radius(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),a=abs.(t))
    domain(f) → (a./abs.(centraldiff(t./a,d)))
end
function osculatingplane(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    domain(f) → (t.∧n)
end
function unitosculatingplane(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    domain(f) → ((t./abs.(t)).∧(n./abs.(n)))
end
function binormal(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    domain(f) → (.⋆(t.∧n))
end
function unitbinormal(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d))
    a = t./abs.(t)
    n = centraldiff(t./abs.(t),d)
    domain(f) → (.⋆(a.∧(n./abs.(n))))
end
function curvaturebivector(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    a=abs.(t); ut=t./a
    domain(f) → (abs.(centraldiff(ut,d))./a.*(ut.∧(n./abs.(n))))
end
function torsion(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d),b=t.∧n)
    domain(f) → ((b.∧centraldiff(n,d))./abs.(.⋆b).^2)
end
function curvaturetrivector(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d),b=t.∧n)
    a=abs.(t); ut=t./a
    domain(f) → ((abs.(centraldiff(ut,d)./a).^2).*(b.∧centraldiff(n,d))./abs.(.⋆b).^2)
end
#torsion(f::TensorField,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d),a=abs.(t)) = domain(f) → (abs.(centraldiff(.⋆((t./a).∧(n./abs.(n))),d))./a)
function trihedron(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    (ut,un)=(t./abs.(t),n./abs.(n))
    domain(f) → Chain.(ut,un,.⋆(ut.∧un))
end
function frenet(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    (ut,un)=(t./abs.(t),n./abs.(n))
    domain(f) → centraldiff(Chain.(ut,un,.⋆(ut.∧un)),d)
end
function wronskian(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d))
    domain(f) → (f.cod.∧t.∧n)
end

function curvaturetorsion(f::SpaceCurve,d=centraldiff(f.dom),t=centraldiff(f.cod,d),n=centraldiff(t,d),b=t.∧n)
    a = abs.(t)
    domain(f) → Chain.(value.(abs.(centraldiff(t./a,d))./a),getindex.((b.∧centraldiff(n,d))./abs.(.⋆b).^2,1))
end

function bishoppolar(f::SpaceCurve,κ=value.(curvature(f).cod))
    d = diff(f.dom)
    τs = getindex.((torsion(f).cod).*(speed(f).cod),1)
    θ = (d/2).*cumsum(τs[2:end]+τs[1:end-1])
    pushfirst!(θ,zero(eltype(θ)))
    domain(f) → Chain.(κ,θ)
end
function bishop(f::SpaceCurve,κ=value.(curvature(f).cod))
    d = diff(f.dom)
    τs = getindex.((torsion(f).cod).*(speed(f).cod),1)
    θ = (d/2).*cumsum(τs[2:end]+τs[1:end-1])
    pushfirst!(θ,zero(eltype(θ)))
    domain(f) → Chain.(κ.*cos.(θ),κ.*sin.(θ))
end
#bishoppolar(f::TensorField) = domain(f) → Chain.(value.(curvature(f).cod),getindex.(trapz(torsion(f)).cod,1))
#bishop(f::TensorField,κ=value.(curvature(f).cod),θ=getindex.(trapz(torsion(f)).cod,1)) = domain(f) → Chain.(κ.*cos.(θ),κ.*sin.(θ))
