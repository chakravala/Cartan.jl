
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

export centraldiff, centraldiff_slow, centraldiff_fast
export gradient, gradient_slow, gradient_fast, unitgradient
export integral, integrate, ∫

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
(m::GridBundle{1})(t::Chain) = linterp(m,t)
(m::GridBundle{1})(t::AbstractFloat) = linterp(m,t)
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
function parametric(t,m,d=diff(fiber(m))./diff(points(m)))
    p = points(m)
    i,i0 = searchpoints(p,t)
    fiber(m)[i]+(t-p[i])*d[i]
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

leaf(m::TensorField{B,F,2,<:FiberProductBundle} where {B,F},i::Int) = TensorField(base(m).s,fiber(m)[:,i])
function (m::TensorField{B,F,2,<:FiberProductBundle} where {B,F})(t::Real)
    k = 2; p = base(m).g.v[1]
    i,i0 = searchpoints(p,t)
    TensorField(base(m).s,linterp(t,p[i],p[i+1],m.cod[:,i],m.cod[:,i+1]))
end

#(m::TensorField)(t::TensorField) = TensorField(base(t),m.(fiber(t)))
#(m::GridBundle)(t::TensorField) = GridBundle(PointArray(points(t),m.(fiber(t))),immersion(m))
(X::VectorField{B,F,N} where {B,F})(Y::VectorField{B,<:Chain{V,1,T,N} where {V,T},1} where B) where N = TensorField(base(Y),X.(fiber(Y)))
(m::GridBundle{N})(t::VectorField{B,<:Chain{V,1,T,N} where {V,T},1} where B) where N = TensorField(GridBundle(PointArray(points(t),m.(fiber(t))),immersion(t)),fiber(t))
#(m::GridBundle{N})(t::VectorField{B,<:Chain{V,1,T,N} where {V,T},1} where B) where N = GridBundle(PointArray(points(t),m.(fiber(t))),immersion(t))

(m::GridBundle{2})(t::Chain) = bilinterp(m,t)
(m::GridBundle{2})(x::AbstractFloat,y::AbstractFloat) = bilinterp(m,Chain(x,y))
(m::TensorField{B,F,N,<:RealSpace{2}} where {B,F,N})(t::Chain) = bilinterp(m,t)
(m::TensorField{B,F,N,<:RealSpace{2}} where {B,F,N})(t::Complex) = m(real(t),imag(t))
(m::TensorField{B,F,N,<:RealSpace{2}} where {B,F,N})(t::PseudoCouple) = m(Grassmann.realvalue(t),Grassmann.imagvalue(t))
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
            i1,i2,j1,j2 = (i01 && !iq1,i02 && !iq2,j01 && !jq1,j02 && !jq2)
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

(m::GridBundle{3})(t::Chain) = trilinterp(m,t)
(m::GridBundle{3})(x::AbstractFloat,y::AbstractFloat,z::AbstractFloat) = trilinterp(m,Chain(x,y,z))
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

(m::GridBundle{4})(t::Chain) = quadlinterp(m,t)
(m::GridBundle{4})(x::AbstractFloat,y::AbstractFloat,z::AbstractFloat,w::AbstractFloat) = quadlinterp(m,Chain(x,y,z,w))
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

(m::GridBundle{5})(t::Chain) = quintlinterp(m,t)
(m::GridBundle{5})(x::AbstractFloat,y::AbstractFloat,z::AbstractFloat,w::AbstractFloat,v::AbstractFloat) = quintlinterp(m,Chain(x,y,z,w,v))
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

import Base: @nloops, @nref, @ncall

macro nthreads(N, itersym, rangeexpr, args...)
    _nthreads(N, itersym, rangeexpr, args...)
end
function _nthreads(N::Int, itersym::Symbol, arraysym::Symbol, args::Expr...)
    @gensym d
    _nthreads(N, itersym, :($d->Base.axes($arraysym, $d)), args...)
end
function _nthreads(N::Int, itersym::Symbol, rangeexpr::Expr, args::Expr...)
    ex = Base.Cartesian._nloops(N, itersym, rangeexpr, args...)
    Expr(:block,ex.args[1],Expr(:macrocall,Symbol("@threads"),nothing,ex.args[2]))
end

for fun ∈ (:_slow,:_fast)
    cd,grad = Symbol(:centraldiff,fun),Symbol(:gradient,fun)
    cdg,cdp,cdf = Symbol(cd,:_calc),Symbol(cd,:_points),Symbol(cd,:_fiber) 
    @eval begin
        $cdf(f,args...) = $cdg(GridBundle(PointArray(0,fiber(f)),immersion(f)),args...)
        $cdp(f,args...) = $cdg(GridBundle(PointArray(0,points(f)),immersion(f)),args...)
        $cdp(f::TensorField{B,F,Nf,<:RealSpace{Nf,P,<:InducedMetric} where P} where {B,F,Nf},n::Val{N},args...) where N = $cd(points(f).v[N],subtopology(immersion(f),n),args...)
        function $grad(f::IntervalMap,d::AbstractVector=$cdp(f))
            TensorField(base(f), $cdf(f,d))
        end
        function $grad(f::TensorField{B,F,N,<:AbstractArray} where {B,F,N},d::AbstractArray=$cd(base(f)))
            TensorField(base(f), $cdf(f,d))
        end
        function $grad(f::IntervalMap,::Val{1},d::AbstractVector=$cdp(f))
            TensorField(base(f), $cdf(f,d))
        end
        function $grad(f::TensorField{B,F,Nf,<:RealSpace{Nf,P,<:InducedMetric} where P} where {B,F,Nf},n::Val{N},d::AbstractArray=$cd(points(f).v[N])) where N
            TensorField(base(f), $cdf(f,n,d))
        end
        function $grad(f::TensorField{B,F,Nf,<:RealSpace} where {B,F,Nf},n::Val{N},d::AbstractArray=$cd(points(f).v[N])) where N
            l = size(points(f))
            dg = sqrt.(getindex.(metricextensor(f),N+1,N+1))
            @threads for i ∈ l[1]; for j ∈ l[2]
                dg[i,j] *= d[isone(N) ? i : j]
            end end
            TensorField(base(f), $cdf(f,n,dg))
        end
        function $grad(f::TensorField,n::Val,d::AbstractArray=$cdp(f,n))
            TensorField(base(f), $cdf(f,n,d))
        end
        $grad(f::TensorField,n::Int,args...) = $grad(f,Val(n),args...)
        $cd(f::AbstractArray,args...) = $cdg(GridBundle(PointArray(0,f)),args...)
        $cd(f::AbstractArray,q::QuotientTopology,args...) = $cdg(GridBundle(PointArray(0,f),q),args...)
        function $cdg(f::GridBundle{1},dt::Real,s::Tuple=size(f))
            d = similar(points(f))
            @threads for i ∈ OneTo(s[1])
                d[i] = $cdg(f,s,i)/$cdg(i,dt,l)
            end
            return d
        end
        function $cdg(f::GridBundle{1},dt::DenseVector,s::Tuple=size(f))
            d = similar(points(f))
            @threads for i ∈ OneTo(s[1])
                d[i] = $cdg(f,s,i)/dt[i]
            end
            return d
        end
        function $cdg(f::GridBundle{1},s::Tuple=size(f))
            d = similar(points(f))
            @threads for i ∈ OneTo(s[1])
                d[i] = $cdg(f,s,i)
            end
            return d
        end
        $cdg(f::GridBundle{1},s::Tuple,i::Int) = $cdg(f,s[1],Val(1),i)
        @generated function $cdg(f::GridBundle{N},s::Tuple,i::Vararg{Int}) where N
            :(Chain($([:($$cdg(f,s[$n],Val($n),i...)) for n ∈ list(1,N)]...)))
        end
        $cd(f::RealRegion) = ProductSpace($cd.(f.v))
        $cd(f::GridBundle{N,Coordinate{P,G},<:PointArray{P,G,N,<:RealRegion,<:Global{N,<:InducedMetric}},<:ProductTopology}) where {N,P,G} = ProductSpace($cd.(points(f).v))
        $cd(f::GridBundle{N,Coordinate{P,G},<:PointArray{P,G,N,<:RealRegion,<:Global{N,<:InducedMetric}},<:OpenTopology}) where {N,P,G} = ProductSpace($cd.(points(f).v))
        $cd(f::GridBundle{N,Coordinate{P,G},<:PointArray{P,G,N,<:RealRegion,<:Global{N,<:InducedMetric}}}) where {N,P,G} = sum.(value.($cdg(f)))
        #$cd(f::GridBundle{N,Coordinate{P,G},<:PointArray{P,G,N,<:RealRegion},<:ProductTopology}) where {N,P,G} = applymetric.($cd(points(f)),metricextensor(f))
        $cd(f::GridBundle{N,Coordinate{P,G},<:PointArray{P,G,N,<:RealRegion},<:OpenTopology}) where {N,P,G} = applymetric.($cd(points(f)),metricextensor(f))
        $cd(f::GridBundle{N,Coordinate{P,G},<:PointArray{P,G,N,<:RealRegion}}) where {N,P,G} = applymetric.(sum.(value.($cdg(f))),metricextensor(f))
        $cd(f::AbstractRange,q::OpenTopology,s::Tuple=size(f)) = $cd(f,s)
        function $cd(f::AbstractRange,q::QuotientTopology,s::Tuple=size(f))
            d = Vector{eltype(f)}(undef,s[1])
            @threads for i ∈ OneTo(s[1])
                d[i] = $cdg(i,q,step(f),s[1])
            end
            return d
        end
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
    for N ∈ list(2,5)
        @eval begin
            function $cdg(f::GridBundle{$N},dt::AbstractArray{T,$N} where T,s::Tuple=size(f))
                d = Array{Chain{Submanifold($N),1,pointtype(f),$N},$N}(undef,s...)
                @nthreads $N i d begin
                    (@nref $N d i) = Chain((@ncall $N $cdg f s i).v./(@nref $N dt i).v)
                end
                return d
            end
            function $cdg(f::GridBundle{$N},s::Tuple=size(f))
                d = Array{Chain{Submanifold($N),1,pointtype(f),$N},$N}(undef,s...)
                @nthreads $N i d begin
                    (@nref $N d i) = (@ncall $N $cdg f s i)
                end
                return d
            end
            function $cdg(f::GridBundle{$N},n::Val{M},dt::AbstractMatrix,s::Tuple=size(f)) where M
                d = Array{pointtype(f),$N}(undef,s...)
                sM = @inbounds s[M]
                @nthreads $N i d begin
                    (@nref $N d i) = (@ncall $N $cdg f sM n i)/(@nref $N dt i)
                end
                return d
            end
            function $cdg(f::GridBundle{$N},n::Val{M},s::Tuple=size(f)) where M
                d = Array{pointtype(f),$N}(undef,s...)
                sM = @inbounds s[M]
                @nthreads $N i d begin
                    (@nref $N d i) = (@ncall $N $cdg f sM n i)
                end
                return d
            end
        end
        for M ∈ list(1,N)
            @eval function $cdg(f::GridBundle{$N},n::Val{$M},dt::AbstractVector,s::Tuple=size(f))
                d = Array{pointtype(f),$N}(undef,s...)
                sM = @inbounds s[$M]
                @nthreads $N i d begin
                    (@nref $N d i) = (@ncall $N $cdg f sM n i)/dt[$(Symbol(:i_,M))]
                end
                return d
            end
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

function centraldiff_slow_calc(f::GridBundle{M,T,PA,<:OpenTopology} where {M,T,PA},l::Int,n::Val{N},i::Vararg{Int}) where N #l=size(f)[N]
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
function Cartan.centraldiff_slow_calc(f::GridBundle,l::Int,n::Val{N},i::Vararg{Int}) where N #l=size(f)[N]
    if isone(i[N])
        r = immersion(f).r[2N-1]
        if iszero(r)
            18f[1,n,i...]-9f[2,n,i...]+2f[3,n,i...]-11points(f)[i...]
        elseif immersion(f).p[r]≠2N-1
            f[-2,n,i...]+7(f[0,n,i...]-points(f)[i...])+8(f[1,n,i...]-f[-1,n,i...])-f[2,n,i...]
        else
            (-f[-2,n,i...])+7(-f[0,n,i...]-points(f)[i...])+8(f[1,n,i...]+f[-1,n,i...])-f[2,n,i...]
        end
    elseif i[N]==l
        r = immersion(f).r[2N]
        if iszero(r)
            11points(f)[i...]-18f[-1,n,i...]+9f[-2,n,i...]-2f[-3,n,i...]
        elseif immersion(f).p[r]≠2N
            f[-2,n,i...]+8(f[1,n,i...]-f[-1,n,i...])+7(points(f)[i...]-f[0,n,i...])-f[2,n,i...]
        else
            f[-2,n,i...]+8(-f[1,n,i...]-f[-1,n,i...])+7(points(f)[i...]-f[0,n,i...])+f[2,n,i...]
        end
    elseif i[N]==2
        r = immersion(f).r[2N-1]
        if iszero(r)
            6f[1,n,i...]-f[2,n,i...]-3points(f)[i...]-2f[-1,n,i...]
        elseif immersion(f).p[r]≠2N-1
            f[-2,n,i...]-f[-1,n,i...]+8f[1,n,i...]-7getpoint(f,-1,n,i...)-f[2,n,i...]
        else
            (-f[-2,n,i...])+f[-1,n,i...]+8f[1,n,i...]-7getpoint(f,-1,n,i...)-f[2,n,i...]
        end
    elseif i[N]==l-1
        r = immersion(f).r[2N]
        if iszero(r)
            3points(f)[i...]-6f[-1,n,i...]+f[-2,n,i...]+2f[1,n,i...]
        elseif immersion(f).p[r]≠2N
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
function centraldiff_slow_calc(i::Int,q::QuotientTopology,dt::Real,l::Int)
    if (i∈(1,2) && iszero(q.r[1])) || (i∈(l-1,l) && iszero(q.r[2]))
        6dt # (8-2)*dt
    else
        12dt # (8-2)*2dt
    end
end

function centraldiff_fast_calc(f::GridBundle{M,T,PA,<:OpenTopology} where {M,T,PA},l::Int,n::Val{N},i::Vararg{Int}) where N
    if isone(i[N]) # 4f[1,k,i...]-f[2,k,i...]-3f.v[i...]
        18f[1,n,i...]-9f[2,n,i...]+2f[3,n,i...]-11points(f)[i...]
    elseif i[N]==l # 3f.v[i...]-4f[-1,k,i...]+f[-2,k,i...]
        11points(f)[i...]-18f[-1,n,i...]+9f[-2,n,i...]-2f[-3,n,i...]
    else
        f[1,n,i...]-f[-1,n,i...]
    end
end
function Cartan.centraldiff_fast_calc(f::GridBundle,l::Int,n::Val{N},i::Vararg{Int}) where N
    if isone(i[N])
        r = immersion(f).r[2N-1]
        if iszero(r)
            18f[1,n,i...]-9f[2,n,i...]+2f[3,n,i...]-11points(f)[i...]
        elseif immersion(f).p[r]≠2N-1
            (f[1,n,i...]-points(f)[i...])+(f[0,n,i...]-f[-1,n,i...])
        else # mirror
            (f[1,n,i...]-points(f)[i...])-(f[0,n,i...]-f[-1,n,i...])
        end
    elseif i[N]==l
        r = immersion(f).r[2N]
        if iszero(r)
            11points(f)[i...]-18f[-1,n,i...]+9f[-2,n,i...]-2f[-3,n,i...]
        elseif immersion(f).p[r]≠2N
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
function centraldiff_fast_calc(i::Int,q::QuotientTopology,dt::Real,l::Int)
    if (isone(i) && iszero(q.r[1])) || (i==l && iszero(q.r[2]))
        6dt
    else
        2dt
    end
end

qnumber(n,q) = (q^n-1)/(q-1)
qfactorial(n,q) = prod(cumsum([q^k for k ∈ 0:n-1]))
qbinomial(n,k,q) = qfactorial(n,q)/(qfactorial(n-k,q)*qfactorial(k,q))

richardson(k) = [(-1)^(j+k)*2^(j*(1+j))*qbinomial(k,j,4)/prod([4^n-1 for n ∈ 1:k]) for j ∈ k:-1:0]

# parallelization

select1(n,j,k=:k,f=:f) = :($f[$([i≠j ? :(:) : k for i ∈ 1:n]...)])
select2(n,j,k=:k,f=:f) = :($f[$([i≠j ? :(:) : k for i ∈ 1:n if i≠j]...)])
psum(A,j) = psum(A,Val(j))
pcumsum(A,j) = pcumsum(A,Val(j))
for N ∈ list(2,5)
    for J ∈ list(1,N)
        @eval function psum(A::AbstractArray{T,$N} where T,::Val{$J})
            S = similar(A);
            @threads for k in axes(A,$J)
                @views sum!($(select1(N,J,:k,:S)),$(select1(N,J,:k,:A)),dims=$J)
            end
            return S
        end
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

arclength(f::DenseVector) = sum(value.(abs.(diff(f))))
trapz(f::IntervalMap,d::AbstractVector=diff(points(f))) = sum((d/2).*(f.cod[2:end]+f.cod[1:end-1]))
trapz1(f::DenseVector,h::Real) = h*((f[1]+f[end])/2+sum(f[2:end-1]))
trapz(f::IntervalMap,j::Int) = trapz(f,Val(j))
trapz(f::IntervalMap,j::Val{1}) = trapz(f)
trapz(f::ParametricMap,j::Int) = trapz(f,Val(j))
trapz(f::ParametricMap,j::Val{J}) where J = remove(base(f),j) → trapz2(fiber(f),j,diff(points(f).v[J]))
trapz(f::ParametricMap{B,F,N,<:AlignedSpace} where {B,F,N},j::Val{J}) where J = remove(base(f),j) → trapz1(fiber(f),j,step(points(f).v[J]))
gentrapz1(n,j,h=:h,f=:f) = :($h*(($(select1(n,j,1,f))+$(select1(n,j,:(size(f)[$j]),f)))/2+$(select1(n,j,1,:(sum($(select1(n,j,:(2:$(:end)-1),f)),dims=$j))))))
@generated function trapz1(f::DenseArray{T,N} where T,::Val{J},h::Real,s::Tuple=size(f)) where {N,J}
    gentrapz1(N,J)
end
@generated function trapz1(f::DenseArray{T,N} where T,h::D...) where {N,D<:Real}
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
    trapz1(fiber(f),step(points(f)))
end
for N ∈ list(2,5)
    @eval function trapz(f::ParametricMap{B,F,$N,<:AlignedSpace{$N}} where {B,F})
        trapz1(fiber(f),$([:(step(points(f).v[$j])) for j ∈ 1:N]...))
    end
    @eval function trapz(m::ParametricMap{B,F,$N,T} where {B,F,T},D::AbstractVector=diff.(points(m).v))
        c = fiber(m)
        f,s,d = similar(c),size(c),D./2
        $(Expr(:block,vcat([gentrapz2(j,j,j≠N ? :f : :c).args for j ∈ N:-1:1]...)...))
    end
    for J ∈ list(1,N)
        @eval function trapz2(c::DenseArray{T,$N} where T,j::Val{$J},D)
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

function arcresample(f::IntervalMap,i::Int=length(f))
    at = arctime(f)
    ts = at.(LinRange(0,points(at)[end],i))
    TensorField(ts, f.(ts))
end
function arcsample(f::IntervalMap,i::Int=length(f))
    at = arctime(f)
    ral = LinRange(0,points(at)[end],i)
    ts = at.(ral)
    TensorField(ral, f.(ts))
end

arcparametrize(f::IntervalMap) = TensorField(fiber(arclength(f)),fiber(f))
arctime(f) = TensorField(fiber(arclength(f)), points(f))
arcsteps(f::IntervalMap) = Real.(abs.(diff(fiber(f)),refdiff(metricextensor(f))))
totalarclength(f::IntervalMap) = sum(arcsteps(f))
function arclength(f::IntervalMap)
    int = cumsum(arcsteps(f))
    pushfirst!(int,zero(eltype(int)))
    TensorField(base(f), int)
end # cumtrapz(speed(f))
function cumtrapz(f::IntervalMap,d::AbstractVector=diff(points(f)))
    i = (d/2).*cumsum(f.cod[2:end]+f.cod[1:end-1])
    pushfirst!(i,zero(eltype(i)))
    TensorField(base(f), i)
end
function cumtrapz1(f::DenseVector,h::Real)
    i = (h/2)*cumsum(f[2:end]+f[1:end-1])
    pushfirst!(i,zero(eltype(i)))
    return i
end
cumtrapz(f::IntervalMap,j::Int) = cumtrapz(f,Val(j))
cumtrapz(f::IntervalMap,j::Val{1}) = cumtrapz(f)
cumtrapz(f::ParametricMap,j::Int) = cumtrapz(f,Val(j))
cumtrapz(f::ParametricMap,j::Val{J}) where J = TensorField(base(f), cumtrapz2(fiber(f),j,diff(points(f).v[J])))
cumtrapz(f::ParametricMap{B,F,N,<:AlignedSpace{N}} where {B,F,N},j::Val{J}) where J = TensorField(base(f), cumtrapz1(fiber(f),j,step(points(f).v[J])))
selectzeros(n,j) = :(zeros(T,$([i≠j ? :(s[$i]) : 1 for i ∈ 1:n]...)))
selectzeros2(n,j) = :(zeros(T,$([i≠j ? i<j ? :(s[$i]) : :(s[$i]-1) : 1 for i ∈ 1:n]...)))
gencat(n,j=n,cat=n≠2 ? :cat : j≠2 ? :vcat : :hcat) = :($cat($(selectzeros2(n,j)),$(j≠1 ? gencat(n,j-1) : :i);$((cat≠:cat ? () : (Expr(:kw,:dims,j),))...)))
gencumtrapz1(n,j,h=:h,f=:f) = :(($h/2)*cumsum($(select1(n,j,:(2:$(:end)),f)).+$(select1(n,j,:(1:$(:end)-1),f)),dims=$j))
@generated function cumtrapz1(f::DenseArray{T,N},::Val{J},h::Real,s::Tuple=size(f)) where {T,N,J}
    :(cat($(selectzeros(N,J)),$(gencumtrapz1(N,J)),dims=$J))
end
@generated function cumtrapz1(f::DenseArray{T,N},h::D...) where {T,N,D<:Real}
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
    TensorField(base(f), cumtrapz1(fiber(f),step(points(f))))
end
for N ∈ 2:5
    @eval function cumtrapz(f::ParametricMap{B,F,$N,<:AlignedSpace{$N}} where {B,F})
        TensorField(base(f), cumtrapz1(fiber(f),$([:(step(points(f).v[$j])) for j ∈ 1:N]...)))
    end
    @eval function cumtrapz(m::ParametricMap{B,F,$N,T} where {B,F},D::AbstractVector=diff.(points(m).v)) where T
        f = fiber(m)
        s,d = size(f),D./2
        $(Expr(:block,vcat([gencumtrapz2(N,j,:(d[$j])).args for j ∈ 1:N]...)...))
        TensorField(base(m), $(gencat(N)))
    end
    for J ∈ 1:N
        @eval function cumtrapz2(c::DenseArray{T,$N},::Val{$J}) where T
            s,d = size(f),D/2
            $(gencumtrapz2(N,J,:d,:f))
            cat($(selectzeros(N,J)),i,dims=$J)
        end
    end
end
function linecumtrapz(γ::IntervalMap,f::Function)
    cumtrapz(TensorField(base(γ),f.(fiber(γ)).⋅fiber(gradient(γ))))
end

