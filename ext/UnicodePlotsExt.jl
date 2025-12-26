module UnicodePlotsExt

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

using Grassmann, Cartan
isdefined(Cartan, :Requires) ? (import Cartan: UnicodePlots) : (using UnicodePlots)

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
UnicodePlots.lineplot(t::ScalarMap;args...) = UnicodePlots.lineplot(getindex.(base(t),2),fiber(t);args...)
UnicodePlots.lineplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::ScalarMap;args...) = UnicodePlots.lineplot!(p,getindex.(base(t),2),fiber(t);args...)
UnicodePlots.lineplot(t::PlaneCurve;args...) = UnicodePlots.lineplot(getindex.(fiber(t),1),getindex.(fiber(t),2);args...)
UnicodePlots.lineplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::PlaneCurve;args...) = UnicodePlots.lineplot!(p,getindex.(fiber(t),1),getindex.(fiber(t),2);args...)
UnicodePlots.lineplot(t::RealFunction;args...) = UnicodePlots.lineplot(Real.(points(t)),Real.(fiber(t));args...)
UnicodePlots.lineplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::RealFunction;args...) = UnicodePlots.lineplot!(p,Real.(points(t)),Real.(fiber(t));args...)
UnicodePlots.lineplot(t::ComplexMap{B,<:AbstractComplex,1};args...) where B<:Coordinate{<:AbstractReal} = UnicodePlots.lineplot(real.(Complex.(fiber(t))),imag.(Complex.(fiber(t)));args...)
UnicodePlots.lineplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::ComplexMap{B,<:AbstractComplex,1};args...) where B<:Coordinate{<:AbstractReal} = UnicodePlots.lineplot!(p,real.(Complex.(fiber(t))),imag.(Complex.(fiber(t)));args...)
UnicodePlots.lineplot(t::GradedField{G,B,F,1} where {G,F};args...) where B<:Coordinate{<:AbstractReal} = UnicodePlots.lineplot(Real.(points(t)),Grassmann.array(fiber(t));args...)
UnicodePlots.lineplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::GradedField{G,B,F,1} where {G,F};args...) where B<:Coordinate{<:AbstractReal} = UnicodePlots.lineplot!(p,Real.(points(t)),Grassmann.array(fiber(t));args...)
UnicodePlots.polarplot(t::RealFunction;args...) = UnicodePlots.polarplot(Real.(points(t)),Real.(fiber(t));args...)
UnicodePlots.polarplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::RealFunction;args...) = UnicodePlots.polarplot!(p,Real.(points(t)),Real.(fiber(t));args...)
UnicodePlots.scatterplot(t::TensorField;args...) = UnicodePlots.scatterplot(fiber(t);args...)
UnicodePlots.scatterplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::TensorField;args...) = UnicodePlots.scatterplot!(p,fiber(t);args...)
UnicodePlots.scatterplot(t::RealFunction;args...) = UnicodePlots.scatterplot(Real.(points(t)),Real.(fiber(t));args...)
UnicodePlots.scatterplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::RealFunction;args...) = UnicodePlots.scatterplot!(p,Real.(points(t)),Real.(fiber(t));args...)
UnicodePlots.scatterplot(t::AbstractArray{<:Chain{V,G,K,2} where {V,G,K}};args...) = UnicodePlots.scatterplot(getindex.(vec(t),1),getindex.(vec(t),2);args...)
UnicodePlots.scatterplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::AbstractArray{<:Chain{V,G,K,2} where {V,G,K}};args...) = UnicodePlots.scatterplot!(p,getindex.(vec(t),1),getindex.(vec(t),2);args...)
UnicodePlots.densityplot(t::TensorField;args...) = UnicodePlots.densityplot(fiber(t);args...)
UnicodePlots.densityplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::TensorField;args...) = UnicodePlots.densityplot!(p,fiber(t);args...)
UnicodePlots.densityplot(t::AbstractArray{<:Chain{V,G,K,2} where {V,G,K}};args...) = UnicodePlots.densityplot(getindex.(vec(t),1),getindex.(vec(t),2);args...)
UnicodePlots.densityplot!(p::UnicodePlots.Plot{<:UnicodePlots.Canvas},t::AbstractArray{<:Chain{V,G,K,2} where {V,G,K}};args...) = UnicodePlots.densityplot!(p,getindex.(vec(t),1),getindex.(vec(t),2);args...)
UnicodePlots.contourplot(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F};args...) = UnicodePlots.contourplot(points(t).v[1][2:end-1],points(t).v[2][2:end-1],(x,y)->radius(t(Chain(x,y)));args...)
UnicodePlots.contourplot(t::SurfaceGrid;args...) = UnicodePlots.contourplot(points(t).v[1][2:end-1],points(t).v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
UnicodePlots.surfaceplot(t::SurfaceGrid;args...) = UnicodePlots.surfaceplot(points(t).v[1][2:end-1],points(t).v[2][2:end-1],(x,y)->t(Chain(x,y));args...)
UnicodePlots.surfaceplot(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F};args...) = UnicodePlots.surfaceplot(points(t).v[1][2:end-1],points(t).v[2][2:end-1],(x,y)->radius(t(Chain(x,y)));colormap=:twilight,args...)
UnicodePlots.isosurface(t::VolumeGrid;args...) = UnicodePlots.isosurface(points(t).v[1][2:end-1],points(t).v[2][2:end-1],points(t).v[3][2:end-1],(x,y,z)->t(Chain(x,y,z));args...)
UnicodePlots.histogram(t::ScalarField;args...) = UnicodePlots.histogram(Real.(vec(fiber(t)));args...)
UnicodePlots.boxplot(t::ScalarField;args...) = UnicodePlots.boxplot(Real.(vec(fiber(t)));args...)
UnicodePlots.boxplot(t::TensorField{B,<:Chain} where B;args...) = UnicodePlots.boxplot(fiber(t);args...)
UnicodePlots.boxplot(t::AbstractVector{<:Chain{V,G,K,N} where K};args...) where {V,G,N} = UnicodePlots.boxplot(string.(Grassmann.chainbasis(V,G)),[getindex.(t,k) for k ∈ 1:N];args...)
UnicodePlots.boxplot(t::AbstractArray{<:Chain{V,G,K,N} where K};args...) where {V,G,N} = UnicodePlots.boxplot(vec(t);args...)
UnicodePlots.spy(t::SurfaceGrid;args...) = UnicodePlots.spy(Real.(fiber(t));args...)
UnicodePlots.spy(p::SimplexBundle) = UnicodePlots.spy(antiadjacency(p))
UnicodePlots.heatmap(t::SurfaceGrid;args...) = UnicodePlots.heatmap(Real.(fiber(t));xfact=step(points(t).v[1]),yfact=step(points(t).v[2]),xoffset=points(t).v[1][1],yoffset=points(t).v[2][1],args...)
UnicodePlots.heatmap(t::ComplexMap{B,F,2,<:RealSpace{2}} where {B,F};args...) = UnicodePlots.heatmap(Real.(angle.(fiber(t)));xfact=step(points(t).v[1]),yfact=step(points(t).v[2]),xoffset=points(t).v[1][1],yoffset=points(t).v[2][1],colormap=:twilight,args...)
Base.display(t::PlaneCurve) = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
Base.display(t::RealFunction) = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
Base.display(t::ComplexMap{B,<:AbstractComplex,1,<:Interval}) where B = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
Base.display(t::ComplexMap{B,<:AbstractComplex,2,<:RealSpace{2}} where B) = (display(typeof(t)); display(UnicodePlots.heatmap(t)))
Base.display(t::GradedField{G,B,<:TensorGraded,1,<:Interval} where {G,B}) = (display(typeof(t)); display(UnicodePlots.lineplot(t)))
Base.display(t::SurfaceGrid) = (display(typeof(t)); display(UnicodePlots.heatmap(t)))

end # module
