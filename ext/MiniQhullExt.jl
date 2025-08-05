module MiniQhullExt

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
isdefined(Cartan, :Requires) ? (import Cartan: MiniQhull) : (using MiniQhull)

MiniQhull.delaunay(p::Vector{<:Chain},args...) = MiniQhull.delaunay(PointCloud(p),1:length(p),args...)
MiniQhull.delaunay(p::Vector{<:Chain},n::AbstractVector,args...) = MiniQhull.delaunay(PointCloud(p),n,args...)
MiniQhull.delaunay(p::PointCloud,args...) = MiniQhull.delaunay(p,1:length(p),args...)
function MiniQhull.delaunay(p::PointCloud,n::AbstractVector,args...)
    N,m = mdims(p),length(n)
    l = Cartan.list(1,N)
    T = MiniQhull.delaunay(Matrix(submesh(m==length(p) ? p : fullpoints(p)[n])'),args...)
    p(SimplexTopology([Values{N,Int}(getindex.(Ref(n),Int.(T[l,k]))) for k ∈ 1:size(T,2)],length(p)))
end

end # module
