module QHullExt

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
isdefined(Cartan, :Requires) ? (import Cartan: QHull) : (using QHull)

QHull.chull(p::Vector{<:Chain},n=1:length(p)) = QHull.chull(PointCloud(p),n)
function QHull.chull(p::PointCloud,n=1:length(p))
    T = QHull.chull(submesh(length(n)==length(p) ? p : p[n]))
    p(SimplexTopology([Values(getindex.(Ref(n),k)) for k ∈ T.simplices],length(p)))
end
function Cartan.SimplexBundle(t::Chull)
    p = PointCloud(Cartan.initpoints(t.points'))
    p(SimplexTopology(Values.(t.simplices),length(p)))
end

end # module
