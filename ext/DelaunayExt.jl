module DelaunayExt

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
isdefined(Cartan, :Requires) ? (import Cartan: Delaunay) : (using Delaunay)

Delaunay.delaunay(p::PointCloud) = Delaunay.delaunay(points(p))
Delaunay.delaunay(p::Vector{<:Chain}) = initmesh(Delaunay.delaunay(submesh(p)))
Cartan.initmesh(t::Delaunay.Triangulation) = Cartan.initmeshdata(t.points',t.convex_hull',t.simplices')

end # module
