module MeshesExt

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
isdefined(Cartan, :Requires) ? (import Cartan: Meshes) : (using Meshes)

function Cartan.SimplexBundle(m::Meshes.SimpleMesh{N}) where N
    c,f = Meshes.vertices(m),m.topology.connec
    s = N+1; V = Submanifold(ℝ^s) # s
    n = length(f[1].indices)
    p = PointCloud([Chain{V,1}(Values{s,Float64}(1.0,k.coords...)) for k ∈ c])
    p(SimplexTopology([Values{n,Int}(k.indices) for k ∈ f],length(p)))
end

end # module
