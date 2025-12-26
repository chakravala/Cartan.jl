module ToeplitzMatricesExt

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
isdefined(Cartan, :Requires) ? (import Cartan: ToeplitzMatrices) : (using ToeplitzMatrices)

Cartan.derivetoeplitz(N,h=2pi/N,c=Cartan.toeplitz(N,h)) = ToeplitzMatrices.Toeplitz(c,-c)
Cartan.derivetoeplitz2(N,h=2pi/N,c=Cartan.toeplitz2(N,h)) = ToeplitzMatrices.Toeplitz(c,c)

end # module
