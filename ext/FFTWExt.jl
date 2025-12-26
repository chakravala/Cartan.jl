module FFTWExt

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
isdefined(Cartan, :Requires) ? (import Cartan: FFTW) : (using FFTW)

for fun ∈ (:dct,:dct!,:idct,:idct!)
    @eval FFTW.$fun(t::TensorField,args...) = TensorField(r2rspace(base(t)), FFTW.$fun(fiber(t),args...))
end
for fun ∈ (:r2r,:r2r!)
    @eval FFTW.$fun(t::TensorField,kind,args...) = TensorField(r2rspace(base(t),kind),FFTW.$fun(fiber(t),kind,args...))
end

end # module
