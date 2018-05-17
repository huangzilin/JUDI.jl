# Unit tests for JUDI linear operators
# Philipp Witte (pwitte.slim@gmail.com)
# May 2018
#

using JUDI.TimeModeling, SeisIO, Base.Test

example_info(; n=(120,100), nsrc=2, ntComp=1000) = Info(prod(n), nsrc, ntComp)
example_model(; n=(120,100), d=(10f0, 10f0), o=(0f0, 0f0), m=randn(Float32, n)) = Model(n, d, o, m)

# judiModeling
info = example_info()
model = example_model()
F_forward = judiModeling(info, model; options=Options())
F_adjoint = judiModelingAdjoint(info, model; options=Options())

@test isequal(typeof(F_forward), judiModeling{Float32, Float32})
@test isequal(typeof(F_adjoint), judiModelingAdjoint{Float32, Float32})
@test isequal(size(F_forward), reverse(size(F_adjoint)))

# conj, transpose, ctranspose
@test isequal(size(F_forward), size(conj(F_forward)))
@test isequal(size(F_adjoint), size(conj(F_adjoint)))

@test isequal(reverse(size(F_forward)), size(transpose(F_forward)))
@test isequal(reverse(size(F_adjoint)), size(transpose(F_adjoint)))

@test isequal(reverse(size(F_forward)), size(ctranspose(F_forward)))
@test isequal(reverse(size(F_adjoint)), size(ctranspose(F_adjoint)))

F_sub = F_forward[1]
@test isequal(F_sub.info.nsrc, 1)
@test isequal(F_sub.model, F_forward.model)
@test isequal(size(F_sub), convert(Tuple{Int64, Int64}, size(F_forward) ./ 2))

F_sub = F_forward[1:2]
@test isequal(F_sub.info.nsrc, 2)
@test isequal(F_sub.model, F_forward.model)
@test isequal(size(F_sub), size(F_forward))


