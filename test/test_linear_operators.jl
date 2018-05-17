# Unit tests for JUDI linear operators (without PDE solves)
# Philipp Witte (pwitte.slim@gmail.com)
# May 2018
#

using JUDI.TimeModeling, SeisIO, Base.Test

# Example structures

example_info(; n=(120,100), nsrc=2, ntComp=1000) = Info(prod(n), nsrc, ntComp)
example_model(; n=(120,100), d=(10f0, 10f0), o=(0f0, 0f0), m=randn(Float32, n)) = Model(n, d, o, m)

function example_rec_geometry(; nsrc=2, nrec=120)
    xrec = linspace(50f0, 1150f0, nrec)
    yrec = 0f0
    zrec = linspace(50f0, 50f0, nrec)
    return Geometry(xrec, yrec, zrec; dt=4f0, t=1000f0, nsrc=nsrc)
end

function example_src_geometry(; nsrc=2)
    xrec = linspace(100f0, 1000f0, nsrc)
    yrec = 0f0
    zrec = linspace(50f0, 50f0, nsrc)
    return Geometry(xrec, yrec, zrec; dt=4f0, t=1000f0, nsrc=nsrc)
end

# Tests

function test_transpose(Op)
    @test isequal(size(Op), size(conj(Op)))
    @test isequal(reverse(size(Op)), size(transpose(Op)))
    @test isequal(reverse(size(Op)), size(ctranspose(Op)))
    @test isequal(reverse(size(Op)), size(transpose(Op)))
    return true
end

function test_getindex(Op)
    # requires: Op.info.nsrc == 2
    Op_sub = Op[1]
    @test isequal(Op_sub.info.nsrc, 1)
    @test isequal(Op_sub.model, Op.model)
    @test isequal(size(Op_sub), convert(Tuple{Int64, Int64}, size(Op) ./ 2))

    Op_sub = Op[1:2]
    @test isequal(Op_sub.info.nsrc, 2)
    @test isequal(Op_sub.model, Op.model)
    @test isequal(size(Op_sub), size(Op))
    return true
end

########################################################### judiModeling ###############################################

info = example_info()
model = example_model()
F_forward = judiModeling(info, model; options=Options())
F_adjoint = judiModelingAdjoint(info, model; options=Options())

@test isequal(typeof(F_forward), judiModeling{Float32, Float32})
@test isequal(typeof(F_adjoint), judiModelingAdjoint{Float32, Float32})

# conj, transpose, ctranspose
@test test_transpose(F_forward)
@test test_transpose(F_adjoint)

# get index
@test test_getindex(F_forward)
@test test_getindex(F_adjoint)

############################################################# judiPDE ##################################################

info = example_info()
model = example_model()
rec_geometry = example_rec_geometry()

PDE_forward = judiPDE("PDE", info, model, rec_geometry; options=Options())
PDE_adjoint = judiPDEadjoint("PDEadjoint", info, model, rec_geometry; options=Options())

@test isequal(typeof(PDE_forward), judiPDE{Float32, Float32})
@test isequal(typeof(PDE_adjoint), judiPDEadjoint{Float32, Float32})

# conj, transpose, ctranspose
@test test_transpose(PDE_forward)
@test test_transpose(PDE_adjoint)

# get index
@test test_getindex(PDE_forward)
@test test_getindex(PDE_adjoint)

# Multiplication w/ judiProjection
src_geometry = example_src_geometry()
Ps = judiProjection(info, src_geometry)

PDE = PDE_forward*Ps'
@test isequal(typeof(PDE), judiPDEfull{Float32, Float32})
@test isequal(PDE.recGeometry, rec_geometry)
@test isequal(PDE.srcGeometry, src_geometry)

PDEad = PDE_adjoint*Ps'
@test isequal(typeof(PDEad), judiPDEfull{Float32, Float32})
@test isequal(PDEad.srcGeometry, rec_geometry)
@test isequal(PDEad.recGeometry, src_geometry)



