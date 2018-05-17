# Unit tests for judiVector
# Philipp Witte (pwitte.slim@gmail.com)
# May 2018
#

using JUDI.TimeModeling, SeisIO, Base.Test

function example_rec_geometry(; nsrc=2, nrec=120)
    xrec = linspace(50f0, 1150f0, nrec)
    yrec = 0f0
    zrec = linspace(50f0, 50f0, nrec)
    return Geometry(xrec, yrec, zrec; dt=4f0, t=1000f0, nsrc=nsrc)
end

# number of sources/receivers
nsrc = 2
nrec = 120
ns = 251

################################################# test constructors ####################################################

# set up judiVector fr,om array
dsize = (nsrc*nrec*ns, 1)
rec_geometry = example_rec_geometry(nsrc=nsrc, nrec=nrec)
data = randn(Float32, ns, nrec)
d_obs = judiVector(rec_geometry, data)

@test isequal(d_obs.nsrc, nsrc)
@test isequal(typeof(d_obs.data), Array{Array, 1})
@test isequal(typeof(d_obs.geometry), GeometryIC)
@test iszero(norm(d_obs.data[1] - d_obs.data[2]))
@test isequal(size(d_obs), dsize)

# set up judiVector from cell array
data = Array{Array}(nsrc)
for j=1:nsrc
    data[j] = randn(Float32, 251, nrec)
end
d_obs =  judiVector(rec_geometry, data)

@test isequal(d_obs.nsrc, nsrc)
@test isequal(typeof(d_obs.data), Array{Array, 1})
@test isequal(typeof(d_obs.geometry), GeometryIC)
@test iszero(norm(d_obs.data - d_obs.data))
@test isequal(size(d_obs), dsize)

# contructor for in-core data container
block = segy_read("../data/unit_test_shot_records.segy")
d_block = judiVector(block; segy_depth_key="RecGroupElevation")
dsize = (prod(size(block.data)), 1)

@test isequal(d_block.nsrc, nsrc)
@test isequal(typeof(d_block.data), Array{Array, 1})
@test isequal(typeof(d_block.geometry), GeometryIC)
@test isequal(size(d_block), dsize)

# contructor for in-core data container and given geometry
d_block = judiVector(rec_geometry, block)

@test isequal(d_block.nsrc, nsrc)
@test isequal(typeof(d_block.data), Array{Array, 1})
@test isequal(typeof(d_block.geometry), GeometryIC)
@test compareGeometry(rec_geometry, d_block.geometry)
@test isequal(size(d_block), dsize)

# contructor for out-of-core data container from single container
container = segy_scan("../data/", "unit_test_shot_records", ["GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])
d_cont = judiVector(container; segy_depth_key="RecGroupElevation")

@test isequal(d_cont.nsrc, nsrc)
@test isequal(typeof(d_cont.data), Array{SeisIO.SeisCon, 1})
@test isequal(typeof(d_cont.geometry), GeometryOOC)
@test isequal(size(d_cont), dsize)

# contructor for single out-of-core data container and given geometry
d_cont = judiVector(rec_geometry, container)

@test isequal(d_cont.nsrc, nsrc)
@test isequal(typeof(d_cont.data), Array{SeisIO.SeisCon, 1})
@test isequal(typeof(d_cont.geometry), GeometryIC)
@test compareGeometry(rec_geometry, d_cont.geometry)
@test isequal(size(d_cont), dsize)

# contructor for out-of-core data container from cell array of containers
container_cell = Array{SeisIO.SeisCon}(nsrc)
for j=1:nsrc
    container_cell[j] = split(container, j)
end
d_cont =  judiVector(container_cell; segy_depth_key="RecGroupElevation")

@test isequal(d_cont.nsrc, nsrc)
@test isequal(typeof(d_cont.data), Array{SeisIO.SeisCon, 1})
@test isequal(typeof(d_cont.geometry), GeometryOOC)
@test isequal(size(d_cont), dsize)

# contructor for out-of-core data container from cell array of containers and given geometry
d_cont =  judiVector(rec_geometry, container_cell)

@test isequal(d_cont.nsrc, nsrc)
@test isequal(typeof(d_cont.data), Array{SeisIO.SeisCon, 1})
@test isequal(typeof(d_cont.geometry), GeometryIC)
@test compareGeometry(rec_geometry, d_cont.geometry)
@test isequal(size(d_cont), dsize)


#################################################### test operations ###################################################

# conj, transpose, ctranspose
@test isequal(size(d_obs), size(conj(d_obs)))
@test isequal(size(d_block), size(conj(d_block)))
@test isequal(size(d_cont), size(conj(d_cont)))

@test isequal(reverse(size(d_obs)), size(transpose(d_obs)))
@test isequal(reverse(size(d_block)), size(transpose(d_block)))
@test isequal(reverse(size(d_cont)), size(transpose(d_cont)))

@test isequal(reverse(size(d_obs)), size(ctranspose(d_obs)))
@test isequal(reverse(size(d_block)), size(ctranspose(d_block)))
@test isequal(reverse(size(d_cont)), size(ctranspose(d_cont)))

# +, -, *, /
@test iszero(norm(2*d_obs - (d_obs + d_obs)))
@test iszero(norm(d_obs - (d_obs + d_obs)/2))

@test iszero(norm(2*d_block - (d_block + d_block)))
@test iszero(norm(d_block - (d_block + d_block)/2))

@test iszero(norm(2*d_cont - (d_cont + d_cont)))    # creates in-core judiVector
@test iszero(norm(1*d_cont - (d_cont + d_cont)/2))

# vcat
d_vcat = [d_block; d_block]
@test isequal(length(d_vcat), 2*length(d_block))
@test isequal(d_vcat.nsrc, 2*d_block.nsrc)
@test isequal(d_vcat.geometry.xloc[1], d_block.geometry.xloc[1])

# dot, norm, abs
@test isapprox(norm(d_block), sqrt(dot(d_block, d_block)))
@test isapprox(norm(d_cont), sqrt(dot(d_cont, d_cont)))

# vector space axioms
u = judiVector(rec_geometry, randn(Float32, ns, nrec))
v = judiVector(rec_geometry, randn(Float32, ns, nrec))
w = judiVector(rec_geometry, randn(Float32, ns, nrec))
a = randn(1)[1]
b = randn(1)[1]

@test isapprox(u + (v + w), (u + v) + w; rtol=eps(1f0))
@test isapprox(u + v, v + u; rtol=eps(1f0))
@test isapprox(u, u + 0; rtol=eps(1f0))
@test iszero(norm(u + u*(-1)))
@test isapprox(a*(b*u), (a*b)*u; rtol=eps(1f0))
@test isapprox(u, u*1; rtol=eps(1f0))
@test isapprox(a*(u + v), a*u + a*v; rtol=eps(1f0))
@test isapprox((a + b)*v, a*v + b*v; rtol=eps(1f0))


