
export fwi_objective

function fwi_objective(model_full::Model, source::judiVector, dObs::judiVector, srcnum::Integer; options=Options(), frequencies=[])
# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to OPESCI/devito

    # Load full geometry for out-of-core geometry containers
    typeof(dObs.geometry) == GeometryOOC && (dObs.geometry = Geometry(dObs.geometry))
    typeof(source.geometry) == GeometryOOC && (source.geometry = Geometry(source.geometry))
    length(model_full.n) == 3 ? dims = [3,2,1] : dims = [2,1]   # model dimensions for Python are (z,y,x) and (z,x)

    # for 3D modeling, limit model to area with sources/receivers
    if options.limit_m == true && length(model_full.n) == 3 # only supported for 3D
        model = deepcopy(model_full)
        model = limit_model_to_receiver_area(source.geometry,dObs.geometry,model,options.buffer_size)
    else
        model = model_full
    end

    # Source/receiver parameters
    tmaxSrc = source.geometry.t[1]
    tmaxRec = dObs.geometry.t[1]

    # Set up Python model structure (force origin to be zero due to current devito bug)
    modelPy = pm.Model(origin=(0.,0.,0.), spacing=model.d, shape=model.n, vp=process_physical_parameter(sqrt.(1f0./model.m), dims), nbpml=model.nb,
                       rho=process_physical_parameter(model.rho, dims), space_order=options.space_order)
    dtComp = modelPy[:critical_dt]

    # Extrapolate input data to computational grid
    qIn = time_resample(source.data[1],source.geometry,dtComp)[1]
    if typeof(dObs.data[1]) == SeisIO.SeisCon
        data = convert(Array{Float32,2},dObs.data[1][1].data)
        dObs = judiVector(dObs.geometry,data)
    end
    dObserved = time_resample(dObs.data[1],dObs.geometry,dtComp)[1]
    ntComp = size(dObserved,2)
    ntSrc = Int(trunc(tmaxSrc/dtComp+1))
    ntRec = Int(trunc(tmaxRec/dtComp+1))

    # Set up coordinates
    src_coords = setup_grid(source.geometry, model.n, model.o)  # shifts source coordinates by origin
    rec_coords = setup_grid(dObs.geometry, model.n, model.o)    # shifts rec coordinates by origin

    # Forward modeling to generate synthetic data and background wavefields
    if options.optimal_checkpointing == true
        op_F = pycall(ac.forward_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'), op_return=true)
        argout1, argout2 = pycall(ac.adjoint_born, PyObject, modelPy, PyReverseDims(rec_coords'), PyReverseDims(dObserved'),
                                  op_forward=op_F, is_residual=false, n_checkpoints=options.num_checkpoints, maxmem=options.checkpoints_maxmem)
    elseif ~isempty(options.frequencies)
                typeof(options.frequencies) == Array{Any,1} && (options.frequencies = options.frequencies[srcnum])
                dPredicted, uf_real, uf_imag = pycall(ac.forward_freq_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'),
                                                      options.frequencies, space_order=options.space_order, nb=model.nb, factor=options.dft_subsampling_factor)
                argout1 = .5f0*dot(vec(dPredicted) - vec(dObserved), vec(dPredicted) - vec(dObserved))*dtComp  # FWI objective function value
                argout2 = pycall(ac.adjoint_freq_born, Array{Float32, length(model.n)}, modelPy, PyReverseDims(rec_coords'), PyReverseDims((dPredicted - dObserved)'),
                                 options.frequencies, factor=options.dft_subsampling_factor, uf_real, uf_imag, space_order=options.space_order, nb=model.nb)
    else
        dPredicted, u0 = pycall(ac.forward_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'), save=true)
        argout1 = .5f0*dot(vec(dPredicted) - vec(dObserved), vec(dPredicted) - vec(dObserved))*dtComp # FWI objective function value
        argout2 = pycall(ac.adjoint_born, Array{Float32}, modelPy, PyReverseDims(rec_coords'), PyReverseDims((dPredicted  - dObserved)'),
                         u=u0, is_residual=true)
    end
    argout2 = remove_padding(argout2, model.nb, true_adjoint=options.sum_padding)
    if options.limit_m==true && length(model_full.n) == 3
        argout2 = extend_gradient(model_full,model,argout2)
    end
    return [argout1; vec(argout2)]
end


######################################################################################################################################################################################################################
# Hacked version for GCP-demo: generate observed data on the fly
#

function fwi_objective(model_full::Model, source::judiVector, F::judiPDEfull, srcnum::Integer; options=Options(), frequencies=[])
# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to OPESCI/devito

    # Forward model data
    typeof(source.geometry) == GeometryOOC && (source.geometry = Geometry(source.geometry))
    println("Generate observed data")
    dObs = F*source
    println("Done")

    # Load full geometry for out-of-core geometry containers
    typeof(dObs.geometry) == GeometryOOC && (dObs.geometry = Geometry(dObs.geometry))
    length(model_full.n) == 3 ? dims = [3,2,1] : dims = [2,1]   # model dimensions for Python are (z,y,x) and (z,x)

    # for 3D modeling, limit model to area with sources/receivers
    if options.limit_m == true && length(model_full.n) == 3 # only supported for 3D
        model = deepcopy(model_full)
        model = limit_model_to_receiver_area(source.geometry,dObs.geometry,model,options.buffer_size)
    else
        model = model_full
    end

    # Source/receiver parameters
    tmaxSrc = source.geometry.t[1]
    tmaxRec = dObs.geometry.t[1]

    # Set up Python model structure (force origin to be zero due to current devito bug)
    modelPy = pm.Model(origin=(0.,0.,0.), spacing=model.d, shape=model.n, vp=process_physical_parameter(sqrt.(1f0./model.m), dims), nbpml=model.nb,
                       rho=process_physical_parameter(model.rho, dims), space_order=options.space_order)
    dtComp = modelPy[:critical_dt]

    # Extrapolate input data to computational grid
    qIn = time_resample(source.data[1],source.geometry,dtComp)[1]
    if typeof(dObs.data[1]) == SeisIO.SeisCon
        data = convert(Array{Float32,2},dObs.data[1][1].data)
        dObs = judiVector(dObs.geometry,data)
    end
    dObserved = time_resample(dObs.data[1],dObs.geometry,dtComp)[1]
    ntComp = size(dObserved,2)
    ntSrc = Int(trunc(tmaxSrc/dtComp+1))
    ntRec = Int(trunc(tmaxRec/dtComp+1))

    # Set up coordinates
    src_coords = setup_grid(source.geometry, model.n, model.o)  # shifts source coordinates by origin
    rec_coords = setup_grid(dObs.geometry, model.n, model.o)    # shifts rec coordinates by origin

    # Forward modeling to generate synthetic data and background wavefields
    if options.optimal_checkpointing == true
        op_F = pycall(ac.forward_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'), op_return=true)
        argout1, argout2 = pycall(ac.adjoint_born, PyObject, modelPy, PyReverseDims(rec_coords'), PyReverseDims(dObserved'),
                                  op_forward=op_F, is_residual=false, n_checkpoints=options.num_checkpoints, maxmem=options.checkpoints_maxmem)
    elseif ~isempty(options.frequencies)
                typeof(options.frequencies) == Array{Any,1} && (options.frequencies = options.frequencies[srcnum])
                dPredicted, uf_real, uf_imag = pycall(ac.forward_freq_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'),
                                                      options.frequencies, space_order=options.space_order, nb=model.nb, factor=options.dft_subsampling_factor)
                argout1 = .5f0*dot(vec(dPredicted) - vec(dObserved), vec(dPredicted) - vec(dObserved))*dtComp  # FWI objective function value
                argout2 = pycall(ac.adjoint_freq_born, Array{Float32, length(model.n)}, modelPy, PyReverseDims(rec_coords'), PyReverseDims((dPredicted - dObserved)'),
                                 options.frequencies, factor=options.dft_subsampling_factor, uf_real, uf_imag, space_order=options.space_order, nb=model.nb)
    else
        dPredicted, u0 = pycall(ac.forward_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'), save=true)
        argout1 = .5f0*dot(vec(dPredicted) - vec(dObserved), vec(dPredicted) - vec(dObserved))*dtComp # FWI objective function value
        argout2 = pycall(ac.adjoint_born, Array{Float32}, modelPy, PyReverseDims(rec_coords'), PyReverseDims((dPredicted  - dObserved)'),
                         u=u0, is_residual=true)
    end
    argout2 = remove_padding(argout2, model.nb, true_adjoint=options.sum_padding)
    if options.limit_m==true && length(model_full.n) == 3
        argout2 = extend_gradient(model_full,model,argout2)
    end
    return [argout1; vec(argout2)]
end



