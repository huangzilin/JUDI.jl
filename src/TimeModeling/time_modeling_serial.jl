
export time_modeling

# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to OPESCI/devito
function time_modeling(model_full::Model, srcGeometry, srcData, recGeometry, recData, dm, srcnum::Int64, op::Char, mode::Int64, options)

    # Load full geometry for out-of-core geometry containers
    typeof(recGeometry) == GeometryOOC && (recGeometry = Geometry(recGeometry))
    typeof(srcGeometry) == GeometryOOC && (srcGeometry = Geometry(srcGeometry))
    length(model_full.n) == 3 ? dims = [3,2,1] : dims = [2,1]   # model dimensions for Python are (z,y,x) and (z,x)

    # for 3D modeling, limit model to area with sources/receivers
    if options.limit_m == true
        model = deepcopy(model_full)
        if op=='J' && mode==1
            model,dm = limit_model_to_receiver_area(srcGeometry,recGeometry,model,options.buffer_size;pert=dm)
        else
            model = limit_model_to_receiver_area(srcGeometry,recGeometry,model,options.buffer_size)
        end
    else
        model = model_full
    end

    # Set up Python model structure
    if op=='J' && mode == 1
        modelPy = pm.Model(origin=(0.,0.,0.), spacing=model.d, shape=model.n, vp=process_physical_parameter(sqrt.(1f0./model.m), dims), nbpml=model.nb,
            rho=process_physical_parameter(model.rho, dims), dm=process_physical_parameter(reshape(dm,model.n), dims), space_order=options.space_order)
    else
        modelPy = pm.Model(origin=(0.,0.,0.), spacing=model.d, shape=model.n, vp=process_physical_parameter(sqrt.(1f0./model.m), dims), nbpml=model.nb,
            rho=process_physical_parameter(model.rho, dims), space_order=options.space_order)
    end
    
    # Remove receivers outside the modeling domain (otherwise leads to segmentation faults)
    if mode==1 && recGeometry != nothing
        recGeometry = remove_out_of_bounds_receivers(recGeometry, model)
    elseif mode==-1 && recGeometry != nothing
        recGeometry, dIn = remove_out_of_bounds_receivers(recGeometry, recData, model)
    end

    argout = devito_interface(modelPy, srcGeometry, srcData, recGeometry, recData, dm, options)
    return argout
end

# Function instance without options
time_modeling(model::Model, srcGeometry, srcData, recGeometry, recData, perturbation, srcnum::Int64, op::Char, mode::Int64) =
    time_modeling(model, srcGeometry, srcData, recGeometry, recData, perturbation, srcnum, op, mode, Options())

######################################################################################################################################################

function get_origin(modelPy:: PyCall.PyObject) 
    orig = modelPy[:origin]
    if length(orig) == 2
        o = (orig[1][:data], orig[2][:data])
    else
        o = (orig[1][:data], orig[2][:data], orig[3][:data])
    end
end

# d_obs = Pr*F*Ps'*q
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry, recData::Void, dm::Void, options::Options)
    
    # Interpolate input data to computational grid
    dtComp = modelPy[:critical_dt]
    qIn = time_resample(srcData[1],srcGeometry,dtComp)[1]
    ntComp = size(qIn,1)
    ntRec = Int(trunc(recGeometry.t[1]/dtComp + 1))

    # Set up coordinates with devito dimensions
    origin = get_origin(modelPy)
    src_coords = setup_grid(srcGeometry, modelPy[:shape], origin)
    rec_coords = setup_grid(recGeometry, modelPy[:shape], origin)

    # Devito call
    dOut = pycall(ac.forward_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'), 
                  space_order=options.space_order, nb=modelPy[:nbpml])[1]
    ntRec > ntComp && (dOut = [dOut zeros(size(dOut,1), ntRec - ntComp)])
    dOut = time_resample(dOut,dtComp,recGeometry)

    # Output shot record as judiVector
    if options.save_data_to_disk
        container = write_shot_record(srcGeometry,srcData,recGeometry,dOut,options)
        return judiVector(container)
    else
        return judiVector(recGeometry,dOut)
    end
end

# q_ad = Ps*F'*Pr'*d_obs
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Geometry, srcData::Void, recGeometry::Geometry, recData::Array, dm::Void, options::Options)

    # Interpolate input data to computational grid
    dtComp = modelPy[:critical_dt]
    if typeof(recData[1]) == SeisIO.SeisCon
        recDataCell = Array{Any}(1); recDataCell[1] = convert(Array{Float32,2},recData[1][1].data); recData = recDataCell
    elseif typeof(recData[1]) == String
        recData = load(recData[1])["d"].data
    end
    dIn = time_resample(recData[1],recGeometry,dtComp)[1]
    ntComp = size(dIn,1)
    ntSrc = Int(trunc(srcGeometry.t[1]/dtComp + 1))

    # Set up coordinates with devito dimensions
    origin = get_origin(modelPy)
    src_coords = setup_grid(srcGeometry, modelPy[:shape], origin)
    rec_coords = setup_grid(recGeometry, modelPy[:shape], origin)

    # Devito call
    qOut = pycall(ac.adjoint_modeling, Array{Float32,2}, modelPy, PyReverseDims(src_coords'), PyReverseDims(rec_coords'), PyReverseDims(dIn'),
                  space_order=options.space_order, nb=modelPy[:nbpml])
    ntSrc > ntComp && (qOut = [qOut zeros(size(qOut), ntSrc - ntComp)])
    qOut = time_resample(qOut,dtComp,srcGeometry)

    # Output adjoint data as judiVector
    return judiVector(srcGeometry,qOut)
end

# u = F*Ps'*q
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Geometry, srcData::Array, recGeometry::Void, recData::Void, dm::Void, options::Options)
    
    # Interpolate input data to computational grid
    dtComp = modelPy[:critical_dt]
    qIn = time_resample(srcData[1],srcGeometry,dtComp)[1]
    ntComp = size(qIn,1)

    # Set up coordinates with devito dimensions
    origin = get_origin(modelPy)
    src_coords = setup_grid(srcGeometry, modelPy[:shape], origin)

    # Devito call
    u = pycall(ac.forward_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), nothing, 
               space_order=options.space_order, nb=modelPy[:nbpml])

    # Output forward wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy[:shape]), 1, ntComp), u)
end

# v = F'*Pr'*d_obs
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Void, srcData::Void, recGeometry::Geometry, recData::Array, dm::Void, options::Options)

    # Interpolate input data to computational grid
    dtComp = modelPy[:critical_dt]
    if typeof(recData[1]) == SeisIO.SeisCon
        recDataCell = Array{Any}(1); recDataCell[1] = convert(Array{Float32,2},recData[1][1].data); recData = recDataCell
    elseif typeof(recData[1]) == String
        recData = load(recData[1])["d"].data
    end
    dIn = time_resample(recData[1],recGeometry,dtComp)[1]
    ntComp = size(dIn,1)

    # Set up coordinates with devito dimensions
    origin = get_origin(modelPy)
    rec_coords = setup_grid(recGeometry, modelPy[:shape], origin)

    # Devito call
    v = pycall(ac.adjoint_modeling, PyObject, modelPy, nothing, PyReverseDims(rec_coords'), PyReverseDims(dIn'),
               space_order=options.space_order, nb=modelPy[:nbpml])

    # Output adjoint wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy[:shape]), 1, ntComp), v)
end

# d_obs = Pr*F*u
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Void, srcData::Array, recGeometry::Geometry, recData::Void, dm::Void, options::Options)
  
    # Interpolate input data to computational grid
    dtComp = modelPy[:critical_dt]
    ntComp = srcData[1][:shape][1]
    ntRec = Int(trunc(recGeometry.t[1]/dtComp + 1))

    # Set up coordinates with devito dimensions
    origin = get_origin(modelPy)
    rec_coords = setup_grid(recGeometry, modelPy[:shape], origin)

    # Devito call
    dOut = pycall(ac.forward_modeling, PyObject, modelPy, nothing, srcData[1], PyReverseDims(rec_coords'), 
                  space_order=options.space_order, nb=modelPy[:nbpml])[1]
    ntRec > ntComp && (dOut = [dOut zeros(size(dOut,1), ntRec - ntComp)])
    dOut = time_resample(dOut,dtComp,recGeometry)

    # Output shot record as judiVector
    if options.save_data_to_disk
        throw("Writing shot record to SEG-Y file not supported for modeling with wavefield as right-hand-side.")
    else
        return judiVector(recGeometry,dOut)
    end
end

# q_ad = Ps*F'*v
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Geometry, srcData::Void, recGeometry::Void, recData::Array, dm::Void, options::Options)

    # Interpolate input data to computational grid
    dtComp = modelPy[:critical_dt]
    ntComp = recData[1][:shape][1]
    ntSrc = Int(trunc(srcGeometry.t[1]/dtComp + 1))

    # Set up coordinates with devito dimensions
    origin = get_origin(modelPy)
    src_coords = setup_grid(srcGeometry, modelPy[:shape], origin)

    # Devito call
    qOut = pycall(ac.adjoint_modeling, Array{Float32,2}, modelPy, PyReverseDims(src_coords'), nothing, recData[1],
                  space_order=options.space_order, nb=modelPy[:nbpml])
    ntSrc > ntComp && (qOut = [qOut zeros(size(qOut), ntSrc - ntComp)])
    qOut = time_resample(qOut,dtComp,srcGeometry)

    # Output adjoint data as judiVector
    return judiVector(srcGeometry,qOut)
end

# u_out = F*u_in
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Void, srcData::Array, recGeometry::Void, recData::Void, dm::Void, options::Options)
  
    # Interpolate input data to computational grid
    dtComp = modelPy[:critical_dt]
    ntComp = srcData[1][:shape][1]

    # Devito call
    u = pycall(ac.forward_modeling, PyObject, modelPy, nothing, srcData[1], nothing, 
                  space_order=options.space_order, nb=modelPy[:nbpml])
    
    # Output forward wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy[:shape]), 1, ntComp), u)
end

# v_out = F'*v_in
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Void, srcData::Void, recGeometry::Void, recData::Array, dm::Void, options::Options)

    # Interpolate input data to computational grid
    dtComp = modelPy[:critical_dt]
    ntComp = recData[1][:shape][1]

    # Devito call
    v = pycall(ac.adjoint_modeling, PyObject, modelPy, nothing, nothing, recData[1],
                  space_order=options.space_order, nb=modelPy[:nbpml])

    # Output adjoint wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy[:shape]), 1, ntComp), v)
end

# d_lin = J*dm
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry, recData::Void, dm::Array, options::Options)

    # Interpolate input data to computational grid
    dtComp = modelPy[:critical_dt]
    tmaxRec = recGeometry.t[1]
    qIn = time_resample(srcData[1],srcGeometry,dtComp)[1]
    ntComp = size(qIn,1)
    ntRec = Int(trunc(tmaxRec/dtComp + 1))

    # Set up coordinates with devito dimensions
    origin = get_origin(modelPy)
    src_coords = setup_grid(srcGeometry, modelPy[:shape], origin)
    rec_coords = setup_grid(recGeometry, modelPy[:shape], origin)
    
    # Devito call
    dOut = pycall(ac.forward_born, Array{Float32,2}, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'),
                  space_order=options.space_order, nb=modelPy[:nbpml], isic=options.isic)
    ntRec > ntComp && (dOut = [dOut zeros(size(dOut,1), ntRec - ntComp)])
    dOut = time_resample(dOut,dtComp,recGeometry)

    # Output linearized shot records as judiVector
    if options.save_data_to_disk
        container = write_shot_record(srcGeometry,srcData,recGeometry,dOut,options)
        return judiVector(container)
    else
        return judiVector(recGeometry,dOut)
    end
end

# dm = J'*d_lin
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry, recData::Array, dm::Void, options::Options)

    # Interpolate input data to computational grid
    dtComp = modelPy[:critical_dt]
    if typeof(recData[1]) == SeisIO.SeisCon
        recDataCell = Array{Any}(1); recDataCell[1] = convert(Array{Float32,2},recData[1][1].data); recData = recDataCell
    elseif typeof(recData[1]) == String
        recData = load(recData[1])["d"].data
    end
    qIn = time_resample(srcData[1],srcGeometry,dtComp)[1]
    dIn = time_resample(recData[1],recGeometry,dtComp)[1]
    
    # Set up coordinates with devito dimensions
    origin = get_origin(modelPy)
    src_coords = setup_grid(srcGeometry, modelPy[:shape], origin)
    rec_coords = setup_grid(recGeometry, modelPy[:shape], origin)

    if options.optimal_checkpointing == true
        op_F = pycall(ac.forward_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'),
                      op_return=true, space_order=options.space_order, nb=modelPy[:nbpml])
        grad = pycall(ac.adjoint_born, Array{Float32, length(modelPy[:shape])}, modelPy, PyReverseDims(rec_coords'), PyReverseDims(dIn'), op_forward=op_F,
                      space_order=options.space_order, nb=modelPy[:nbpml], is_residual=true, isic=options.isic)
    elseif ~isempty(options.frequencies)    # gradient in frequency domain
        typeof(options.frequencies) == Array{Any,1} && (options.frequencies = options.frequencies[srcnum])
        d_pred, uf_real, uf_imag = pycall(ac.forward_freq_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'),
                                          options.frequencies, space_order=options.space_order, nb=modelPy[:nbpml])
        grad = pycall(ac.adjoint_freq_born, Array{Float32, length(modelPy[:shape])}, modelPy, PyReverseDims(rec_coords'), PyReverseDims(dIn'),
                      options.frequencies, uf_real, uf_imag, space_order=options.space_order, nb=modelPy[:nbpml], isic=options.isic)
    else
        u0 = pycall(ac.forward_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'),
                    space_order=options.space_order, nb=modelPy[:nbpml], save=true)[2]
        grad = pycall(ac.adjoint_born, Array{Float32, length(modelPy[:shape])}, modelPy, PyReverseDims(rec_coords'), PyReverseDims(dIn'), u=u0,
                      space_order=options.space_order, nb=modelPy[:nbpml], isic=options.isic)
    end
    
    # Remove PML and return gradient as Array
    grad = remove_padding(grad,modelPy[:nbpml], true_adjoint=options.sum_padding)
    if options.limit_m == true
        grad = extend_gradient(model_full,model,grad)
    end
    return vec(grad)
end


  
