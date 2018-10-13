# Parallel instance of fwi_objective function
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

"""
    fwi_objective(model, source, dobs; options=Options())

Evaluate the full-waveform-inversion (reduced state) objective function. Returns a tuple with function value and vectorized \\
gradient. `model` is a `Model` structure with the current velocity model and `source` and `dobs` are the wavelets and \\
observed data of type `judiVector`.

Example
=======

    function_value, gradient = fwi_objective(model, source, dobs)

"""
function fwi_objective(model::Model, source::judiVector, dObs::judiVector; options=Options())
# fwi_objective function for multiple sources. The function distributes the sources and the input data amongst the available workers.

    # Process shots from source channel asynchronously
    fwi_objective = retry(TimeModeling.fwi_objective)
    results = Array{Any}(dObs.nsrc)
    @sync begin
        for j=1:dObs.nsrc
            results[j] = @spawn fwi_objective(model, source[j], dObs[j], j; options=options)   
        end
    end

    # Collect and reduce gradients
    gradient = zeros(Float32,prod(model.n)+1)
    for j=1:dObs.nsrc
        gradient += fetch(results[j]); results[j] = []
    end

    # first value corresponds to function value, the rest to the gradient
    return gradient[1], gradient[2:end]
end

function fwi_objective(model::Model, source::judiVector, F::judiPDEfull; options=Options())
# fwi_objective function for multiple sources. The function distributes the sources and the input data amongst the available workers.

    # Process shots from source channel asynchronously
    fwi_objective = retry(TimeModeling.fwi_objective)
    results = Array{Any}(F.info.nsrc)
    @sync begin
        for j=1:F.info.nsrc
            results[j] = @spawn fwi_objective(model, source[j], F[j], j; options=options)   
        end
    end

    # Collect and reduce gradients
    gradient = zeros(Float32,prod(model.n)+1)
    for j=1:F.info.nsrc
        gradient += fetch(results[j]); results[j] = []
    end

    # first value corresponds to function value, the rest to the gradient
    return gradient[1], gradient[2:end]
end


