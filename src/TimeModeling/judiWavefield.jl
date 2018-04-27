############################################################
# judiWavefield ##############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: June 2017

export judiWavefield, judiWavefieldException, judiDFTwavefield, muteWavefield

############################################################

type judiWavefield{vDT<:Number} <: joAbstractLinearOperator{vDT,vDT}
	name::String
	m::Integer
	n::Integer
	info::Info
	data
end

type judiWavefieldException <: Exception
	msg :: String
end


############################################################

## outer constructors

function judiWavefield(info,data::Union{Array, PyCall.PyObject}; vDT::DataType=Float32)
	
	# length of vector
	m = info.n * sum(info.nt)
	n = 1
	dataCell = Array{Any}(info.nsrc)
	for j=1:info.nsrc
		dataCell[j] = data
	end
	return judiWavefield{vDT}("judiWavefield",m,n,info,dataCell)
end

function judiWavefield(info,data::Array{Any,1};vDT::DataType=Float32)
	# length of vector
	m = info.n * sum(info.nt)
	n = 1
	return judiWavefield{vDT}("judiWavefield",m,n,info,data)
end


####################################################################
## overloaded Base functions

# conj(jo)
conj{vDT}(A::judiWavefield{vDT}) =
	judiWavefield{vDT}("conj("*A.name*")",A.m,A.n,A.info,A.data)

# transpose(jo)
transpose{vDT}(A::judiWavefield{vDT}) =
	judiWavefield{vDT}(""*A.name*".'",A.n,A.m,A.info,A.data)
   
# ctranspose(jo)
ctranspose{vDT}(A::judiWavefield{vDT}) =
	judiWavefield{vDT}(""*A.name*"'",A.n,A.m,A.info,A.data)

####################################################################

function vcat{avDT,bvDT}(a::judiWavefield{avDT},b::judiWavefield{bvDT})
	m = a.m + b.m
	n = 1
	nsrc = a.info.nsrc + b.info.nsrc
	data = Array{Any}(nsrc)
	nt = Array{Any}(nsrc)
	for j=1:a.info.nsrc
		data[j] = a.data[j]
		nt[j] = a.info.nt[j]
	end
	for j=a.info.nsrc+1:nsrc
		data[j] = b.data[j-a.info.nsrc]
		nt[j] = b.info.nt[j-a.info.nsrc]
	end
	info = Info(a.info.n,nsrc,nt)
	return judiWavefield(info,data)
end

# add and subtract, mulitply and divide, norms, dot ...


# DFT operator for wavefields, acts along time dimension
function fft_wavefield(x_in,mode)
	nsrc = x_in.info.nsrc
	if mode==1
		x = judiWavefield(x_in.info,deepcopy(x_in.data); vDT=Complex{Float32})
		for i=1:nsrc
			x.data[i] = convert(Array{Complex{Float32}},x.data[i])
			nx = size(x.data[i],2)
			nz = size(x.data[i],3)
			for j=1:nx
				for k=1:nz
					x.data[i][:,j,k] = fft(x.data[i][:,j,k])
				end
			end
		end
	elseif mode==-1
		x = judiWavefield(x_in.info,deepcopy(x_in.data); vDT=Float32)
		for i=1:nsrc
			nx = size(x.data[i],2)
			nz = size(x.data[i],3)
			for j=1:nx
				for k=1:nz
					x.data[i][:,j,k] = real(ifft(x.data[i][:,j,k]))
				end
			end
			x.data[i] = convert(Array{Float32},real(x.data[i]))
		end
	end
	return x
end

#function judiDFTwavefield(n; DDT::DataType=Float32,RDT::DataType=(DDT<:Real?Complex{DDT}:DDT))
## JOLI wrapper for the DFT of wavefield vector along time
#
#	F = joLinearFunctionFwdT(n,n,
#							 v -> fft_wavefield(v,1),
# 							 w -> fft_wavefield(w,-1),
#							 DDT,RDT,name="DFT of wavefields along time")
#	return F
#end

# Overload multiplication for judiDFT*judiWavefield
#function *{ADDT,ARDT,vDT}(A::joLinearFunction{ADDT,ARDT},v::judiWavefield{vDT})
#	A.n == size(v,1) || throw(judiWavefieldException("shape mismatch"))
#	jo_check_type_match(ADDT,vDT,join(["DDT for *(judiDFT,judiWavefield):",A.name,typeof(A),vDT]," / "))
#	V = A.fop(v)
#	jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiDFT,judiWavefield):",A.name,typeof(A),eltype(V)]," / "))
#	return V
#end

# Sampling mask to extract wavefields from full vector
subsample(u::judiWavefield,srcnum) = judiWavefield(u.info,u.data[srcnum];vDT=eltype(u))

function muteWavefield(u_in::judiWavefield,ts_keep)
	u = deepcopy(u_in)
	for j=1:u.info.nsrc
		idx = ones(size(u.data[j],1));
		idx[ts_keep] = 0
		zero_idx = find(idx)
		u.data[j][zero_idx,:,:] *= 0.f0
	end
	return u
end


