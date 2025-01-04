module MRIGIRF

using Interpolations
using FFTW
import DSP

# rf step = ratio of dwell times of rf to adc
# tr in time units of adc
# rf_start is where rf != 0 and adc != 0
# adc_end and rf_end are the last index of rf and adc in adc time resolution
# TODO: this is becoming more and more specific to Siemens' Poet simualtion (DSV files), it should be outsourced, probably into the SiemensDSV.jl package
function splitTR(rf::AbstractVector{<: Real}, adc::AbstractVector{<: Real}, δt_rf::Real, δt_adc::Real, tr::Integer)

	rf_step = floor(Int, δt_rf / δt_adc)
	@assert rf_step * δt_adc == δt_rf

	rf_start = Vector{Int}(undef, 0)
	rf_end = Vector{Int}(undef, 0)
	adc_start = Vector{Int}(undef, 0)
	adc_end = Vector{Int}(undef, 0)
	num_tr = length(adc) ÷ tr
	sizehint!.((rf_start, rf_end, adc_start, adc_end), num_tr)

	t = 1
	τ = 1
	hadpulse = false
	hadadc = false

	while t <= length(adc)

		if τ <= length(rf) && rf[τ] != 0
			if hadpulse
				pop!.((rf_start, rf_end)) # remove last pulse, because no ADC happened
			end
			push!(rf_start, t)
			hadpulse = true
			while rf[τ] != 0 || adc[t] != 0
				τ += 1
				t += rf_step
				τ > length(rf) && break
			end
			# Found that this cuts off last element in rf_end, unexpectedly: τ <= length(rf) && <line below>
			push!(rf_end, t-1)
			continue # need this in case t > length(adc) or length(rf)
		end

		if hadpulse && adc[t] == 1
			t_start = t

			noadc = false
			while t ≤ length(adc) && adc[t] == 1
				if τ <= length(rf) && rf[τ] != 0 # All this because DICO measurement is simulated as ADC
					noadc = true
					break
				end
				τ = t ÷ rf_step + 1
				t += 1
			end
			noadc && continue

			push!(adc_start, t_start)
			push!(adc_end, t-1)

			hadpulse = false
		end

		τ = t ÷ rf_step + 1
		t += 1
	end
	return rf_start, rf_end, adc_start, adc_end
end


# k returned in units of [g] * [δt_g]
# Integrates from just after the rf pulse
function compute_trajectory(
	g::AbstractVector{<: Real},
	rf_end::AbstractVector{<: Integer},
	adc_start::AbstractVector{<: Integer},
	adc_end::AbstractVector{<: Integer}, # should remove this, as it can be computed from adc_start+adc_length-1. This way it forces the user to use readouts of same length which makes sense (for a given purpose)
	# TODO: what about those two indices missing in adc_length, e.g. length=1000, but end-start+1 == 998?
	adc_length::Integer, # in units of δt_adc
	num_samples::Integer,
	δt_adc::Real,
	δt_g::Real
)
	num_tr = length(rf_end)
	@assert num_tr == length(adc_start) == length(adc_end)

	nt = floor(Int, δt_g / δt_adc)
	@assert nt * δt_adc == δt_g
	num_g = adc_length ÷ nt

	# Compute trajectory on gradient raster time
	k = Matrix{Float64}(undef, num_g, num_tr)
	@views Threads.@threads for i = 1:num_tr
		# rf_end[i] is part of rf-pulse, hence need to +1 to take first index after pulse
		# Exclude last index because otherwise integration goes one too far, i.e. gradient value in last dt doesn't matter
		idx = ((rf_end[i]+1-1) ÷ nt + 1) : ((adc_end[i]-1) ÷ nt)
		κ = Vector{Float64}(undef, length(idx) + 1) # TODO this could be optimised
		κ[1] = 0
		cumsum!(κ[2:end], 0.5 .* (g[idx] .+ g[idx .+ 1]))
		k[:, i] .= κ[end-num_g+1 : end]
	end

	# Interpolate to target dwell time
	k_itp = Matrix{Float64}(undef, num_samples, num_tr)
	δ = num_g / num_samples
	@views Threads.@threads for i = 1:num_tr
		itp = extrapolate(interpolate(k[:, i], BSpline(Linear())), Line())
		k_itp[:, i] = itp(((0:num_samples-1) .+ 0.5) .* δ .+ 1) # Experimental shift by half an index (0.5Δk shift)
	end
	k_itp .*= δt_g

	return k_itp
end

# Integrates the whole curve, useful?
#function compute_trajectory(
#	g::AbstractVector{<: Real},
#	adc_start::AbstractVector{<: Integer},
#	adc_length::Integer, # in units of δt_adc
#	num_samples::Integer,
#	δt_adc::Real,
#	δt_g::Real
#)
#	num_tr = length(adc_start)
#
#	nt = floor(Int, δt_g ÷ δt_adc)
#	@assert nt * δt_adc == δt_g
#	num_g = adc_length ÷ nt
#
#	# Compute trajectory through integration
#	k = Vector{Float64}(undef, length(g))
#	k[1] = 0
#	@views cumsum!(k[2:end], 0.5 .* (g[1:end-1] .+ g[2:end]))
#
#	# Interpolate to target dwell time
#	k_itp = Matrix{Float64}(undef, num_samples, num_tr)
#	δ = num_g / num_samples
#	@views Threads.@threads for i = 1:num_tr
#		j = (adc_start[i]-1) ÷ nt
#		itp = extrapolate(interpolate(k[j+1:j+num_g+1], BSpline(Linear())), Line())
#		k_itp[:, i] = itp((0:num_samples-1) .* δ .+ 1 .+ 0.0δ) # TODO: Experimental shift by half an index (0.5Δk shift)
#	end
#	k_itp .*= δt_g
#
#	return k_itp
#end


#= Sidenote: Can the GIRF be truncated if the gradient waveform is shorter than the GIRF?
Yes:
	a ⋅ (g ⨂ χ) = a ⋅ g ⨂ (a ⋅ χ)

But: it would mean that the response outside that interval is lost, yet it is required for
calculating the realised trajectory
=#

# Requires χ in the time-domain, with the first index corresponding to time = 0
# Returns the non-fftshifted spectrum of the GIRF using rfft, if necessary it has been padded in time-domain to match gradient
# TODO: could prep the rfft for the apply, but shouldn't give much performance benefit since usually only applied to three curves
function prepare_for_fourier(χ::AbstractArray{<: Number, 3}, dt_χ::Real, T_g::Real)
	# Match frequency resolution by zero-padding time domain
	T_χ = size(χ, 1) * dt_χ
	ΔT = T_g - T_χ
	if ΔT > 0
		n_pad = round(Int, ΔT / dt_χ)
		@assert isapprox(n_pad * dt_χ, ΔT) "Could not assimilate frequency resolution, need linear interpolation (not implemented)"
		χ_pad = Array{eltype(χ), 3}(undef, size(χ, 1) + n_pad, size(χ, 2), size(χ, 3))
		m = size(χ, 1) ÷ 2 + 1
		χ_pad[1:m,         :, :]  = χ[1:m,     :, :]
		χ_pad[m+1:end-m+1, :, :] .= 0
		χ_pad[end-m+2:end, :, :]  = χ[m+1:end, :, :]
	elseif ΔT < 0
		error("Gradient waveform is shorter than GIRF, which produces truncation errors,
		pad sufficiently before and after the gradients are non-zero")
	end

	return rfft(χ_pad, 1)
end

# MAKE SURE TO ZERO PAD GRADIENTS TO AVOID WRAP AROUND ARTEFACTS (Periodicity due to FT)
#= Explanation:
	There must be enough wiggle-room in the gradient waveform w.r.t. where gradients are non-zero and
	also considering the delay of the GIRF as well as its width.
	If that's not the case, errors will be introduced because the assumption is that both curves are periodic.
=#
# Assumes that g and χ correspond to the same time-window (and periodic outside) and Fχ the non-fftshifted rfft of χ
# g[time, curve]
# TODO: need unit test for this, maybe comparing to DSP.jl's conv()
function apply(Fχ::AbstractArray{<: Number, 3}, g::AbstractMatrix{<: Real})
	@assert size(g, 2) == size(Fχ, 2)

	Fg = rfft(g, 1)
	n = min(size(Fg, 1), size(Fχ, 1))
	Fgr = Array{Complex{eltype(g)}}(undef, size(Fg, 1), size(g, 2), size(Fχ, 3))
	@views @. Fgr[1:n, :, :]  = Fg[1:n, :] * Fχ[1:n, :, :]
	Fgr[n+1:end, :, :]       .= 0 # Sinc-interpolation to get dt of g, should be good
	gr = irfft(Fgr, size(g, 1), 1)

	return gr
end

# TODO: there needs to be a version of apply where a standard convolution is evaluated, since girf length is small compared to whole array, fft doesn't pay off
#function apply(χ::AbstractArray{<: Number, 3}, g::AbstractMatrix{<: Real}, rf_end::AbstractVector{<: Integer}, adc_end::AbstractVector{<: Integer})
#
#	gr = Array{eltype(g)}(undef, size(g, 1), size(g, 2), size(χ, 3))
#	for i in eachindex(rf_end)
#		for j in axes(g, 2)
#			@views gr[:, j, i] = DSP.conv(χ[:, ], g[..., j])
#		end
#	end
#
#	@views @. Fgr[1:n, :, :]  = Fg[1:n, :] * Fχ[1:n, :, :]
#	Fgr[n+1:end, :, :]       .= 0 # Sinc-interpolation to get dt of g, should be good
#	gr = irfft(Fgr, size(g, 1), 1)
#
#	return gr
#end

# Prepare for time-domain convolution, only need to adjust temporal resolution
# Tested for position of peak
function prepare_for_time(χ::AbstractArray{<: Number, 3}, dt_χ::Real, dt_g::Real)
	# Match frequency resolution by zero-padding time domain
	X = rfft(χ, 1)
	ratio = dt_χ / dt_g
	n_pad = round(Int, size(χ, 1) * ratio)
	m = size(χ, 1) ÷ 2 + 1 # TODO: absolutely sure this is correct? Seems to work for odd number of elements
	if ratio > 1
		@show "padding"
		# fmax = n_freqdomain * resolution_freq_domain
		@assert isapprox(2π / dt_g, n_pad * 2π / (size(χ, 1) * dt_χ)) "Could not assimilate sampling times"
		X_pad = Array{eltype(X), 3}(undef, (n_pad ÷ 2)+1, size(X, 2), size(X, 3))
		X_pad[1:m, :, :] = X
		X_pad[m+1:end, :, :] .= 0
		X = X_pad
	elseif ratio < 1
		error("Gradient waveform has coarser resolution than GIRF, not implemented.")
	end

	χ = irfft(X, n_pad, 1)
	χ = cat(χ[m+1:end, :, :], χ[1:m, :, :]; dims=1)
	return χ
end

# Tested for shift in time with two deltas
function apply_in_time(χ::AbstractArray{<: Number, 3}, g::AbstractMatrix{<: Real})
	@assert size(g, 2) == size(χ, 2) # Spatial dimensions, not time

	gc = similar(g)
	Tχ = size(χ, 1)
	Tg = size(g, 1)
	t0 = Tχ ÷ 2 # this is t = 0
	@views for j in axes(χ, 3), i in axes(χ, 2)
		Threads.@threads for t = 1:Tg
			v = 0.0
			for t′ = 0:Tχ-1
				t″ = t + t′ - t0
				t″ < 1 && continue
				t″ > Tg && break
				v += χ[Tχ-t′, i, j] * g[t″, i]
				# Debug GIRF peak Tχ-t′ == t0 && @show χ[Tχ-t′, 1, 1]
			end
			gc[t, i] = v
		end
	end
	return gc
end

end

