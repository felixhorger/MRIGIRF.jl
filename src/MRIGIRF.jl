module MRIGIRF

using Interpolations
using FFTW

# rf step = ratio of dwell times of rf to adc
# tr in time units of adc
function splitTR(rf::AbstractVector{<: Real}, adc::AbstractVector{<: Real}, δt_rf::Real, δt_adc::Real, tr::Integer)

	rf_step = floor(Int, δt_rf / δt_adc)
	@assert rf_step * δt_adc == δt_rf

	tr_start = Vector{Int}(undef, 0)
	adc_start = Vector{Int}(undef, 0)
	adc_end = Vector{Int}(undef, 0)
	num_tr = length(adc) ÷ tr
	sizehint!.((tr_start, adc_start, adc_end), num_tr)

	t = 1
	τ = 1
	hadpulse = false
	hadadc = false

	while t <= length(adc)

		if τ <= length(rf) && rf[τ] > 0
			hadpulse && pop!(tr_start) # remove last pulse, because no ADC happened
			hadpulse = true
			while rf[τ] != 0 || adc[t] != 0
				τ += 1
				t += rf_step
				τ > length(rf) && break
			end
			# Found that this cuts off last element in tr_start, unexpectedly: τ <= length(rf) &&
			push!(tr_start, t)
			continue # need this in case t > length(adc) or length(rf)
		end

		if hadpulse && adc[t] == 1
			push!(adc_start, t)

			while t ≤ length(adc) && adc[t] == 1
				t += 1
			end

			t <= length(adc) && push!(adc_end, t-1)

			hadpulse = false
		end

		τ = t ÷ rf_step + 1
		t += 1
	end
	return tr_start, adc_start, adc_end
end


function compute_trajectory(
	g::AbstractVector{<: Real},
	tr_start::AbstractVector{<: Integer},
	adc_start::AbstractVector{<: Integer},
	adc_end::AbstractVector{<: Integer}, # should remove this, as it can be computed from adc_start+adc_length-1. This way it forces the user to use readouts of same length which makes sense (for a given purpose)
	adc_length::Integer,
	num_samples::Integer,
	δt_adc::Real,
	δt_g::Real
)
	num_tr = length(tr_start)
	@assert num_tr == length(adc_start) == length(adc_end)

	nt = floor(Int, δt_g ÷ δt_adc)
	@assert nt * δt_adc == δt_g
	num_g = adc_length ÷ nt

	# Compute trajectory
	k = Matrix{Float64}(undef, num_g, num_tr)
	@views Threads.@threads :static for i = 1:num_tr
		idx = tr_start[i]:nt:adc_end[i]
		idx = (idx .- 1) .÷ nt .+ 1
		κ = cumsum(0.5 .* (g[idx .- 1] .+ g[idx]))
		k[:, i] .= κ[end-num_g+1 : end]
	end

	# Interpolate to target dwell time
	k_itp = Matrix{Float64}(undef, num_samples, num_tr)
	@views Threads.@threads for i = 1:num_tr
		itp = extrapolate(interpolate(k[:, i], BSpline(Linear())), Line())
		k_itp[:, i] = itp((0:num_samples-1) .* (num_g / num_samples) .+ 1)
	end
	k_itp .*= δt_g

	return k_itp
end


#= Sidenote: Can the GIRF be truncated if the gradient waveform is shorter than the GIRF?
Yes:
	a ⋅ (g ⨂ χ) = a ⋅ g ⨂ (a ⋅ χ)

But: it would mean that the response outside that interval is lost, yet it is required for
calculating the realised trajectory
=#

# Requires χ in the time-domain, with the first index corresponding to time = 0
# Returns the non-fftshifted spectrum of the GIRF using rfft, if necessary it has been padded in time-domain to match gradient
function prepare(χ::AbstractArray{<: Number, 3}, dt_χ::Real, T_g::Real)
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

#= Note:
	There must be enough wiggle-room in the gradient waveform w.r.t. where gradients are non-zero and
	also considering the delay of the GIRF as well as its width.
	If that's not the case, errors will be introduced because the assumption is that both curves are periodic.
=#
# Assumes that g and χ correspond to the same time-window (and periodic outside) and Fχ the non-fftshifted rfft of χ
function apply(g::AbstractMatrix{<: Real}, Fχ::AbstractArray{<: Number, 3})
	@assert size(g, 2) == size(Fχ, 2)


	Fg = rfft(g, 1)
	n = min(size(Fg, 1), size(Fχ, 1))
	Fgr = Array{Complex{eltype(g)}}(undef, size(Fg, 1), size(g, 2), size(Fχ, 3))
	@views @. Fgr[1:n, :, :]  = Fg[1:n, :] * Fχ[1:n, :, :]
	Fgr[n+1:end, :, :]       .= 0 # Sinc-interpolation to get dt of g, should be good
	gr = irfft(Fgr, size(g, 1), 1)

	return gr
end


end

