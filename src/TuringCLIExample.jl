# see tutorial https://turinglang.github.io/TuringCallbacks.jl/dev/

module TuringCLIExample

using TuringCallbacks: join
using Turing, TuringCallbacks, TensorBoardLogger, ArgParse, StatsPlots, CSV, DataFrames
import ArgParse.parse_item

function ArgParse.parse_item(::Type{Vector{Float64}}, x::AbstractString)
    return parse.(Float64, split(strip(x, [' ','\"',']','[']), ','))
end

function simulate_and_estimate(d)
    s_prior_alpha, s_prior_theta = d.prior
    @model function demo(x; s_prior_alpha, s_prior_theta)
        s ~ InverseGamma(s_prior_alpha, s_prior_theta)
        m ~ Normal(0, √s)
        for i in eachindex(x)
            x[i] ~ Normal(m, √s)
        end
    end

    xs = randn(100) .+ 1
    model = demo(xs; s_prior_alpha, s_prior_theta)

    # Sampling
    println("Generating $(d.num_samples) samples")
    callback = TensorBoardCallback(joinpath(pkgdir(TuringCLIExample),"tensorboard_logs/run"))
    alg = NUTS(d.num_adapts, d.target_acceptance_rate)
    chain = sample(model, alg, d.num_samples; callback)

    println("Generating trace plot")
    trace_plot = plot(chain, seriestype=:traceplot)
    savefig(trace_plot,  joinpath(callback.logger.logdir, "traceplots.png"))


    println("Summarizing the chain")
    sum_stats = describe(chain)
    param_names = sum_stats[1][:,1]
    param_mean = sum_stats[1][:,2]
    param_sd = sum_stats[1][:,3]
    param_ess = sum_stats[1][:,6]
    param_rhat = sum_stats[1][:,7]
    param_ess_per_sec = sum_stats[1][:, 8]

    CSV.write(
        joinpath(callback.logger.logdir, "summary.csv"),
        DataFrame(
            parameter=param_names, 
            mean=param_mean, 
            sd=param_sd, 
            ess=param_ess, 
            rhat=param_rhat, 
            ess_per_sec= param_ess_per_sec
        )
    )

    # Log the ESS/sec and rhat.  Nice to show as summary results from tensorboard
    for (i, name) = enumerate(param_names)
        TensorBoardLogger.log_value(
            callback.logger,
            "$(name)_ess_per_sec",
            param_ess_per_sec[i],
        )
        TensorBoardLogger.log_value(
            callback.logger,
            "$(name)_rhat",
            param_rhat[i],
        )
    end
end

# Entry for script
function main(args = ARGS)
    d = parse_commandline(args)
    simulate_and_estimate((;d...)) # to named tuple
end

function parse_commandline(args)

    s = ArgParseSettings(fromfile_prefix_chars=['@'])

    # See src/defaults.txt
    @add_arg_table! s begin
        "--num_samples"
        help = "samples to draw in chain"
        arg_type = Int64
        "--num_adapts"
        help = "number of adaptations for NUTS"
        arg_type = Int64
        "--target_acceptance_rate"
        help = "Target acceptance rate for dual averaging."
        arg_type = Float64
        "--prior"
        help = "prior parameters in InverseGamma prior for s"
        arg_type = Vector{Float64}
    end

    args_with_default = vcat("@$(pkgdir(TuringCLIExample))/src/defaults.txt", args)
    return parse_args(args_with_default, s;as_symbols=true)  
end
end #module
