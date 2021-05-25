# see tutorial https://turinglang.github.io/TuringCallbacks.jl/dev/

module TuringCLIExample

using Turing, TuringCallbacks, TensorBoardLogger, ArgParse, StatsPlots, CSV, DataFrames

function simulate_and_estimate(;
    num_samples,
    num_adapts,
    target_acceptance_rate,
    s_prior_alpha,
    s_prior_theta,
    kwargs...,
)
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
    println("Generating $num_samples samples")
    callback = TensorBoardCallback("tensorboard_logs/run")
    alg = NUTS(num_adapts, target_acceptance_rate)
    chain = sample(model, alg, num_samples; callback)

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
function main()
    d = parse_commandline()

    # Generic code to convert the dictionary to 
    dictkeys = (collect(Symbol.(keys(d)))...,)
    dictvalues = (collect(values(d))...,)
    args = NamedTuple{dictkeys}(dictvalues)
    # parses converts all arguments to named tuple then splat into solution
    simulate_and_estimate(; args...)
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--num_samples"
        help = "samples to draw in chain"
        arg_type = Int64
        default = 10000
        "--num_adapts"
        help = "number of adaptations for NUTS"
        arg_type = Int64
        default = 100
        "--target_acceptance_rate"
        help = "Target acceptance rate for dual averaging."
        arg_type = Float64
        default = 0.65
        "--s_prior_alpha"
        help = "alpha in InverseGamma prior for s"
        arg_type = Float64
        default = 2.0
        "--s_prior_theta"
        help = "theta in InverseGamma prior for s"
        arg_type = Float64
        default = 3.0
    end

    return parse_args(s)
end

# Can experiment later, or can package compiler 
# function _precompile_()
#     ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
#     Base.precompile(Tuple{typeof(main)})
# end

# # Only really intended for final versions rather than during interactive use
# # NOTE: Consider commenting out or fix if changing signature of simulate_and_estimte
# if Base.VERSION >= v"1.4.2"
#     precompile(Tuple{typeof(TuringCLIExample.main)})
    
#     # Is this necessary or does the main() above do it? Comment out if any problem.
#     precompile(Tuple{TuringCLIExample.var"#simulate_and_estimate##kw", NamedTuple{(:num_samples, :num_adapts, :target_acceptance_rate, :s_prior_theta, :s_prior_alpha), Tuple{Int64, Int64, Float64, Float64, Float64}}, typeof(TuringCLIExample.simulate_and_estimate)})

#     _precompile_()
# end
end #module
