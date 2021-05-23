# see tutorial https://turinglang.github.io/TuringCallbacks.jl/dev/

module TuringCLIExample

using Turing, TuringCallbacks, TensorBoardLogger, ArgParse

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
    sum_stats = describe(chain)
    param_names = sum_stats[1][:, 1]
    param_rhat = sum_stats[1][:, 7]
    param_ess_per_sec = sum_stats[1][:, 8]

    # Log the ESS/sec and rhat.  Nice to show as summary results from tensorboard
    for i = 1:length(param_names)
        TensorBoardLogger.log_value(
            callback.logger,
            "$(param_names[i])_ess_per_sec",
            param_ess_per_sec[i],
        )
        TensorBoardLogger.log_value(
            callback.logger,
            "$(param_names[i])_rhat",
            param_rhat[i],
        )
    end
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

end #module