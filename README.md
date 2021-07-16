# Demo of CLI for Turing and Tensorboard

## Installation
Do an
Then
`julia --project -e "using Pkg; Pkg.instantiate()"`

To build the sysimage
```bash
julia --threads auto -e 'using Pkg; Pkg.add(\"PackageCompiler\")'
julia --threads auto ./deps/create_sysimage.jl
```
## CLI Usage
To use with default options
```bash
julia --project --threads auto bin/fit.jl
```
(although the `--threads auto` may or may not be useful in this example.)

Or if you have a sysimage:
```bash
julia --project --threads auto --sysimage JuliaSysimage.dll bin/fit.jl
```

Or with options
```bash
julia --project --threads auto bin/fit.jl --num_samples 1000
```
To run tensorboard, ensure tensorboard is installed (e.g. with  `pip install -r requirements.txt` ) and
```bash
tensorboard --logdir tensorboard_logs
```
## Options
```bash
❯ julia --project bin/fit.jl --help
usage: fit.jl [--num_samples NUM_SAMPLES] [--num_adapts NUM_ADAPTS]
              [--target_acceptance_rate TARGET_ACCEPTANCE_RATE]
              [--s_prior_alpha S_PRIOR_ALPHA]
              [--s_prior_theta S_PRIOR_THETA] [-h]

optional arguments:
  --num_samples NUM_SAMPLES
                        samples to draw in chain (type: Int64,
                        default: 10000)
  --num_adapts NUM_ADAPTS
                        number of adaptations for NUTS (type: Int64,
                        default: 100)
  --target_acceptance_rate TARGET_ACCEPTANCE_RATE
                        Target acceptance rate for dual averaging.
                        (type: Float64, default: 0.65)
  --s_prior_alpha S_PRIOR_ALPHA
                        alpha in InverseGamma prior for s (type:
                        Float64, default: 2.0)
  --s_prior_theta S_PRIOR_THETA
                        theta in InverseGamma prior for s (type:
                        Float64, default: 3.0)
  -h, --help            show this help message and exit
```