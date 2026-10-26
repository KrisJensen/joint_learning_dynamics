#%%
import subprocess as sp
ROOT = "/ceph/behrens/kris/research/learning_codex/code/"

subdir = "report"


#%% generate figure that qualitatively compares different types of learning
sp.call(["python", ROOT + "scripts/compare_policy_regimes.py", "--output-subdir", subdir] )

#%% compare action correctness probabilities for different prefix lengths
sp.call(["python", ROOT + "scripts/compare_prefix_length_observables.py", "--output-subdir", subdir] )

#%% compare learning for different prefix lengths
sp.call(["python", ROOT + "scripts/compare_prefix_lengths.py", "--output-subdir", subdir, "--include-constant-prefixes", "--include-optimal-prefix", "--suffix", "2factor", "--T", "6", "--K", "10"] )
sp.call(["python", ROOT + "plot_functions/plot_prefix_lengths.py", "--output-subdir", subdir, "--suffix", "2factor"] )

sp.call(["python", ROOT + "scripts/compare_prefix_lengths.py", "--output-subdir", subdir, "--include-optimal-prefix", "--include-eta-on2", "--suffix", "3factor", "--T", "6", "--K", "10"] )
sp.call(["python", ROOT + "plot_functions/plot_prefix_lengths.py", "--output-subdir", subdir, "--suffix", "3factor"] )


sp.call(["python", ROOT + "scripts/compare_prefix_lengths.py", "--output-subdir", subdir, "--include-optimal-prefix", "--include-eta-on2", "--T", "9", "--K", "10", "--suffix", "3factor-long"] )
sp.call(["python", ROOT + "plot_functions/plot_prefix_lengths.py", "--output-subdir", subdir, "--suffix", "3factor-long"] )

#%% now run dynamic programming

shared_args = ["python", ROOT + "scripts/run_dynamic_programming.py", "--partial-evaluation", "--tau-element", str(0.0625/8), "--temperature-mode", "optimal", "--dp-step", "0.001", "--T", "8", "--horizon", "30", "--eta-on1", "1.5", "--eta-on2",  "0.0", "--eta-off", "1.5", "--eta-wm", "1.5", "--output-subdir", subdir, "--tau-on", "0.0625", "--tau-off", "0.0625"]

# first with no offline period
sp.call(shared_args+["--variant", "online", "--suffix", "optimal_online", "--rho-points", "25"])

sp.call(["python", ROOT + "plot_functions/plot_dynamic_programming.py", "--variant", "online", "--partial-evaluation", "--suffix", "optimal_online", "--output-subdir", subdir, "--plot-individuals"])

# now run with offline

sp.call(shared_args+["--variant", "switching", "--suffix", "optimal_switching", "--rho-points", "25"])


sp.call(["python", ROOT + "plot_functions/plot_dynamic_programming.py", "--variant", "switching", "--partial-evaluation", "--suffix", "optimal_switching", "--output-subdir", subdir, "--plot-individuals"])


