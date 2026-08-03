#%%
import numpy as np
import subprocess as sp
ROOT = "/ceph/behrens/kris/research/joint_learning_dynamics/code/"

subdir = "report"


#%% generate figure that qualitatively compares different types of learning
sp.call(["python", ROOT + "scripts/compare_policy_regimes.py", "--output-subdir", subdir] )
sp.call(["python", ROOT + "plot/plot_policy_regimes.py", "--output-subdir", subdir] )

#%% compare action correctness probabilities for different prefix lengths
sp.call(["python", ROOT + "scripts/compare_prefix_length_observables.py", "--output-subdir", subdir] )
sp.call(["python", ROOT + "plot/plot_prefix_length_observables.py", "--output-subdir", subdir] )

#%% compare learning for different prefix lengths
sp.call(["python", ROOT + "scripts/compare_prefix_lengths.py", "--output-subdir", subdir, "--suffix", "T6", "--T", "6", "--K", "10"] )

sp.call(["python", ROOT + "plot/plot_prefix_lengths.py", "--output-subdir", subdir, "--suffix", "T6", "--output", ROOT + "../figures/report/prefix_length_comparison_2factor.pdf", "--include-fixed-M", "--include-optimal-M"] )
sp.call(["python", ROOT + "plot/plot_prefix_lengths.py", "--output-subdir", subdir, "--suffix", "T6", "--output", ROOT + "../figures/report/prefix_length_comparison_3factor.pdf", "--include-optimal-M", "--include-eta-3f"] )


sp.call(["python", ROOT + "scripts/compare_prefix_lengths.py", "--output-subdir", subdir, "--T", "9", "--K", "10", "--suffix", "T9"] )
sp.call(["python", ROOT + "plot/plot_prefix_lengths.py", "--output-subdir", subdir, "--suffix", "T9", "--output", ROOT + "../figures/report/prefix_length_comparison_3factor-long.pdf", "--include-optimal-M", "--include-eta-3f"] )

#%% now run dynamic programming

T = "6"
shared_args = ["python", ROOT + "scripts/run_dynamic_programming.py", "--temperature-mode", "optimal", "--dp-step-size", "0.125", "--T", T, "--horizon", "20", "--eta-2f", "1.0", "--learning-rate-mode", "optimal", "--eta-max", "1.0", "--output-subdir", subdir, "--delta-on", "0.0625", "--delta-off", "0.0625", "--dp-K-max", "10", "--rho-points", "25", "--t-switch", "1.0"]

# solve online-only and switching 2-factor DPs from one action workspace
sp.call(shared_args+["--variant", "both", "--online-suffix", f"2f_{T}", "--switching-suffix", f"2f_{T}", "--fix-eta-3f-zero"])
sp.call(["python", ROOT + "plot/plot_dynamic_programming.py", "--variant", "online", "--suffix", f"2f_{T}", "--output-subdir", subdir, "--plot-individuals"])

sp.call(["python", ROOT + "plot/plot_dynamic_programming.py", "--variant", "switching", "--suffix", f"2f_{T}", "--output-subdir", subdir, "--plot-individuals"])

# solve online-only and switching 3-factor DPs from a second shared workspace
sp.call(shared_args+["--variant", "both", "--online-suffix", f"3f_{T}", "--switching-suffix", f"3f_{T}", "--fix-eta-2f-zero", "--dp-M", T])
sp.call(["python", ROOT + "plot/plot_dynamic_programming.py", "--variant", "online", "--suffix", f"3f_{T}", "--output-subdir", subdir, "--plot-individuals"])

sp.call(["python", ROOT + "plot/plot_dynamic_programming.py", "--variant", "switching", "--suffix", f"3f_{T}", "--output-subdir", subdir, "--plot-individuals"])


#%% add phase modulation

sp.run(["python", ROOT+"scripts/run_dynamic_programming.py", "--variant", "switching", "--t-switch", "0.1", "--learning-rate-mode",
        "optimal","--p-flip","0","--fix-eta-2f-zero","--T","5","--dp-M","5","--eta-e","0.3","--rho-points", "31",
        "--suffix","drift","--horizon","9","--dp-step-size","0.1","--eta-max", "3", "--delta-on","0.05", "--delta-off",
        "0.05", "--C-T", "2", "--C-phi", "0", "--C-A", "0.95", "--with-modulation", "--output-subdir", subdir])
sp.run(["python", ROOT+"plot/plot_dynamic_programming.py", "--variant","switching","--suffix","drift", "--output-subdir", subdir, "--plot-individuals"])




#%% generate phase diagram

rho_se_values = ",".join(
    f"{value:.12g}" for value in np.linspace(0.0, 0.99, 50)
)
rho_me_values = ",".join(
    f"{value:.12g}" for value in np.linspace(0.0, 0.999, 26)
)

T = "6"
mode = ["--fix-eta-2f-zero", "--dp-M", T]
suffix = "3f"

mode = ["--fix-eta-3f-zero"]
suffix = "2f"


sp.run(["python", ROOT+"scripts/phase_diagram_initial.py", "--switch-times","1.0, 2.0, 3.0","--T",T,*mode,"--temperature-mode","constant",
        "--beta","3","--initial-horizons","10,15,20","--dp-step-size","0.2","--rho-points","26","--dp-K-max","10", "--suffix", suffix, "--learning-rate-mode", "optimal", "--eta-max", "1.0", "--delta-on","0.05","--delta-off","0.05",
        "--rho-se-values", rho_se_values, "--rho-me-values", rho_me_values, "--output-subdir", subdir])

sp.run(["python", ROOT+"plot/plot_phase_diagrams.py", "--output-subdir", subdir, "--switch-times","1.0,2.0,3.0", "--suffix", suffix, "--initial-horizons","10,15,20"])
sp.run(["python", ROOT+"plot/plot_phase_examples.py", "--examples", "0.05,0.05;0.05,0.9;0.4,0.9", "--suffix", suffix, "--output-subdir", subdir])#, "--switch-times","1.0,2.0,3.0","--initial-horizons","10,15,20"])
