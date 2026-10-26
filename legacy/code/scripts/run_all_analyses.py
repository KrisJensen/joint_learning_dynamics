
import argparse
import subprocess as sp
ROOT = "/ceph/behrens/kris/research/learning_codex/code/"

output_subdir = "examples"

#%% compare empirical and analytical
sp.call(["python", ROOT + "scripts/compare_empirical_analytical.py", "--eta-e", "0.1", "--eta-on1", "1.5"] )
sp.call(["python", ROOT + "plot_functions/plot_empirical_analytical.py"])

#%% compare empirical and analytical with partial WM evaluation, M=3
sp.call(["python", ROOT + "scripts/compare_empirical_analytical.py", "--eta-e", "0.1", "--eta-on1", "1.5", "--prefix-length", "3", "--suffix", "prefix_M3"] )
sp.call(["python", ROOT + "plot_functions/plot_empirical_analytical.py", "--suffix", "prefix_M3"])

#%% compare 'RL' and 'copying' for different degrees of planner goodness
sp.call(["python", ROOT + "scripts/compare_rho_me.py", "--fix-eta-on1-zero", "--suffix", "eta_on1_zero", "--eta-on2", "2.5"] )
sp.call(["python", ROOT + "plot_functions/plot_rho_me.py", "--suffix", "eta_on1_zero"])

sp.call(["python", ROOT + "scripts/compare_rho_me.py", "--fix-eta-on2-zero", "--suffix", "eta_on2_zero", "--eta-on1", "2.5"] )
sp.call(["python", ROOT + "plot_functions/plot_rho_me.py", "--suffix", "eta_on2_zero"])

#%% compare different prefix lengths
sp.call(["python", ROOT + "scripts/compare_prefix_lengths.py", "--temperature-mode", "optimal", "--T", "9", "--eta-max", "20.0","--K", "10", "--learning-rate-mode", "optimal", "--suffix", "optimal"] )
sp.call(["python", ROOT + "plot_functions/plot_prefix_lengths.py", "--suffix", "optimal"])

sp.call(["python", ROOT + "scripts/compare_prefix_lengths.py", "--eta-on2", "2.5", "--eta-on1", "2.5", "--temperature-mode", "optimal", "--T", "9", "--K", "10", "--suffix", "constant", "--duration", "30"] )
sp.call(["python", ROOT + "plot_functions/plot_prefix_lengths.py", "--suffix", "constant"])

#%% run learning rate comparisons
sp.call(["python", ROOT + "scripts/compare_learning_rates.py"] )
sp.call(["python", ROOT + "plot_functions/plot_learning_rates.py"])


# #%% temperature sweep
sp.call(["python", ROOT + "scripts/compare_temperatures.py", "--rho-me0", "1.0", "--eta-wm", "0.0"] )
sp.call(["python", ROOT + "plot_functions/plot_temperatures.py"])

#%% different values of K

sp.call(["python", ROOT + "scripts/compare_K.py", "--rho-me0", "1.0", "--eta-wm", "0.0", "--suffix", "student"] )
sp.call(["python", ROOT + "plot_functions/plot_K.py", "--suffix", "student"])

sp.call(["python", ROOT + "scripts/compare_K.py", "--suffix", "joint"] )
sp.call(["python", ROOT + "plot_functions/plot_K.py", "--suffix", "joint"])


#%% compare RL and planning for different values of T

sp.call(["python", ROOT + "scripts/compare_rl_planning_T.py"] )
sp.call(["python", ROOT + "plot_functions/plot_rl_planning_T.py"])

#%% compare planning/greedy training and evaluation regimes
sp.call(["python", ROOT + "scripts/compare_policy_regimes.py"])

#%% compare global environment-modulation amplitudes
sp.call(["python", ROOT + "scripts/compare_modulation.py", "--eta-e", "0.1"])
sp.call(["python", ROOT + "plot_functions/plot_modulation.py"])

#%% try dynamic programming

sp.call(["python", ROOT + "scripts/run_dynamic_programming.py", "--variant", "online", "--fixed-wm", "--fixed-rho-me", "0.999", "--partial-evaluation", "--tau-element", str(0.0625/8), "--temperature-mode", "optimal", "--suffix", "optimal_KM_copy", "--dp-step", "0.001", "--T", "8", "--horizon", "40", "--rho-points", "100", "--eta-on1", "2.5", "--eta-on2",  "0.0"])
sp.call(["python", ROOT + "plot_functions/plot_dynamic_programming.py", "--variant", "online", "--fixed-wm", "--partial-evaluation", "--suffix", "optimal_KM_copy"])

## also run for switching

sp.call(["python", ROOT + "scripts/run_dynamic_programming.py", "--variant", "switching"])
sp.call(["python", ROOT + "plot_functions/plot_dynamic_programming.py", "--variant", "switching"])

## also run with environment modulation

sp.call(["python", ROOT + "scripts/run_dynamic_programming.py", "--variant", "switching", "--switching-cost", "0.05", "--learning-rate-mode", "optimal", "--p-flip", "0.00", "--fix-eta-on1-zero", "--eta-e", "0.3", "--rho-points", "31", "--suffix", "drift", "--horizon", "9.0", "--dp-step", "0.002", "--eta-max", "30", "--tau-on", "0.05", "--tau-off", "0.05", "--T-M", "2.0", "--phi-M", "-0.0", "--M", "0.95", "--interpolation", "structured"])
sp.call(["python", ROOT + "plot_functions/plot_dynamic_programming.py", "--variant", "switching", "--suffix", "drift"])


#%% phase diagram

sp.call(["python", ROOT + "scripts/phase_diagram_initial.py", "--switching-cost", "0.5", "--T", "6", "--temperature-mode", "constant", "--beta", "4.0", "--suffix", "constant_beta", "--horizon", "16.0"])
sp.call(["python", ROOT + "plot_functions/plot_phase_diagrams.py", "--analysis", "initial", "--switching-cost", "0.5", "--T", "6", "--suffix", "constant_beta"])
sp.call(["python", ROOT + "plot_functions/plot_phase_examples.py", "--analysis", "initial", "--switching-cost", "0.5", "--T", "6", "--suffix", "constant_beta", "--examples", "0.0,0.0;0.0,0.8;0.5,0.5"]) # plot 


