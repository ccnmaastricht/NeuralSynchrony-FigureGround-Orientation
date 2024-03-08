import os

session_ids = ["{}".format(i) for i in range(1, 10)]

rule all:
    input:
        ["results/statistics/gee_full.pkl",
         "results/empirical/transfer_model_comparison.npz",
         "results/simulation/parameter_space_exploration.npz",
         "results/simulation/crossval_estimation.npz",
         "results/simulation/learning_simulation.npz",
         "results/simulation/highres_arnold_tongues.npy",
         "figures/first_figure/panel_d.svg",
         "figures/second_figure/bottom_row_transfer.svg"]

rule run_statistics:
    output:
        "results/statistics/gee_full.pkl"
    shell:
        "python -m scripts.statistics.gee_accuracy"

rule run_behavioral_arnold_tongue:
    input:
        "results/statistics/gee_full.pkl"
    output:
        expand("results/empirical/session_{session}/optimal_psychometric_parameters.npy", session=session_ids) +
        expand("results/empirical/session_{session}/average_bat.npy", session=session_ids) +
        expand("results/empirical/session_{session}/continuous_bat.npy", session=session_ids) +
        expand("results/empirical/session_{session}/individual_bats.npy", session=session_ids)
    shell:
        "python -m scripts.analysis.behavioral_arnold_tongue"

rule run_transfer_session_model_comparison:
    input:
        expand("results/empirical/session_{session}/average_bat.npy", session=session_ids)
    output:
        "results/empirical/transfer_model_comparison.npz"
    shell:
        "python -m scripts.analysis.transfer_session_model_comparison"

rule run_parameter_exploration:
    input:
        expand("results/empirical/session_{session}/average_bat.npy", session=session_ids)
    output:
        "results/simulation/parameter_space_exploration.npz"
    shell:
        "python -m scripts.simulation.parameter_exploration"

rule run_crossval_estimation:
    input:
        expand("results/empirical/session_{session}/individual_bats.npy", session=session_ids)
    output:
        "results/simulation/crossval_estimation.npz"
    shell:
        "python -m scripts.simulation.crossval_estimation"

rule run_crossval_prediction:
    input:
        ["results/simulation/crossval_estimation.npz"] +
        expand("results/empirical/session_{session}/individual_bats.npy", session=session_ids)
    output:
        "results/simulation/learning_simulation.npz"
    shell:
        "python -m scripts.simulation.crossval_prediction"

rule run_high_resolution_simulations:
    input:
        "results/simulation/crossval_estimation.npz"
    output:
        "results/simulation/highres_arnold_tongues.npy"
    shell:
        "python -m scripts.simulation.high_resolution_simulations"

rule run_first_figure:
    input:
        expand("results/empirical/session_{session}/average_bat.npy", session=session_ids) +
        expand("results/empirical/session_{session}/continuous_bat.npy", session=session_ids) +
        ["results/simulation/parameter_space_exploration.npz",
        "results/simulation/highres_arnold_tongues.npy"]
    output:
        ["figures/first_figure/panel_a.svg",
        "figures/first_figure/panel_b.svg",
        "figures/first_figure/panel_c.svg",
        "figures/first_figure/panel_d.svg"]
    shell:
        "python -m scripts.plotting.first_figure"

rule run_second_figure:
    input:
        ["results/simulation/highres_arnold_tongues.npy",
        "results/empirical/transfer_model_comparison.npz"] +
        expand("results/empirical/session_{session}/average_bat.npy", session=session_ids) +
        expand("results/empirical/session_{session}/continuous_bat.npy", session=session_ids)
    output:
        "figures/second_figure/bottom_row_transfer.svg"
    shell:
        "python -m scripts.plotting.second_figure"
