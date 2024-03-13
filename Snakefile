import os

session_ids = ["{}".format(i) for i in range(1, 10)]

rule all:
    input:
        ["results/info/system.toml",
        "results/statistics/gee_full.pkl",
         "results/empirical/transfer_model_comparison.npz",
         "results/simulation/parameter_space_exploration.npz",
         "results/simulation/crossval_estimation.npz",
         "results/simulation/learning_simulation.npz",
         "results/simulation/highres_arnold_tongues.npy",
         "results/figures/figure_two/panel_d.svg",
         "results/figures/figure_three/bottom_row_transfer.svg"]

rule run_system_info:
    output:
        "results/info/system.toml"
    shell:
        "python -m scripts.info.system"

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

rule create_figure_two:
    input:
        expand("results/empirical/session_{session}/average_bat.npy", session=session_ids) +
        expand("results/empirical/session_{session}/continuous_bat.npy", session=session_ids) +
        ["results/simulation/parameter_space_exploration.npz",
        "results/simulation/highres_arnold_tongues.npy"]
    output:
        ["results/figures/figure_two/panel_a.svg",
        "results/figures/figure_two/panel_b.svg",
        "results/figures/figure_two/panel_c.svg",
        "results/figures/figure_two/panel_d.svg"]
    shell:
        "python -m scripts.plotting.figure_two"

rule run_figure_three:
    input:
        ["results/simulation/highres_arnold_tongues.npy",
        "results/empirical/transfer_model_comparison.npz"] +
        expand("results/empirical/session_{session}/average_bat.npy", session=session_ids) +
        expand("results/empirical/session_{session}/continuous_bat.npy", session=session_ids)
    output:
        "results/figures/figure_three/bottom_row_transfer.svg"
    shell:
        "python -m scripts.plotting.figure_three"
