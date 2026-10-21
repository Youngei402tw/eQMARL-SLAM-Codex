from pathlib import Path

from eqmarl_slam.execution import load_config


def test_committed_scientific_notation_keeps_numeric_type():
    config = load_config(Path(__file__).parents[1] / "configs/stage03/beliefs.yaml")
    assert config["gates"]["new_unknown_landmark_pose_covariance_atol"] == 1e-7
    assert config["gates"]["same_step_trace_reduction_min"] == 1e-9


def test_native_yaml_configs_still_supported(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("stage: '03'\nthreshold: 0.001\n")
    assert load_config(path) == {"stage": "03", "threshold": .001}
