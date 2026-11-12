# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.11.0", "PyYAML==6.0.3"]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# ///
"""Prepare a self-contained Reach TorchScript bundle from an existing play export.

Unlike LEAPP, this bundle uses reach_policy.py for observation/action processing.
Only the explicitly validated six-joint, 43-observation Reach interface is accepted.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml

if __package__:
    from .reach_policy import JOINT_NAMES, OBSERVATION_TERMS, ReachController, read_kinematic_chain, sha256
else:
    from reach_policy import JOINT_NAMES, OBSERVATION_TERMS, ReachController, read_kinematic_chain, sha256


class ConfigLoader(yaml.SafeLoader):
    """Read Isaac Lab YAML as data; never instantiate Python objects."""


ConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", ConfigLoader.construct_sequence)
ConfigLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice", ConfigLoader.construct_sequence)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def contract_from_config(env, agent, urdf):
    action = env["actions"]["arm_action"]
    require(action.get("target_update_decimation", 1) == 1, "Export only the original physics-rate training contract")
    require(
        action["class_type"]
        in {
            "mycobot450_isaaclab.tasks.reach.mdp.actions:SmoothJointVelocityAction",
            "mycobot450_isaaclab.tasks.reach.mdp:SmoothJointVelocityAction",  # Before the config/mdp reorganization.
        },
        "Only SmoothJointVelocityAction is supported",
    )
    require(action["joint_names"] == JOINT_NAMES and action["preserve_order"], "Unsupported action joint order")
    require(
        not action["use_default_offset"] and action["offset"] == 0 and action["clip"] is None,
        "Unsupported action offset/clip",
    )
    require(env["actions"].get("gripper_action") is None, "Reach deployment does not command the F100")
    require(agent["clip_actions"] == 1.0, "Expected RSL-RL clip_actions=1")
    observations = env["observations"]["policy"]
    require(
        observations["concatenate_terms"] and observations.get("history_length") in (None, 0),
        "Unsupported observation group",
    )
    terms = {key: value for key, value in observations.items() if isinstance(value, dict) and "func" in value}
    require(list(terms) == list(OBSERVATION_TERMS), "Unexpected observation terms or ordering")
    for name, term in terms.items():
        expected = OBSERVATION_TERMS[name]
        supported = {expected}
        if expected.startswith("mycobot450_"):
            supported.add(expected.replace(".mdp.observations:", ".mdp:"))
        require(term["func"] in supported, f"Unsupported observation function: {name}")
        for field in ("scale", "clip", "modifiers"):
            require(term.get(field) is None, f"Unsupported {name}.{field}")
        for field in ("history_length", "delay_min_lag", "delay_max_lag"):
            require(term.get(field, 0) == 0, f"Unsupported {name}.{field}")
    for name in ("joint_pos", "joint_vel"):
        selection = terms[name]["params"]["asset_cfg"]
        require(
            selection["joint_names"] == JOINT_NAMES and selection["preserve_order"], f"Unsupported {name} joint order"
        )
    require(terms["pose_command"]["params"] == {"command_name": "ee_pose"}, "Unsupported pose command")
    require(terms["pose_error"]["params"] in ({}, {"command_name": "ee_pose"}), "Unsupported pose error")
    initial = env["scene"]["robot"]["init_state"]
    require(all(value == 0 for value in initial["joint_vel"].values()), "Nonzero default joint velocities")
    command = env["commands"]["ee_pose"]
    require(command["body_name"] == "f100_base_link" and command["make_quat_unique"], "Unexpected tracked frame")
    return {
        "version": 1,
        "action_type": "filtered_joint_velocity_integrated_position_v1",
        "joint_order": JOINT_NAMES,
        "observation_dim": 43,
        "quaternion_order": "xyzw",
        "tracked_frame": "f100_base_link",
        "home_rad": [initial["joint_pos"][name] for name in JOINT_NAMES],
        "joint_limits_rad": [action["target_limits"][name] for name in JOINT_NAMES],
        "velocity_scale_rad_s": action["scale"],
        "filter_time_constant_s": action["time_constant"],
        "max_tracking_error_rad": action.get("max_tracking_error"),
        "control_dt_s": env["sim"]["dt"],
        "policy_dt_s": env["sim"]["dt"] * env["decimation"],
        "workspace_m": [command["ranges"]["pos_" + axis] for axis in "xyz"],
        "kinematic_chain": read_kinematic_chain(urdf),
        "urdf_sha256": sha256(urdf),
        "normalization": "inside policy.pt; supply raw physical observations",
    }


def prepare_bundle(checkpoint, policy, output, params_dir=None):
    checkpoint, policy, output = Path(checkpoint), Path(policy), Path(output)
    require(not output.exists(), "Use a new output directory; deployment bundles are never overwritten")
    params = Path(params_dir) if params_dir else checkpoint.parent / "params"
    env = yaml.load((params / "env.yaml").read_text(), Loader=ConfigLoader)
    agent = yaml.load((params / "agent.yaml").read_text(), Loader=ConfigLoader)
    urdf = Path(__file__).resolve().parents[1] / "assets/data/robots/sim_450_f100/config/450_f100.urdf"
    contract = contract_from_config(env, agent, urdf)
    controller = ReachController(contract)
    actor = torch.load(checkpoint, map_location="cpu", weights_only=True)["actor_state_dict"]
    model = torch.jit.load(str(policy), map_location="cpu").eval()
    states = model.state_dict()
    require(set(states) == set(actor) - {"distribution.std_param"}, "Export/checkpoint state keys do not match")
    for key, value in states.items():
        require(torch.equal(value, actor[key]), f"Export does not belong to checkpoint: {key}")
        require(torch.isfinite(value).all(), f"Non-finite model state: {key}")
    require(float(model.obs_normalizer.eps) == 0.01, "Unexpected observation normalization epsilon")
    # Independently compare the exported computation, not only its saved weights.
    generator = torch.Generator().manual_seed(450)
    observations = torch.randn(32, 43, generator=generator)
    with torch.inference_mode():
        expected = (observations - actor["obs_normalizer._mean"]) / (actor["obs_normalizer._std"] + 0.01)
        for index in (0, 2, 4, 6):
            expected = torch.nn.functional.linear(expected, actor[f"mlp.{index}.weight"], actor[f"mlp.{index}.bias"])
            if index < 6:
                expected = torch.nn.functional.elu(expected)
        actual = model(observations)
        require(actual.shape == (32, 6), "Expected a 43-input, six-output policy")
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        pose = controller.fk(controller.home)
        sample = controller.observe(controller.home, torch.zeros(6), pose)
        require(torch.isfinite(model(sample)).all(), "Invalid home-pose policy output")
    contract.update(
        {
            "policy_sha256": sha256(policy),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "env_sha256": sha256(params / "env.yaml"),
            "agent_sha256": sha256(params / "agent.yaml"),
            "export_max_abs_error": (actual - expected).abs().max().item(),
        }
    )
    output.mkdir(parents=True)
    try:
        shutil.copyfile(policy, output / "policy.pt")
        (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy", type=Path, help="Default: checkpoint directory/exported/policy.pt")
    parser.add_argument("--params-dir", type=Path, help="Saved params directory for a copied checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract = prepare_bundle(
        args.checkpoint, args.policy or args.checkpoint.parent / "exported/policy.pt", args.output, args.params_dir
    )
    print(
        json.dumps(
            {
                "bundle": str(args.output),
                "policy_sha256": contract["policy_sha256"],
                "max_abs_error": contract["export_max_abs_error"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
