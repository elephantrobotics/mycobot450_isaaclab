# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the MyCobot Pro450 + F100 gripper."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.schemas.schemas_cfg import ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.configclass import configclass


def resolve_mycobot_pro450_usd_path() -> str:
    """Resolve the combined Pro450 + F100 USD used by Isaac Lab."""
    overlay = Path(__file__).resolve().parents[3] / "assets" / "robots" / "mycobot_pro450_lift.usda"
    env_path = os.environ.get("MYCOBOT_PRO450_USD_PATH")
    repo_root = Path(__file__).resolve().parents[5]
    candidates = [
        overlay,
        Path(env_path) if env_path else None,
        repo_root.parent / "mycobot450_isaacsim" / "USD" / "sim_450_f100" / "450_f100.usda",
        Path.home() / "mycobot450_isaacsim" / "USD" / "sim_450_f100" / "450_f100.usda",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return str(candidate)
    searched = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        "MyCobot Pro450 USD not found. Set MYCOBOT_PRO450_USD_PATH or place 450_f100.usda "
        f"under the Isaac Sim repo. Searched: {searched}"
    )


MYCOBOT_PRO450_USD_PATH = resolve_mycobot_pro450_usd_path()

_GRIPPER_ROOT_SCHEMA_NAMES = (
    "PhysicsArticulationRootAPI",
    "NewtonArticulationRootAPI",
    "PhysxArticulationAPI",
)


def _is_arm_articulation_root(path: str) -> bool:
    return path.endswith("/Geometry") and "/mygripper" not in path


def _remove_extra_articulation_roots(root_prim) -> None:
    """Keep a single arm ArticulationRoot so Isaac Lab can wrap arm + gripper together."""
    from pxr import Usd, UsdPhysics

    try:
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None

    for prim in Usd.PrimRange(root_prim):
        if prim.IsInstanceProxy():
            continue
        path = str(prim.GetPath())
        if _is_arm_articulation_root(path):
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if PhysxSchema is not None and prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        for schema_name in _GRIPPER_ROOT_SCHEMA_NAMES:
            try:
                prim.RemoveAPI(schema_name)
            except Exception:
                continue


def _strip_extra_roots_for_expr(prim_path_expr: str) -> None:
    from isaaclab.sim.utils import find_matching_prims

    for prim in find_matching_prims(prim_path_expr):
        _remove_extra_articulation_roots(prim)


def _bind_instanced_collision_materials(root_prim, material_path: str) -> None:
    """USD spawn only binds friction on non-instance prims; F100 pads are instanceable meshes."""
    from pxr import Usd

    from isaaclab.sim.utils import bind_physics_material

    stage = root_prim.GetStage()
    if not stage.GetPrimAtPath(material_path).IsValid():
        return
    seen: set[str] = set()
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsInstance():
            continue
        proto = prim.GetPrototype()
        proto_path = str(proto.GetPath())
        if not proto.IsValid() or proto_path in seen:
            continue
        seen.add(proto_path)
        bind_physics_material(proto_path, material_path, stage=stage, stronger_than_descendants=True)


# Match URDF J3 visual rpy about +Z so the box sits on the pad, not in the jaw gap.
_PAD_COLLIDER_SPECS = (
    (
        "mygripper_f100/Geometry/f100_base_link/joint2_left_link/joint3_left_link/pad_col",
        (0.010, 0.0, 0.0),
        (0.0, 0.0, -0.481, 0.877),
    ),
    (
        "mygripper_f100/Geometry/f100_base_link/joint2_right_link/joint3_right_link/pad_col",
        (0.010, 0.0, 0.0),
        (0.0, 0.0, 0.481, 0.877),
    ),
)


def _add_pad_box_colliders(root_prim) -> None:
    """Thin high-friction boxes on the F100 pads. These are not instanceable, so friction binds."""
    stage = root_prim.GetStage()
    root = str(root_prim.GetPath())
    pad_cfg = sim_utils.CuboidCfg(
        size=(0.028, 0.008, 0.022),
        visible=False,
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.5,
            dynamic_friction=2.0,
            restitution=0.0,
            friction_combine_mode="max",
        ),
    )
    for rel_path, translation, orientation in _PAD_COLLIDER_SPECS:
        prim_path = f"{root}/{rel_path}"
        if stage.GetPrimAtPath(prim_path).IsValid():
            continue
        pad_cfg.func(prim_path, pad_cfg, translation=translation, orientation=orientation)


def spawn_mycobot_pro450_usd(prim_path, cfg, translation=None, orientation=None):
    """Spawn the 450+F100 USD and merge the gripper into the arm articulation."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from isaaclab.sim.utils import select_usd_variants

    prim = spawn_from_usd(prim_path, cfg, translation, orientation)
    select_usd_variants(str(prim.GetPath()), {"Gripper": "base", "Physics": "physx"})
    _remove_extra_articulation_roots(prim)
    _add_pad_box_colliders(prim)
    _bind_instanced_collision_materials(prim, f"{prim.GetPath()}/physicsMaterial")
    return prim


@configclass
class MycobotPro450ArticulationCfg(ArticulationCfg):
    """Articulation cfg that merges the F100 gripper into the arm tree."""

    def _post_spawn(self, stage) -> None:
        author_prim_path = (
            self.spawn.spawn_path if self.spawn is not None and self.spawn.spawn_path is not None else self.prim_path
        )
        from isaaclab.sim.utils import find_matching_prims

        for prim in find_matching_prims(author_prim_path):
            _remove_extra_articulation_roots(prim)
            _add_pad_box_colliders(prim)
            _bind_instanced_collision_materials(prim, f"{prim.GetPath()}/physicsMaterial")
        super()._post_spawn(stage)


MYCOBOT_PRO450_ARM_JOINT_NAMES = ["joint[1-6]"]
MYCOBOT_PRO450_GRIPPER_JOINT_NAMES = ["joint2_left_joint"]
MYCOBOT_PRO450_GRIPPER_MIMIC_JOINT_NAMES = [
    "joint1_left_joint",
    "joint2_right_joint",
    "joint1_right_joint",
    "joint3_left_joint",
    "joint3_right_joint",
]
MYCOBOT_PRO450_BASE_BODY_NAME = "base_link"
MYCOBOT_PRO450_EE_BODY_NAME = "f100_base_link"
MYCOBOT_PRO450_EE_PRIM_PATH = "{ENV_REGEX_NS}/Robot/mygripper_f100/Geometry/f100_base_link"
MYCOBOT_PRO450_BASE_PRIM_PATH = "{ENV_REGEX_NS}/Robot/Geometry/base_link"
# F100 approach axis is +X of f100_base_link (world -Z when hovering over the table).
# Pad center: 0.127 m open, 0.146 m closed. Approach uses the open value so the cube
# enters the jaw gap instead of the fingertips hitting the cube first.
MYCOBOT_PRO450_TCP_OFFSET_POS = (0.127, 0.0, 0.0)
MYCOBOT_PRO450_GRIPPER_OPEN_POS = -1.012
MYCOBOT_PRO450_GRIPPER_CLOSE_POS = 0.0
# Fully closed (0.0) ejects a 4.8 cm DexCube. Intermediate angle keeps pad contact.
MYCOBOT_PRO450_GRIPPER_PINCH_POS = -0.45
MYCOBOT_PRO450_GRIPPER_OPEN_JOINT_POS = {
    "joint2_left_joint": -1.012,
    "joint1_left_joint": -1.012,
    "joint2_right_joint": 1.012,
    "joint1_right_joint": 1.012,
    "joint3_left_joint": 1.012,
    "joint3_right_joint": -1.012,
}

MYCOBOT_PRO450_CFG = MycobotPro450ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    articulation_root_prim_path="/Geometry",
    spawn=UsdFileCfg(
        func=spawn_mycobot_pro450_usd,
        usd_path=MYCOBOT_PRO450_USD_PATH,
        activate_contact_sensors=False,
        variants={"Gripper": "base", "Physics": "physx"},
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=1.5,
            restitution=0.0,
            friction_combine_mode="max",
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.70,
            "joint3": -2.35,
            "joint4": 0.0,
            "joint5": 1.57,
            "joint6": 0.0,
            **MYCOBOT_PRO450_GRIPPER_OPEN_JOINT_POS,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=MYCOBOT_PRO450_ARM_JOINT_NAMES,
            effort_limit_sim=60.0,
            stiffness=80.0,
            damping=12.0,
            armature=1e-3,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=MYCOBOT_PRO450_GRIPPER_JOINT_NAMES,
            effort_limit_sim=80.0,
            stiffness=2000.0,
            damping=100.0,
        ),
        "gripper_mimic": ImplicitActuatorCfg(
            joint_names_expr=MYCOBOT_PRO450_GRIPPER_MIMIC_JOINT_NAMES,
            effort_limit_sim=0.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

MYCOBOT_PRO450_HIGH_PD_CFG = MYCOBOT_PRO450_CFG.copy()
MYCOBOT_PRO450_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
MYCOBOT_PRO450_HIGH_PD_CFG.actuators["arm"].stiffness = 400.0
MYCOBOT_PRO450_HIGH_PD_CFG.actuators["arm"].damping = 80.0
