"""Shared lighting for the white 450 arm in Reach and Lift scenes."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg


def configure_arm_lighting(scene):
    scene.light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.95, 1.0), intensity=200.0),
    )
    scene.key_light = AssetBaseCfg(
        prim_path="/World/key_light",
        # XYZW: Euler XYZ (-30, -35, 0) degrees; down from -X/+Y.
        init_state=AssetBaseCfg.InitialStateCfg(rot=(-0.24684011, -0.2904595, -0.07782839, 0.92121983)),
        spawn=sim_utils.DistantLightCfg(color=(1.0, 0.95, 0.9), intensity=650.0, angle=5.0),
    )
    scene.fill_light = AssetBaseCfg(
        prim_path="/World/fill_light",
        # Fill shadows from the default viewer's +X/+Y side.
        init_state=AssetBaseCfg.InitialStateCfg(rot=(-0.24684011, 0.2904595, 0.07782839, 0.92121983)),
        spawn=sim_utils.DistantLightCfg(color=(1.0, 1.0, 1.0), intensity=350.0, angle=20.0),
    )
