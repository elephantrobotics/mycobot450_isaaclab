# Robot assets

Isaac Lab prefers `mycobot_pro450_lift.usda` in this folder. That overlay
references `~/mycobot450_isaacsim/USD/sim_450_f100/450_f100.usda` and removes
the F100 `ArticulationRoot`, so arm and gripper become one PhysX tree.

Fallback order if the overlay is missing:

1. `MYCOBOT_PRO450_USD_PATH`
2. `~/mycobot450_isaacsim/USD/sim_450_f100/450_f100.usda`

Keep payload files next to `450_f100.usda`. Do not copy a lone `.usda` without `payloads/`.
