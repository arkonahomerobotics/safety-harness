"""Modality config for the Franka cube-stack "new_embodiment" tag.

Reconstructed from checkpoint-5000's own experiment_cfg/conf.yaml (data.modality_configs.
new_embodiment) -- "new_embodiment" is NOT pre-registered in
gr00t/configs/data/embodiment_configs.py (checked: MODALITY_CONFIGS has no "new_embodiment"
key), so the original training run must have loaded a file like this one via
--modality-config-path; it was never committed and is lost, same pattern as every other
Brev-only file tonight. This file registers it the same way
gr00t/configs/data/embodiment_configs.py registers its own built-in tags -- a module-level
assignment into the real MODALITY_CONFIGS dict, picked up because
gr00t/experiment/launch_finetune.py's load_modality_config() just imports this file by path
for its side effect (importlib.import_module after sys.path.append(parent)).

Verified against a converted episode via gr00t.data.dataset.sharded_single_step_dataset.
ShardedSingleStepDataset / LeRobotEpisodeLoader directly (not guessed): these exact
ModalityConfig values load table_cam/wrist_cam video, single_arm(9)/gripper(2) state,
single_arm(6)/gripper(1) action (16 delta indices, action_horizon=16), and
annotation.human.task_description language correctly.

Usage: --modality-config-path /path/to/this/file.py
"""

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

MODALITY_CONFIGS["new_embodiment"] = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["table_cam", "wrist_cam"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key=None,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key=None,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}
