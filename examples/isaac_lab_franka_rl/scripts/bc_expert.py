"""Scripted Franka stacking expert shared by collect_bc_franka.py and diag_bc_franka.py: expert_stack.py's phase
machine (unchanged logic) with a configurable pick-and-place list, plus the snapshot phase inference and the
"seated" check. Pure numpy, importable without the Kit app."""

import numpy as np

CUBE_H = 0.0468
HOVER = 0.10
KP = 1.0
AMAX = 0.05
OPEN, CLOSE = 1.0, -1.0
BASE_IK_SCALE = 0.5  # the base IK-Rel task's arm-action scale the expert's gains were tuned for


class Expert:
    """expert_stack.py's phase machine, unchanged, with a configurable task list (one pick-and-place per stage)."""

    def __init__(self, tasks, phase="above"):
        self.tasks = tasks
        self.task_idx = 0
        self.phase = phase
        self.phase_steps = 0
        self.done = False

    def set_phase(self, phase):
        self.phase = phase
        self.phase_steps = 0

    def act(self, eef, cubes, finger):
        self.phase_steps += 1
        if self.done:
            return self._to(eef, eef + np.array([0, 0, 0.0]), OPEN)
        obj_name, dest_name = self.tasks[self.task_idx]
        obj, dest = cubes[obj_name], cubes[dest_name]
        grip = OPEN
        target = eef.copy()
        xy_err_obj = np.linalg.norm((obj - eef)[:2])
        holding = np.linalg.norm(obj - eef) < 0.03 and finger < 0.035

        if self.phase == "above":
            target = obj + np.array([0, 0, HOVER])
            if xy_err_obj < 0.006 and abs(target[2] - eef[2]) < 0.015:
                self.set_phase("descend")
        elif self.phase == "descend":
            target = obj + np.array([0, 0, 0.0])
            if xy_err_obj > 0.02:
                self.set_phase("above")
            elif np.linalg.norm(target - eef) < 0.008 or self.phase_steps > 60:
                self.set_phase("grasp")
            else:
                return self._to(eef, target, grip, z_max=0.02)
        elif self.phase == "grasp":
            target = obj.copy()
            grip = CLOSE
            if self.phase_steps > 12:
                self.set_phase("lift")
        elif self.phase == "lift":
            grip = CLOSE
            target = np.array([eef[0], eef[1], dest[2] + CUBE_H + HOVER])
            if not holding and self.phase_steps > 5:
                self.set_phase("above")
            elif abs(target[2] - eef[2]) < 0.015:
                self.set_phase("move")
        elif self.phase == "move":
            grip = CLOSE
            target = dest + np.array([0, 0, CUBE_H + HOVER])
            if not holding:
                self.set_phase("above")
            elif np.linalg.norm((target - eef)[:2]) < 0.006 and abs(target[2] - eef[2]) < 0.015:
                self.set_phase("lower")
        elif self.phase == "lower":
            grip = CLOSE
            target = dest + np.array([0, 0, CUBE_H + 0.004])
            if not holding:
                self.set_phase("above")
            elif np.linalg.norm((dest - eef)[:2]) > 0.02:
                self.set_phase("move")
            elif np.linalg.norm(target - eef) < 0.006 or self.phase_steps > 60:
                self.set_phase("release")
            else:
                return self._to(eef, target, grip, z_max=0.02)
        elif self.phase == "release":
            grip = OPEN
            target = eef.copy()
            if self.phase_steps > 10:
                self.set_phase("retreat")
        elif self.phase == "retreat":
            grip = OPEN
            target = np.array([eef[0], eef[1], dest[2] + CUBE_H + HOVER])
            if abs(target[2] - eef[2]) < 0.02:
                self.task_idx += 1
                if self.task_idx >= len(self.tasks):
                    self.done = True
                else:
                    self.set_phase("above")
        return self._to(eef, target, grip)

    @staticmethod
    def _to(eef, target, grip, z_max=AMAX):
        pos = np.clip(KP * (target - eef), -AMAX, AMAX)
        pos[2] = np.clip(pos[2], -z_max, z_max)
        return np.array([*pos, 0.0, 0.0, 0.0, grip], dtype=np.float32)


def infer_phase(eef, cubes, finger, obj_name, dest_name):
    """Phase of the expert's pick-and-place at a snapshot start state (stage 2: snapshots cover above..lower)."""
    obj, dest = cubes[obj_name], cubes[dest_name]
    holding = np.linalg.norm(obj - eef) < 0.03 and finger < 0.035
    hover_z = dest[2] + CUBE_H + HOVER
    if holding:
        if np.linalg.norm((obj - dest)[:2]) < 0.02 and eef[2] < hover_z - 0.015:
            return "lower"
        if eef[2] < hover_z - 0.015:
            return "lift"
        return "move"
    seated = abs(obj[2] - dest[2] - CUBE_H) < 0.006 and np.linalg.norm((obj - dest)[:2]) < 0.015
    if seated:
        return "release"
    if np.linalg.norm(obj - eef) < 0.012:
        return "grasp"
    if np.linalg.norm((obj - eef)[:2]) < 0.006 and obj[2] - 0.005 < eef[2] < obj[2] + HOVER:
        return "descend"
    return "above"


def seated(top, bottom):
    return (np.abs(top[:, 2] - bottom[:, 2] - CUBE_H) < 0.006) & (np.linalg.norm(top[:, :2] - bottom[:, :2], axis=1) < 0.015)
