# src/controller/pick_place.py

from __future__ import annotations

from typing import List, Optional

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.storage.native import get_assets_root_path

from src.robots.franka import FrankaRobot


class FrankaPickPlace:
    # Simple Franka pick-and-place controller using a small state machine

    def __init__(self, events_dt: Optional[List[int]] = None):
        self.cube: Optional[RigidPrim] = None
        self.robot: Optional[FrankaRobot] = None

        # Phase durations in simulation steps
        self.events_dt = events_dt or [60, 40, 20, 40, 80, 20, 20]

        self._event = 0
        self._step = 0

        self.cube_initial_position: Optional[np.ndarray] = None
        self.cube_initial_orientation: Optional[np.ndarray] = None
        self.cube_size: Optional[np.ndarray] = None
        self.target_position: Optional[np.ndarray] = None
        self.offset: Optional[np.ndarray] = None

    def setup_scene(
        self,
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        # Set task defaults
        self.cube_size = (
            cube_size if cube_size is not None else np.array([0.0515, 0.0515, 0.0515], dtype=np.float32)
        )
        self.cube_initial_position = (
            cube_initial_position
            if cube_initial_position is not None
            else np.array([0.5, 0.0, 0.0258], dtype=np.float32)
        )
        self.cube_initial_orientation = (
            cube_initial_orientation
            if cube_initial_orientation is not None
            else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        self.target_position = (
            target_position if target_position is not None else np.array([-0.3, -0.3, 0.12], dtype=np.float32)
        )
        self.offset = offset if offset is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.target_position = self.target_position + self.offset

        # Create a fresh stage with a basic light rig
        stage_utils.create_new_stage(template="sunlight")

        # Add a ground environment for physics
        stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
            path="/World/ground",
        )

        # Spawn the robot using our local wrapper
        self.robot = FrankaRobot(robot_path="/World/robot", create_robot=True)

        # Spawn a cube and make it collide and simulate as a rigid body
        cube_shape = Cube(
            paths="/World/Cube",
            positions=self.cube_initial_position,
            orientations=self.cube_initial_orientation,
            sizes=[1.0],
            scales=self.cube_size,
            reset_xform_op_properties=True,
        )
        GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)
        self.cube = RigidPrim(paths=cube_shape.paths)

    def forward(self, ik_method: str = "damped-least-squares") -> bool:
        # Run one tick of the state machine
        if self.is_done():
            return False

        assert self.robot is not None, "Call setup_scene() before forward()"
        assert self.cube is not None, "Call setup_scene() before forward()"

        goal_orientation = self.robot.get_downward_orientation()

        if self._event == 0:
            if self._step == 0:
                print("Phase 0: Move above cube")

            cube_pos = self.cube.get_world_poses()[0].numpy()
            goal_position = np.array([cube_pos[0, 0], cube_pos[0, 1], cube_pos[0, 2] + 0.2], dtype=np.float32)

            self.robot.set_end_effector_pose(goal_position, goal_orientation, ik_method=ik_method)
            self._advance_phase_if_needed(0)

        elif self._event == 1:
            if self._step == 0:
                print("Phase 1: Approach cube")

            cube_pos = self.cube.get_world_poses()[0].numpy()
            goal_position = cube_pos + np.array([0.0, 0.0, 0.1], dtype=np.float32)

            self.robot.set_end_effector_pose(goal_position, goal_orientation, ik_method=ik_method)
            self._advance_phase_if_needed(1)

        elif self._event == 2:
            if self._step == 0:
                print("Phase 2: Close gripper")

            self.robot.close_gripper()
            self._advance_phase_if_needed(2)

        elif self._event == 3:
            if self._step == 0:
                print("Phase 3: Lift")

            _, current_position, _ = self.robot.get_current_state()
            goal_position = current_position + np.array([0.0, 0.0, 0.2], dtype=np.float32)

            self.robot.set_end_effector_pose(goal_position, goal_orientation, ik_method=ik_method)
            self._advance_phase_if_needed(3)

        elif self._event == 4:
            if self._step == 0:
                print("Phase 4: Move to target")

            self.robot.set_end_effector_pose(self.target_position, goal_orientation, ik_method=ik_method)
            self._advance_phase_if_needed(4)

        elif self._event == 5:
            if self._step == 0:
                print("Phase 5: Open gripper")

            self.robot.open_gripper()
            self._advance_phase_if_needed(5)

        elif self._event == 6:
            if self._step == 0:
                print("Phase 6: Retreat up")

            cube_pos = self.cube.get_world_poses()[0].numpy()
            goal_position = cube_pos + np.array([0.0, 0.0, 0.3], dtype=np.float32)

            self.robot.set_end_effector_pose(goal_position, goal_orientation, ik_method=ik_method)
            self._advance_phase_if_needed(6)

        return True

    def _advance_phase_if_needed(self, phase_idx: int) -> None:
        self._step += 1
        if self._step >= self.events_dt[phase_idx]:
            self._event += 1
            self._step = 0

    def is_done(self) -> bool:
        return self._event >= len(self.events_dt)

    def reset(self, cube_position: Optional[np.ndarray] = None, cube_orientation: Optional[np.ndarray] = None) -> None:
        print("Resetting pick-and-place system...")
        self.reset_robot()
        self.reset_cube(position=cube_position, orientation=cube_orientation)
        print("Pick-and-place system reset complete")

    def reset_robot(self) -> None:
        if self.robot is None:
            print("Warning: robot not initialised, cannot reset")
            return

        self.robot.reset_to_default_pose()

        self._event = 0
        self._step = 0

        print("Robot reset to default state")

    def reset_cube(self, position: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None) -> None:
        if self.cube is None:
            print("Warning: cube not initialised, cannot reset")
            return

        reset_position = position if position is not None else self.cube_initial_position
        reset_orientation = orientation if orientation is not None else self.cube_initial_orientation

        assert reset_position is not None
        assert reset_orientation is not None

        self.cube.set_world_poses(
            positions=np.asarray(reset_position, dtype=np.float32).reshape(1, -1),
            orientations=np.asarray(reset_orientation, dtype=np.float32).reshape(1, -1),
        )

        print(f"Cube reset to position: {reset_position}")