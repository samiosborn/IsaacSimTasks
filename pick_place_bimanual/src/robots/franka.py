# src/robots/franka.py

from __future__ import annotations

from typing import Optional, Dict

import numpy as np
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.storage.native import get_assets_root_path


def _quat_conj(q: np.ndarray) -> np.ndarray:
    # Quaternion conjugate
    # q shape: (N, 4) in [w, x, y, z]
    qc = q.copy()
    qc[:, 1:] *= -1.0
    return qc


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Quaternion multiplication a*b
    # a, b shape: (N, 4) in [w, x, y, z]
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )


class FrankaRobot(Articulation):
    # Franka robot wrapper: spawn, IK to end-effector pose, gripper, reset

    def __init__(
        self,
        robot_path: str = "/World/robot",
        create_robot: bool = True,
        end_effector_link: Optional[RigidPrim] = None,
    ):
        if create_robot:
            # Load the Franka Panda from the USD asset
            stage_utils.add_reference_to_stage(
                usd_path=get_assets_root_path()
                + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
                path=robot_path,
                variants=[("Gripper", "AlternateFinger"), ("Mesh", "Performance")],
            )

        # Initialise the Articulation base class for the robot at robot_path
        super().__init__(robot_path)

        # End-effector link (panda_hand)
        self.end_effector_link: RigidPrim = end_effector_link or RigidPrim(f"{robot_path}/panda_hand")
        self.end_effector_link_index: int = self.get_link_indices("panda_hand").list()[0]

        # Default pose (open gripper)
        self.default_q = np.array(
            [[0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.04, 0.04]],
            dtype=np.float32,
        )

        if create_robot:
            # Set default state in the simulator so reset uses this pose
            self.set_default_state(dof_positions=self.default_q[0].tolist())

        # Gripper targets (finger joint positions)
        self.gripper_open = np.array([[0.04, 0.04]], dtype=np.float32)
        self.gripper_closed = np.array([[0.0, 0.0]], dtype=np.float32)

    def get_downward_orientation(self) -> np.ndarray:
        # Standard downward-facing end-effector orientation [w, x, y, z]
        return np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    def reset_to_default_pose(self) -> None:
        # Reset robot joints and targets to default (open gripper)
        self.set_dof_positions(self.default_q)
        self.set_dof_position_targets(self.default_q)

    def open_gripper(self) -> None:
        # Open the gripper
        self.set_dof_position_targets(self.gripper_open, dof_indices=[7, 8])

    def close_gripper(self) -> None:
        # Close the gripper
        self.set_dof_position_targets(self.gripper_closed, dof_indices=[7, 8])

    def get_current_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Returns:
        #   - dof_positions: (N, 9)
        #   - ee_pos: (N, 3)
        #   - ee_quat: (N, 4) in [w, x, y, z]
        q = self.get_dof_positions().numpy()
        p, r = self.end_effector_link.get_world_poses()
        return q, p.numpy(), r.numpy()

    def differential_inverse_kinematics(
        self,
        jacobian_end_effector: np.ndarray,               # (N, 6, 7)
        current_position: np.ndarray,                    # (N, 3)
        current_orientation: np.ndarray,                 # (N, 4)
        goal_position: np.ndarray,                       # (N, 3)
        goal_orientation: Optional[np.ndarray] = None,   # (N, 4)
        method: str = "damped-least-squares",
        method_cfg: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        # Compute delta_q for arm joints (N, 7) using differential IK

        if method_cfg is None:
            method_cfg = {"scale": 1.0, "damping": 0.05, "min_singular_value": 1e-5}

        scale = float(method_cfg.get("scale", 1.0))
        damping = float(method_cfg.get("damping", 0.05))
        min_sv = float(method_cfg.get("min_singular_value", 1e-5))

        if goal_orientation is None:
            goal_orientation = current_orientation

        # Quaternion error q_err = q_goal * conj(q_cur)
        q_err = _quat_mul(goal_orientation, _quat_conj(current_orientation))

        # Sign trick to keep the shortest rotation
        sign_w = np.where(q_err[:, [0]] >= 0.0, 1.0, -1.0)
        rot_err = q_err[:, 1:] * sign_w

        # 6D error as position + orientation vector part
        e = np.concatenate([goal_position - current_position, rot_err], axis=1)
        e = e[:, :, None]

        J = jacobian_end_effector
        JT = np.swapaxes(J, 1, 2)

        if method == "transpose":
            return (scale * (JT @ e)).squeeze(-1)

        if method == "pseudoinverse":
            J_pinv = np.linalg.pinv(J)
            return (scale * (J_pinv @ e)).squeeze(-1)

        if method == "singular-value-decomposition":
            U, S, Vh = np.linalg.svd(J, full_matrices=False)
            inv_s = np.where(S > min_sv, 1.0 / S, 0.0)

            D = np.zeros((J.shape[0], 6, 6), dtype=J.dtype)
            idx = np.arange(6)
            D[:, idx, idx] = inv_s

            J_pinv = np.swapaxes(Vh, 1, 2) @ D @ np.swapaxes(U, 1, 2)
            return (scale * (J_pinv @ e)).squeeze(-1)

        if method == "damped-least-squares":
            I = np.eye(6, dtype=J.dtype)[None, :, :]
            A = (J @ JT) + (damping**2) * I
            return (scale * (JT @ np.linalg.inv(A) @ e)).squeeze(-1)

        raise ValueError(f"Invalid IK method: {method}")

    def set_end_effector_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        ik_method: str = "damped-least-squares",
    ) -> None:
        # One IK step towards a target pose

        q_cur, p_cur, r_cur = self.get_current_state()

        # Ensure batch shapes (N, 3) and (N, 4)
        if position.ndim == 1:
            position = position.reshape(1, 3)
        if orientation.ndim == 1:
            orientation = orientation.reshape(1, 4)

        jac = self.get_jacobian_matrices().numpy()
        J_ee = jac[:, self.end_effector_link_index - 1, :, :7]

        dq = self.differential_inverse_kinematics(
            jacobian_end_effector=J_ee,
            current_position=p_cur,
            current_orientation=r_cur,
            goal_position=position,
            goal_orientation=orientation,
            method=ik_method,
        )

        q_tgt = q_cur[:, :7] + dq
        self.set_dof_position_targets(q_tgt, dof_indices=list(range(7)))