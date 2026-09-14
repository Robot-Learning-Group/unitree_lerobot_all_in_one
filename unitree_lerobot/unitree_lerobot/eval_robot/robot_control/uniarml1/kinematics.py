# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


class RobotKinematics:
    """Robot kinematics using placo library for forward and inverse kinematics."""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ):
        """
        Initialize placo-based kinematics solver.

        Args:
            urdf_path (str): Path to the robot URDF file
            target_frame_name (str): Name of the end-effector frame in the URDF
            joint_names (list[str] | None): List of joint names to use for the kinematics solver
        """
        try:
            import placo  # type: ignore[import-not-found] # C++ library with Python bindings, no type stubs available. TODO: Create stub file or request upstream typing support.
        except ImportError as e:
            raise ImportError(
                "placo is required for RobotKinematics. "
                "Please install the optional dependencies of `kinematics` in the package."
            ) from e

        # Ensure urdf_path is a string (placo does not accept PosixPath)
        urdf_path_str = str(urdf_path)
        self.robot = placo.RobotWrapper(urdf_path_str)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # Fix the base

        self.target_frame_name = target_frame_name

        # Set joint names
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names

        # Initialize frame task for IK
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))
        # self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4), placo.FrameType.body)

        # Joint preference task: make certain joints more inclined to move
        # Format: {joint_name: weight}, higher weight means more inclined to move the joint
        self.joint_preferences = {}

        # Smoothing regularization task (optional)
        self.smoothing_task = None
        self.smoothing_weight = 0.0

    def set_joint_preference(self, joint_name: str, weight: float):
        """
        Set joint preference weight to make IK solver prefer using that joint.

        Args:
            joint_name: Joint name
            weight: Preference weight (higher weight means more inclined to move the joint)
        """
        self.joint_preferences[joint_name] = weight

    def set_joint_preferences(self, preferences: dict[str, float]):
        """
        Batch set joint preference weights.

        Args:
            preferences: {joint_name: weight} dictionary
        """
        self.joint_preferences.update(preferences)

    def set_smoothing_weight(self, weight: float):
        """
        Set smoothing regularization weight.

        Args:
            weight: Smoothing weight (higher means smoother but slower response)
                    Typical value: 0.01-0.1
        """
        self.smoothing_weight = weight

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for given joint configuration given the target frame name in the constructor.

        Args:
            joint_pos_deg: Joint positions in degrees (numpy array)

        Returns:
            4x4 transformation matrix of the end-effector pose
        """

        # Convert degrees to radians
        joint_pos_rad = np.deg2rad(joint_pos_deg[: len(self.joint_names)])

        # Update joint positions in placo robot
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])

        # Update kinematics
        self.robot.update_kinematics()

        # Get the transformation matrix
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
        joint_regularization_weight: float = 0.01,
        smoothing_weight: float | None = None,
    ) -> np.ndarray:
        """
        Compute inverse kinematics using placo solver.

        Args:
            current_joint_pos: Current joint positions in degrees (used as initial guess)
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_weight: Weight for position constraint in IK
            orientation_weight: Weight for orientation constraint in IK, set to 0.0 to only constrain position
            joint_regularization_weight: Regularization weight to keep joints without preference at current position
            smoothing_weight: Smoothing regularization weight (optional), higher values result in smoother motion

        Returns:
            Joint positions in degrees that achieve the desired end-effector pose
        """

        # Convert current joint positions to radians for initial guess
        current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])

        # Set current joint positions as initial guess
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, current_joint_rad[i])

        # Update the target pose for the frame task
        self.tip_frame.T_world_frame = desired_ee_pose

        # Configure the task based on position_only flag
        self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)

        # Add joint regularization task: constrain all joints with regularization
        # Joints with high preference weight have weak regularization (easy to move), joints with low/no preference have strong regularization (hard to move)
        joints_task = self.solver.add_joints_task()
        for i, joint_name in enumerate(self.joint_names):
            # Set regularization target for all joints (current position)
            joints_task.set_joint(joint_name, current_joint_rad[i])

        # Compute regularization weight for each joint
        # Higher preference weight means lower regularization weight (easier to deviate from current position)
        regularization_weights = {}
        for joint_name in self.joint_names:
            preference = self.joint_preferences.get(joint_name, 0.0)
            # Regularization weight = base weight / (preference weight + 1)
            # Preference=0 → regularization=base weight (strongest constraint)
            # Preference=0.8 → regularization=base weight/1.8 (weaker constraint)
            regularization_weights[joint_name] = joint_regularization_weight / (preference + 1.0)

        # Set regularization weight (using average weight)
        avg_weight = float(np.mean(list(regularization_weights.values())))
        joints_task.configure("joints_regularization", "soft", avg_weight)

        # Add smoothing regularization task (optional)
        smoothing_task = None
        if smoothing_weight is not None and smoothing_weight > 0:
            try:
                import placo
                smoothing_task = placo.RegularizationTask()
                smoothing_task.configure("smoothing", "soft", smoothing_weight)
                self.solver.add_task(smoothing_task)
            except Exception as e:
                print(f"Warning: Failed to add smoothing task: {e}")

        # Solve IK
        self.solver.solve(True)
        self.robot.update_kinematics()

        # Clean up tasks
        self.solver.remove_task(joints_task)
        if smoothing_task is not None:
            self.solver.remove_task(smoothing_task)

        # Extract joint positions
        joint_pos_rad = []
        for joint_name in self.joint_names:
            joint = self.robot.get_joint(joint_name)
            joint_pos_rad.append(joint)

        # Convert back to degrees
        joint_pos_deg = np.rad2deg(joint_pos_rad)

        # Preserve gripper position if present in current_joint_pos
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[: len(self.joint_names)] = joint_pos_deg
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        else:
            return joint_pos_deg
