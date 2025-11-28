from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.carla_garage.carla_garage_config import CarlaGarageConfig
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

from navsim.agents.transfuser.transfuser_callback import TransfuserCallback
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.agents.transfuser.transfuser_loss import transfuser_loss
from navsim.agents.transfuser.transfuser_model import TransfuserModel

from navsim.agents.carla_garage.carla_garage_model import LidarCenterNet

class CarlaGarageAgent(AbstractAgent):
    """Constant velocity baseline agent."""

    requires_scene = False

    def __init__(
        self,
        config: CarlaGarageConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
        strict_checkpoint_loading: bool = True,
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        :param trajectory_sampling: trajectory sampling specification
        :param strict_checkpoint_loading: whether to use strict loading when loading model checkpoint
        """
        super().__init__(trajectory_sampling)

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        self._strict_checkpoint_loading = strict_checkpoint_loading
        self._carla_garage_model = LidarCenterNet(config)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            checkpoint = torch.load(self._checkpoint_path)
        else:
            checkpoint = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict: Dict[str, Any] = checkpoint["state_dict"]
        else:
            state_dict: Dict[str, Any] = checkpoint

        # Check if keys need "_carla_garage_model." prefix
        sample_keys = list(state_dict.keys())[:5]
        has_model_prefix = sample_keys and any(key.startswith("_carla_garage_model.") for key in sample_keys)

        if not has_model_prefix:
            # Raw checkpoint without prefix - add _carla_garage_model prefix
            state_dict = {f"_carla_garage_model.{key}": value for key, value in state_dict.items()}

        # Use strict=False for modified models with extra/missing keys
        self.load_state_dict(state_dict, strict=self._strict_checkpoint_loading)


    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        history_steps = [3]
        return SensorConfig(
            cam_f0=history_steps,
            cam_l0=history_steps,
            cam_l1=False,
            cam_l2=False,
            cam_r0=history_steps,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=history_steps if not self._config.latent else False,
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    # TODO: use self._transfuser_model(features)
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        # Unpack features dictionary to match LidarCenterNet.forward() signature
        rgb = features["camera_feature"].unsqueeze(0) if features["camera_feature"].dim() == 3 else features["camera_feature"]
        lidar_bev = features["lidar_feature"].unsqueeze(0) if features["lidar_feature"].dim() == 3 else features["lidar_feature"]
        status = features["status_feature"]

        # status_feature is concatenation of [driving_command (1D), ego_velocity (2D), ego_acceleration (2D)]
        # Total: 5D
        if status.dim() == 1:
            driving_cmd = status[0].long()  # scalar command index
            ego_velocity_2d = status[1:3]  # 2D velocity vector
        else:
            driving_cmd = status[:, 0].long()  # (batch,) command indices
            ego_velocity_2d = status[:, 1:3]  # (batch, 2) velocity vectors

        # Convert 2D velocity to scalar speed
        ego_vel = torch.norm(ego_velocity_2d, dim=-1, keepdim=True)  # (batch, 1) or (1,)
        if ego_vel.dim() == 0:
            ego_vel = ego_vel.unsqueeze(0).unsqueeze(0)
        elif ego_vel.dim() == 1:
            ego_vel = ego_vel.unsqueeze(0)

        # One-hot encode the driving command (assuming 6 classes: VOID, LEFT, RIGHT, STRAIGHT, LANEFOLLOW, CHANGELANELEFT, CHANGELANERIGHT)
        # But nuPlan uses 4 classes, so let's use 6 to match CARLA
        num_commands = 6
        command = torch.nn.functional.one_hot(driving_cmd, num_classes=num_commands).float()
        if command.dim() == 1:
            command = command.unsqueeze(0)

        # TODO: Compute actual target_point from route/waypoints
        # For now, use a dummy target point (straight ahead)
        batch_size = rgb.shape[0]
        target_point = torch.zeros((batch_size, 2), device=rgb.device, dtype=rgb.dtype)
        target_point[:, 0] = 10.0  # 10 meters ahead in x direction

        # Model returns: (pred_wp, pred_target_speed, pred_checkpoint, pred_semantic, pred_bev_semantic,
        #                 pred_depth, pred_bounding_box, attention_weights, pred_wp_1, selected_path)
        model_outputs = self._carla_garage_model(rgb, lidar_bev, target_point, ego_vel, command)

        # Extract predictions
        pred_wp, pred_target_speed, pred_checkpoint, pred_semantic, pred_bev_semantic, \
            pred_depth, pred_bounding_box, attention_weights, pred_wp_1, selected_path = model_outputs

        # Convert to expected dictionary format
        # Since use_wp_gru=False, pred_wp is None. Use pred_checkpoint as trajectory.
        trajectory = pred_checkpoint if pred_checkpoint is not None else pred_wp

        # Ensure trajectory matches the expected number of poses from trajectory_sampling
        # Model predicts predict_checkpoint_len=10, but trajectory_sampling expects num_poses=8
        num_expected_poses = self._trajectory_sampling.num_poses
        if trajectory is not None and trajectory.shape[1] != num_expected_poses:
            # Slice to match expected number of poses
            trajectory = trajectory[:, :num_expected_poses, :]

        # Add heading dimension to trajectory (model outputs only x, y)
        # Compute heading from consecutive waypoint positions
        if trajectory is not None and trajectory.shape[-1] == 2:
            batch_size = trajectory.shape[0]
            num_poses = trajectory.shape[1]

            # Compute heading from position differences
            # For each waypoint, compute heading as arctan2(dy, dx) to next waypoint
            headings = torch.zeros((batch_size, num_poses, 1), device=trajectory.device, dtype=trajectory.dtype)

            # Compute headings from consecutive position differences
            for i in range(num_poses - 1):
                dx = trajectory[:, i + 1, 0] - trajectory[:, i, 0]
                dy = trajectory[:, i + 1, 1] - trajectory[:, i, 1]
                headings[:, i, 0] = torch.atan2(dy, dx)

            # For the last waypoint, use the same heading as the second-to-last
            headings[:, -1, 0] = headings[:, -2, 0]

            # Concatenate (x, y, heading)
            trajectory = torch.cat([trajectory, headings], dim=-1)

        predictions = {
            "trajectory": trajectory,
            "target_speed": pred_target_speed,
        }

        return predictions

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return transfuser_loss(targets, predictions, self._config)

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return torch.optim.Adam(self._carla_garage_model.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [TransfuserCallback(self._config)]