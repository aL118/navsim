from dataclasses import dataclass
from typing import Tuple

import numpy as np
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer


@dataclass
class CarlaGarageConfig:
    """Config for Carla Garage model."""

    image_architecture: str = "regnety_032"
    lidar_architecture: str = "regnety_032"

    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    lidar_seq_len: int = 1

    camera_width: int = 1024
    camera_height: int = 512
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    img_vert_anchors: int = 384 // 32  # 12 - matches checkpoint (crop_image=True, cropped_height=384)
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp = 4
    n_layer = 2
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    gpt_linear_layer_init_mean = 0.0
    gpt_linear_layer_init_std = 0.02
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic: bool = True  # Matches checkpoint
    use_depth: bool = True  # Matches checkpoint
    add_features = True

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    num_bounding_boxes: int = 30

    # loss weights
    trajectory_weight: float = 10.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 10.0

    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),
        4: ("box", [TrackedObjectType.CZONE_SIGN, TrackedObjectType.BARRIER,
                    TrackedObjectType.TRAFFIC_CONE, TrackedObjectType.GENERIC_OBJECT]),
        5: ("box", [TrackedObjectType.VEHICLE]),
        6: ("box", [TrackedObjectType.PEDESTRIAN]),
    }

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height // 2
    bev_pixel_size: float = 0.25

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    # ===== Carla Garage specific parameters =====
    backbone: str = "transFuser"

    # Lateral PID controller
    lateral_k_p: float = 3.118357247806046
    lateral_k_d: float = 1.3782508892109167
    lateral_k_i: float = 0.6406067986034124
    lateral_speed_scale: float = 0.9755321901954155
    lateral_speed_offset: float = 1.9152884533402488
    lateral_default_lookahead: float = 24
    lateral_speed_threshold: float = 23.150102938235136
    lateral_n: int = 6

    # Turn/Speed PID controllers
    turn_kp: float = 1.25
    turn_ki: float = 0.75
    turn_kd: float = 0.3
    turn_n: int = 20
    speed_kp: float = 1.75
    speed_ki: float = 1.0
    speed_kd: float = 2.0
    speed_n: int = 20

    # Longitudinal control
    longitudinal_max_acceleration: float = 1.89
    longitudinal_params: tuple = (1.1990342347353184, -0.8057602384167799, 1.710818710950062, 0.921890257450335, 1.556497522998393, -0.7013479734904027, 1.031266635497984)

    # Target speed / weights
    target_speeds: tuple = (0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0)  # 8 classes to match checkpoint
    target_speed_weights: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    use_speed_weights: bool = False
    semantic_weights: tuple = (1.0,) * 7  # 7 classes to match checkpoint
    bev_semantic_weights: tuple = (1.0,) * 11  # 11 classes to match checkpoint
    num_semantic_classes: int = 7  # Matches checkpoint
    num_bev_semantic_classes: int = 11  # Matches checkpoint

    # GRU / decoder settings
    gru_input_size: int = 256  # Matches checkpoint
    gru_hidden_size: int = 64
    use_wp_gru: bool = False  # Matches checkpoint
    use_controller_input_prediction: bool = True
    predict_checkpoint_len: int = 10
    pred_len: int = 4
    wp_dilation: int = 1
    multi_wp_output: bool = False
    learn_origin: bool = True
    num_decoder_heads: int = 8
    num_transformer_decoder_layers: int = 6

    # Extra sensors
    use_velocity: bool = True
    use_discrete_command: bool = True
    extra_sensor_channels: int = 128
    use_tp: bool = True
    two_tp_input: bool = False
    tp_attention: bool = False

    # Loss settings
    use_focal_loss: bool = False
    focal_loss_gamma: float = 2.0
    use_label_smoothing: bool = False
    label_smoothing_alpha: float = 0.1

    # Deconv / detection settings
    deconv_channel_num_0: int = 128
    deconv_channel_num_1: int = 64
    deconv_channel_num_2: int = 32
    deconv_scale_factor_0: int = 4  # Matches checkpoint
    deconv_scale_factor_1: int = 8  # Matches checkpoint
    num_bb_classes: int = 5  # Matches checkpoint (includes emergency vehicle class)
    bb_confidence_threshold: float = 0.3
    bb_input_channel: int = 64
    top_k_center_keypoints: int = 100
    center_net_max_pooling_kernel: int = 3
    num_dir_bins: int = 12

    # BEV settings
    bev_features_chanels: int = 64
    bev_classes_list: tuple = (1, 2, 3, 4, 5, 6)
    bev_grid_height_downsample_factor: int = 1

    # Aim / brake / clip settings
    aim_distance_fast = 3.0
    aim_distance_slow = 2.25
    aim_distance_threshold: float = 5.5
    brake_speed: float = 0.4
    brake_ratio: float = 1.1
    clip_delta: float = 1.0
    clip_throttle: float = 1.0

    # Plant settings
    plant_max_speed_pred: float = 20.0
    plant_precision_pos: float = 10.0
    plant_precision_angle: float = 0.02
    plant_precision_speed: float = 0.05

    # Camera/lidar settings
    camera_fov: float = 110
    camera_pos: tuple = (-1.5, 0.0, 2.0)
    camera_rot_0: tuple = (0.0, 0.0, 0.0)
    lidar_pos: tuple = (0.0, 0.0, 2.5)
    lidar_rot: tuple = (0.0, 0.0, -90.0)
    carla_fps: int = 20
    ego_extent_x: float = 2.4508416652679443
    ego_extent_y: float = 1.0641621351242065

    # Cropping
    crop_image: bool = True  # Matches checkpoint
    crop_bev: bool = False
    crop_bev_height_only_from_behind: bool = False
    cropped_height: int = 384  # Matches checkpoint
    cropped_width: int = 1024

    # BEV projection bounds
    min_x: float = -32.0
    max_x: float = 32.0
    min_y: float = -32.0
    max_y: float = 32.0
    min_z_projection: float = -10
    max_z_projection: float = 14

    # Misc
    seq_len: int = 1
    normalize_imagenet: bool = True
    data_save_freq: int = 5
    debug: bool = False
    input_path_to_target_speed_network: bool = False

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
