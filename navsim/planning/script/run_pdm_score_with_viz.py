"""
Script to run PDM score evaluation and save visualization footage.
This script generates ego camera + BEV map visualizations for specified scenarios.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.visualization.plots import plot_ego_camera_with_bev, frame_plot_to_gif, frame_plot_to_mp4

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def generate_visualizations(
    cfg: DictConfig,
    output_viz_dir: Path,
    max_viz_scenarios: int = 10,
    viz_tokens: Optional[List[str]] = None,
    fps: int = 10,
    output_format: str = "mp4",
) -> None:
    """
    Generate visualization footage for scenarios.

    :param cfg: hydra configuration
    :param output_viz_dir: directory to save visualizations
    :param max_viz_scenarios: maximum number of scenarios to visualize
    :param viz_tokens: specific tokens to visualize (if None, uses first N scenarios)
    :param fps: frames per second for video output
    :param output_format: output format ('mp4' or 'gif')
    """
    output_viz_dir.mkdir(parents=True, exist_ok=True)

    # Initialize agent
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    # Load scene loader with camera sensors (load cameras for all frames)
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    # Create sensor config that loads sensors for all history frames
    sensor_frame_indices = list(range(scene_filter.num_history_frames))
    sensor_config = SensorConfig.build_all_sensors(include=sensor_frame_indices)
    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    # Determine which tokens to visualize
    available_tokens = list(set(scene_loader.tokens_stage_one) & set(metric_cache_loader.tokens))

    # Filter to only tokens that have sensor data available
    logger.info(f"Filtering {len(available_tokens)} candidate scenes for sensor data availability...")
    tokens_with_sensors = []
    checked_count = 0
    for token in available_tokens:
        checked_count += 1
        if checked_count % 50 == 0:
            logger.info(f"Checked {checked_count}/{len(available_tokens)} scenes, found {len(tokens_with_sensors)} valid so far...")
        try:
            # Try to load the scene to verify sensor data exists
            scene_test = scene_loader.get_scene_from_token(token)
            # Count frames with valid camera data
            valid_frames = sum(1 for f in scene_test.frames if f.cameras.cam_f0.image is not None)
            if valid_frames >= 10:  # At least 0.5 seconds of footage
                tokens_with_sensors.append(token)
                logger.info(f"Found valid scene {token} with {valid_frames} frames of camera data")
                if len(tokens_with_sensors) >= max_viz_scenarios:  # Get enough candidates (was * 2)
                    break
        except Exception as e:
            logger.debug(f"Failed to load scene {token}: {e}")
            continue

    if viz_tokens is not None:
        tokens_to_viz = [t for t in viz_tokens if t in tokens_with_sensors]
    else:
        tokens_to_viz = tokens_with_sensors[:max_viz_scenarios]

    logger.info(f"Found {len(tokens_with_sensors)} scenarios with sensor data available")
    logger.info(f"Generating visualizations for {len(tokens_to_viz)} scenarios...")

    for idx, token in enumerate(tqdm(tokens_to_viz, desc="Generating visualizations")):
        try:
            # Load scene and compute trajectory
            scene = scene_loader.get_scene_from_token(token)
            agent_input = scene_loader.get_agent_input_from_token(token)

            if agent.requires_scene:
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)

            # Generate visualization with trajectory overlay
            def plot_frame_with_trajectory(scene_obj, frame_idx):
                return plot_ego_camera_with_bev(scene_obj, frame_idx, trajectory=trajectory)

            # Generate video only for frames that have valid camera data
            frame_indices = []
            for idx in range(len(scene.frames)):
                if scene.frames[idx].cameras.cam_f0.image is not None:
                    frame_indices.append(idx)

            # Skip this scene if we don't have enough frames with camera data
            if len(frame_indices) < 10:  # Require at least 0.5 seconds of footage
                logger.warning(f"Skipping token {token}: only {len(frame_indices)} frames with camera data")
                continue

            output_file = output_viz_dir / f"{token}.{output_format}"
            logger.info(f"Generating video with {len(frame_indices)} frames for token {token}")

            if output_format == "mp4":
                frame_plot_to_mp4(
                    file_name=str(output_file),
                    callable_frame_plot=plot_frame_with_trajectory,
                    scene=scene,
                    frame_indices=frame_indices,
                    fps=fps,
                )
            else:  # gif
                frame_plot_to_gif(
                    file_name=str(output_file),
                    callable_frame_plot=plot_frame_with_trajectory,
                    scene=scene,
                    frame_indices=frame_indices,
                    duration=int(1000 / fps),  # Convert fps to ms per frame
                )

            logger.info(f"Saved visualization {idx + 1}/{len(tokens_to_viz)}: {output_file}")

        except Exception as e:
            logger.warning(f"Failed to generate visualization for token {token}: {e}")
            continue


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for visualization generation.
    :param cfg: omegaconf dictionary
    """
    # Get visualization parameters from config or use defaults
    output_viz_dir = Path(cfg.get("viz_output_dir", "./visualizations"))
    max_viz_scenarios = cfg.get("max_viz_scenarios", 10)
    viz_tokens = cfg.get("viz_tokens", None)
    fps = cfg.get("viz_fps", 10)
    output_format = cfg.get("viz_format", "mp4")

    logger.info(f"Saving visualizations to: {output_viz_dir}")
    logger.info(f"Max scenarios to visualize: {max_viz_scenarios}")
    logger.info(f"Output format: {output_format} @ {fps} FPS")

    generate_visualizations(
        cfg=cfg,
        output_viz_dir=output_viz_dir,
        max_viz_scenarios=max_viz_scenarios,
        viz_tokens=viz_tokens,
        fps=fps,
        output_format=output_format,
    )

    logger.info(f"Visualization generation complete. Files saved to: {output_viz_dir}")


if __name__ == "__main__":
    main()
