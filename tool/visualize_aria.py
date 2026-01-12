# Copyright (c) Meta Platforms, Inc. and affiliates.
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

import argparse
import json
import os
from typing import List

import numpy as np
import rerun as rr
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.calibration import CameraCalibration
from projectaria_tools.core.mps import MpsDataPathsProvider
from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
    filter_points_from_count,
    get_nearest_pose,
)
from projectaria_tools.core.sensor_data import SensorData, SensorDataType, TimeDomain
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.utils.rerun_helpers import ToTransform3D
from tqdm import tqdm


def log_covision_object(object_file: str, T_world_object: SE3) -> None:
    rr.log(
        "world/object",
        rr.Asset3D(path=object_file, media_type=rr.MediaType.GLB),
        ToTransform3D(T_world_object, False),
        timeless=True,
    )


def log_point_clouds(points_file) -> None:
    points_data = mps.read_global_point_cloud(points_file)
    points_data = filter_points_from_confidence(points_data)

    points_data_down_sampled = filter_points_from_count(points_data, 500_000)
    point_positions = [it.position_world for it in points_data_down_sampled]

    rr.log(
        "world/points",
        rr.Points3D(point_positions, radii=0.006),
        timeless=True,
    )


def log_rgb_camera_calibration(
    rgb_camera_calibration: CameraCalibration,
    down_sampling_factor: int,
) -> None:
    rr.log(
        "world/rgb_camera",
        rr.Pinhole(
            resolution=[
                rgb_camera_calibration.get_image_size()[0] / down_sampling_factor,
                rgb_camera_calibration.get_image_size()[1] / down_sampling_factor,
            ],
            focal_length=float(
                rgb_camera_calibration.get_focal_lengths()[0] / down_sampling_factor
            ),
        ),
        timeless=True,
    )


def log_camera_pose(
    trajectory_data: List[mps.ClosedLoopTrajectoryPose],
    device_time_ns: int,
    rgb_camera_calibration: CameraCalibration,
) -> None:
    if trajectory_data:
        pose_info = get_nearest_pose(trajectory_data, device_time_ns)
        if pose_info:
            T_world_device = pose_info.transform_world_device
            T_device_camera = rgb_camera_calibration.get_transform_device_camera()
            rr.log(
                "world/rgb_camera",
                ToTransform3D(T_world_device @ T_device_camera, False),
            )


def log_RGB_image(
    data: SensorData,
    down_sampling_factor: int,
    jpeg_quality: int,
) -> None:
    if data.sensor_data_type() == SensorDataType.IMAGE:
        img = data.image_data_and_record()[0].to_numpy_array()
        if down_sampling_factor > 1:
            img = img[::down_sampling_factor, ::down_sampling_factor]

        rr.log(
            "world/rgb_camera",
            rr.Image(img).compress(jpeg_quality=jpeg_quality),
        )


def main(args):
    rr.init("DTC Data Viewer", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)

    object_pose_file = os.path.join(args.sequence_folder, "object_pose.json")

    with open(object_pose_file) as f:
        config = json.load(f)

    T_world_object = SE3.from_matrix3x4(
        np.array(config["mesh"]["T_world_object"])[0:3, :]
    )

    mps_folder = os.path.join(args.sequence_folder, "mps")

    mps_data_paths_provider = MpsDataPathsProvider(mps_folder)
    mps_data_paths = mps_data_paths_provider.get_data_paths()

    metadata_file = os.path.join(args.sequence_folder, "metadata.json")
    with open(metadata_file) as f:
        metadata = json.load(f)

    model_name = metadata["model_name"][0]

    object_file = os.path.join(args.model_folder, model_name, "3d-asset.glb")

    log_point_clouds(mps_data_paths.slam.semidense_points)
    log_covision_object(object_file, T_world_object)

    trajectory_data = mps.read_closed_loop_trajectory(
        mps_data_paths.slam.closed_loop_trajectory
    )

    online_calibs = mps.read_online_calibration(mps_data_paths.slam.online_calibrations)
    for camera_calib in online_calibs[-1].camera_calibs:
        if camera_calib.get_label() == "camera-rgb":
            last_rgb_camera_calib = camera_calib

    log_rgb_camera_calibration(last_rgb_camera_calib, args.down_sampling_factor)

    vrs_file = os.path.join(args.sequence_folder, "video.vrs")

    provider = data_provider.create_vrs_data_provider(vrs_file)
    rgb_stream_id = StreamId("214-1")
    deliver_option = provider.get_default_deliver_queued_options()
    deliver_option.deactivate_stream_all()
    deliver_option.activate_stream(rgb_stream_id)
    rgb_frame_count = provider.get_num_data(rgb_stream_id)

    progress_bar = tqdm(total=rgb_frame_count)

    for data in provider.deliver_queued_sensor_data(deliver_option):
        device_time_ns = data.get_time_ns(TimeDomain.DEVICE_TIME) - int(
            last_rgb_camera_calib.get_time_offset_sec_device_camera() * 1e9
        )
        rr.set_time_nanos("device_time", device_time_ns)
        rr.set_time_sequence("timestamp", device_time_ns)
        progress_bar.update(1)

        log_camera_pose(
            trajectory_data,
            device_time_ns,
            last_rgb_camera_calib,
        )

        log_RGB_image(
            data,
            args.down_sampling_factor,
            args.jpeg_quality,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_folder",
        type=str,
        help="path to sequence folder",
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        help="path to model root folder",
    )
    parser.add_argument(
        "--down_sampling_factor", type=int, default=4, help=argparse.SUPPRESS
    )
    parser.add_argument("--jpeg_quality", type=int, default=75, help=argparse.SUPPRESS)

    args = parser.parse_args()

    main(args)
