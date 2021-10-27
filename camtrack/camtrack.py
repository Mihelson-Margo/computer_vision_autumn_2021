#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import sortednp as snp
import random

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    to_camera_center,
    solve_PnP,
    SolvePnPParameters
)


def find_and_add_points3d(point_cloud_builder: PointCloudBuilder,
                          view_mat_1: np.ndarray, view_mat_2: np.ndarray,
                          intrinsic_mat: np.ndarray,
                          corners_1: FrameCorners, corners_2: FrameCorners,
                          max_reproj_error: float = 0.6) -> PointCloudBuilder:
    params = TriangulationParameters(max_reproj_error, 0, 0.05)
    correspondence = build_correspondences(corners_1, corners_2)
    points3d, ids, median_cos = triangulate_correspondences(correspondence, view_mat_1, view_mat_2, intrinsic_mat,
                                                            params)
    
    point_cloud_builder.add_points(ids, points3d)
    return point_cloud_builder


def calc_camera_pose(point_cloud_builder: PointCloudBuilder, corners: FrameCorners,
                     intrinsic_mat: np.ndarray, max_reproj_error: float = 0.5):
    _, (idx_1, idx_2) = snp.intersect(point_cloud_builder.ids.flatten(), corners.ids.flatten(),
                                      indices=True)
    points_3d = point_cloud_builder.points[idx_1]
    points_2d = corners.points[idx_2]
    params = SolvePnPParameters(max_reproj_error, 0)
    view_mat, inliers_mask = solve_PnP(points_2d, points_3d, intrinsic_mat, params)
    return view_mat


def check_distance_between_cameras(view_mat_1: np.array, view_mat_2: np.array) -> bool:
    pose_1 = to_camera_center(view_mat_1)
    pose_2 = to_camera_center(view_mat_2)
    return np.linalg.norm(pose_1 - pose_2) > 0.2


def frame_by_frame_calc(point_cloud_builder: PointCloudBuilder, corner_storage: CornerStorage,
                        view_mats: np.array, known_views: list,
                        intrinsic_mat: np.ndarray):
    random.seed(42)
    n_frames = len(corner_storage)
    step = int(np.sqrt(n_frames))

    for frame in range(0, n_frames, step):
        if frame in known_views:
            continue
        view_mats[frame] = calc_camera_pose(point_cloud_builder, corner_storage[frame], intrinsic_mat)
        if frame > 0:
            prev_frame = frame - step
            point_cloud_builder = find_and_add_points3d(point_cloud_builder,
                                                        view_mats[frame], view_mats[prev_frame],
                                                        intrinsic_mat,
                                                        corner_storage[frame], corner_storage[prev_frame])

    for frame in range(n_frames):
        if frame not in known_views:
            view_mats[frame] = calc_camera_pose(point_cloud_builder, corner_storage[frame], intrinsic_mat)
            for _ in range(2):
                frame_2 = random.randint(0, n_frames//step - 1) * step
                if check_distance_between_cameras(view_mats[frame], view_mats[frame_2]):
                    point_cloud_builder = find_and_add_points3d(point_cloud_builder,
                                                                view_mats[frame], view_mats[frame_2],
                                                                intrinsic_mat,
                                                                corner_storage[frame], corner_storage[frame_2],
                                                                max_reproj_error=0.5)

    return view_mats


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    # TODO: implement
    frame_count = len(corner_storage)
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    # corners_0 = corner_storage[0]
    # point_cloud_builder = PointCloudBuilder(corners_0.ids[:1],
    #                                        np.zeros((1, 3)))

    known_id1, known_id2 = known_view_1[0], known_view_2[0]
    point_cloud_builder = PointCloudBuilder()
    point_cloud_builder = find_and_add_points3d(point_cloud_builder,
                                                pose_to_view_mat3x4(known_view_1[1]),
                                                pose_to_view_mat3x4(known_view_2[1]),
                                                intrinsic_mat,
                                                corner_storage[known_id1], corner_storage[known_id2])

    # best_id = (known_id1 + known_id2)//2
    # view_mats[best_id] = calc_camera_pose(point_cloud_builder, corner_storage[best_id], intrinsic_mat)
    view_mats = frame_by_frame_calc(point_cloud_builder, corner_storage,
                                    view_mats, [known_id1, known_id2], intrinsic_mat)

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
