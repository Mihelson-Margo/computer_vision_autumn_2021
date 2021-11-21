#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

import time
from typing import List, Optional, Tuple

import numpy as np
import sortednp as snp
import random
import cv2

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    Correspondences,
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
    SolvePnPParameters,
    eye3x4,
    rodrigues_and_translation_to_view_mat3x4
)


def find_and_add_points3d(point_cloud_builder: PointCloudBuilder,
                          view_mat_1: np.ndarray, view_mat_2: np.ndarray,
                          intrinsic_mat: np.ndarray,
                          corners_1: FrameCorners, corners_2: FrameCorners,
                          max_reproj_error: float = 0.6) -> PointCloudBuilder:
    params = TriangulationParameters(max_reproj_error, 0.1, 0.1)
    correspondence = build_correspondences(corners_1, corners_2)
    points3d, ids, errors = triangulate_correspondences(
        correspondence,
        view_mat_1,
        view_mat_2,
        intrinsic_mat,
        params
    )
    
    n_updated = point_cloud_builder.add_points(ids, points3d, errors)
    print(f'triangulate {ids.shape[0]} points, update {n_updated} of them')
    return point_cloud_builder


def choose_corners_for_PnP(corners: FrameCorners, present_ids: np.ndarray,
                           image_shape: Tuple, min_dist: int) -> np.ndarray:
    w, h = image_shape
    image_mask = np.ones(image_shape)
    points = corners.points[present_ids]
    corners_mask = np.zeros(present_ids.shape, dtype=bool)
    sorting_ids = np.argsort(corners.min_eigenvals.flatten()[present_ids])
    for i in sorting_ids:
        x = int(points[i, 0])
        y = int(points[i, 1])
        if 0 <= x < w and 0 <= y < h and image_mask[x, y]:
            corners_mask[i] = True
            cv2.circle(image_mask, (x, y), min_dist, color=0, thickness=-1)

    return corners_mask


def calc_camera_pose(point_cloud_builder: PointCloudBuilder,
                     corners: FrameCorners, intrinsic_mat: np.ndarray,
                     image_shape: Tuple, huber: bool,
                     max_reproj_error: float = 0.6) -> Tuple[np.array, float]:
    _, (idx_1, idx_2) = snp.intersect(point_cloud_builder.ids.flatten(),
                                      corners.ids.flatten(), indices=True)
    best_ids = choose_corners_for_PnP(corners, idx_2, image_shape, 10)
    points_3d = point_cloud_builder.points[idx_1[best_ids]]
    points_2d = corners.points[idx_2[best_ids]]
    params = SolvePnPParameters(max_reproj_error, 0)
    if points_2d.shape[0] < 5:
        print(f"Too few points to solve PnP")
        return eye3x4(), 0

    view_mat, n_inliers = solve_PnP(points_2d, points_3d, intrinsic_mat,
                                    huber, params)
    print(f"{n_inliers}/{points_2d.shape[0]} inliers")
    return view_mat, n_inliers/points_2d.shape[0]


def check_distance_between_cameras(view_mat_1: np.array, view_mat_2: np.array)\
        -> bool:
    pose_1 = to_camera_center(view_mat_1)
    pose_2 = to_camera_center(view_mat_2)
    return np.linalg.norm(pose_1 - pose_2) > 0.2


def tricky_range(init_pose: int, end: int, step: int):
    """ generate  a, a + s, a+2s, ..., a-s, a-2s, ..."""
    pos = init_pose
    while 0 <= pos < end:
        yield pos
        pos += step
    pos = init_pose - step
    while 0 <= pos < end:
        yield pos
        pos -= step

    return


def find_first_frame(v1, v2, d, n):
    v1, v2 = min(v1, v2), max(v1, v2)
    if v2-v1 >= d:
        d = 0
    if np.abs(v2+d-n) > np.abs(v1-d):
        return max(v1-d, 0), 1
    else:
        return min(v2+d, ((n-1)//10)*10), -1


def frame_by_frame_calc(point_cloud_builder: PointCloudBuilder,
                        corner_storage: CornerStorage, view_mats: np.array,
                        known_views: list, intrinsic_mat: np.ndarray,
                        image_shape: Tuple):
    random.seed(42)
    n_frames = len(corner_storage)
    step = 10
    first_frame, sign = find_first_frame(known_views[0],
                                         known_views[1], 2*step, n_frames)

    for frame in tricky_range(first_frame, n_frames, step*sign):
        print(f"\nFrame = {frame}")
        if frame not in known_views:
            view_mats[frame], _ = calc_camera_pose(point_cloud_builder,
                                                   corner_storage[frame],
                                                   intrinsic_mat, image_shape,
                                                   huber=False)
        if frame > 0:
            prev_frame = frame - step
            point_cloud_builder = find_and_add_points3d(
                point_cloud_builder,
                view_mats[frame],
                view_mats[prev_frame],
                intrinsic_mat,
                corner_storage[frame],
                corner_storage[prev_frame]
            )
        print(f'{point_cloud_builder.points.shape[0]} 3d points')

    for frame in tricky_range(first_frame, n_frames, sign):  # range(n_frames):
        print(f"\nFrame = {frame}")
        if frame not in known_views:
            view_mats[frame], inliers_rate = calc_camera_pose(
                point_cloud_builder,
                corner_storage[frame],
                intrinsic_mat,
                image_shape,
                huber=False
            )
        for _ in range(10):
            frame_2 = random.randint(0, n_frames//step - 1) * step
            if check_distance_between_cameras(view_mats[frame], view_mats[frame_2]):
                print(f"{frame} <-> {frame_2} triangulation: ", end='')
                point_cloud_builder = find_and_add_points3d(
                    point_cloud_builder,
                    view_mats[frame],
                    view_mats[frame_2],
                    intrinsic_mat,
                    corner_storage[frame],
                    corner_storage[frame_2],
                    max_reproj_error=0.6
                )
        print(f'{point_cloud_builder.points.shape[0]} 3d points')
    return view_mats


def verify_position(correspondence: Correspondences, view_mat: np.ndarray,
                    intrinsic_mat: np.ndarray, max_reproj_error: float = 0.6)\
        -> int:
    #params = TriangulationParameters(max_reproj_error, 0.1, 0.1)
    params = TriangulationParameters(10, 0, 0)
    _, ids, _ = triangulate_correspondences(
        correspondence,
        eye3x4(),
        view_mat,
        intrinsic_mat,
        params
    )
    return ids.shape[0]


def find_and_check_view_mat(correspondence: Correspondences,
                                 intrinsic_mat: np.ndarray) \
        -> Tuple[float, np.ndarray]:
    # default parameters everywhere
    essential_mat, inliers_mask = cv2.findEssentialMat(
        correspondence.points_1,
        correspondence.points_2,
        intrinsic_mat,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
        maxIters=1000
        )
    _, homography_mask = cv2.findHomography(
        correspondence.points_1,
        correspondence.points_2,
        method=cv2.RANSAC,
        ransacReprojThreshold=3,
        maxIters=2000,
        confidence=0.995
    )
    n_essential_inliers = np.sum(inliers_mask)
    n_homography_inliers = np.sum(homography_mask)
    #print(f"ess_inl: {n_essential_inliers}, homo_inl: {n_homography_inliers}")
    #if n_homography_inliers/n_essential_inliers > 0.1:
    #    return np.inf, np.array([])

    r1, r2, t_abs = cv2.decomposeEssentialMat(essential_mat)
    max_n_inliers = 0
    view_mat_best = None
    for r in [r1, r2]:
        for t in [t_abs, -t_abs]:
            view_mat = np.hstack((r, t))
            #print(f"view, r, t shapes = {view_mat.shape}, {r.shape}, {t.shape}")
            n_inliers = verify_position(correspondence, view_mat, intrinsic_mat)
            if n_inliers > max_n_inliers:
                max_n_inliers = n_inliers
                view_mat_best = view_mat
    return n_homography_inliers/max_n_inliers, view_mat_best


def find_and_initialize_frames(corner_storage: CornerStorage,
                               intrinsic_mat: np.ndarray)\
        -> Tuple[Tuple[int, Pose], Tuple[int, Pose]]:
    n_frames = len(corner_storage)
    errs = []
    views = []
    #for _ in range(1):
    #    for _ in range(1):
    #        frame_1 = 0
    #        frame_2 = 20
    for frame_1 in range(0, n_frames, 5):
        for frame_2 in range(frame_1+10, n_frames, 5):
            correspondence = build_correspondences(corner_storage[frame_1],
                                                   corner_storage[frame_2])
            if correspondence.ids.shape[0] < 5:
                continue
            #print(f"{frame_1}, {frame_2}, corrs {correspondence.ids.shape[0]}")
            err, view_mat = find_and_check_view_mat(correspondence,
                                                    intrinsic_mat)
            if not np.isfinite(err) or view_mat is None:
                continue

            view1 = (frame_1, view_mat3x4_to_pose(eye3x4()))
            view2 = (frame_2, view_mat3x4_to_pose(view_mat))

            #print(err)
            if err < 0.34:
                return view1, view2
            errs.append(err)
            views.append((view1, view2))
        # delete this
        if n_frames > 100:
            break
    best_id = np.argmin(np.array(errs))
    return views[best_id]


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = find_and_initialize_frames(corner_storage,
                                                                intrinsic_mat)

    # TODO: implement
    image_shape = (rgb_sequence[0].shape[1], rgb_sequence[0].shape[0])
    frame_count = len(corner_storage)
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    known_id1, known_id2 = known_view_1[0], known_view_2[0]
    point_cloud_builder = PointCloudBuilder()
    point_cloud_builder = find_and_add_points3d(
        point_cloud_builder,
        pose_to_view_mat3x4(known_view_1[1]),
        pose_to_view_mat3x4(known_view_2[1]),
        intrinsic_mat,
        corner_storage[known_id1],
        corner_storage[known_id2]
    )

    view_mats = frame_by_frame_calc(
        point_cloud_builder,
        corner_storage,
        view_mats,
        [known_id1, known_id2],
        intrinsic_mat,
        image_shape
    )

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
