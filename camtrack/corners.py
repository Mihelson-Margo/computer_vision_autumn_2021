#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


class L2DistanceCalculator(object):
    """
    For each given point calculates L2 distance
    between patches around the original and a moved point,
    divided by number of pixels in a window
    """
    def __init__(self, orig_image: np.array, moved_image: np.array,
                 orig_points: np.array, moved_points: np.array,
                 window_size: int):
        self._orig_points = orig_points
        self._moved_points = moved_points
        self._orig_image = orig_image
        self._moved_image = moved_image
        self._window_size = window_size
        self._distance = self.calc_distance()

    @property
    def distance(self) -> np.array:
        return self._distance

    @staticmethod
    def one_patch_ids(window_size: int) -> (np.array, np.array):
        l = -(window_size//2)
        r = window_size//2 + 1
        x_ids = np.repeat(np.arange(l, r).reshape((-1, 1)), window_size, axis=1)
        y_ids = np.repeat(np.arange(l, r).reshape((1, -1)), window_size, axis=0)
        return x_ids, y_ids

    @staticmethod
    def patch_ids_for_points(points: np.array, window_size: int):
        n = points.shape[0]
        x_ids, y_ids = L2DistanceCalculator.one_patch_ids(window_size)
        x_ids_all = np.repeat(np.expand_dims(x_ids, axis=2), n, axis=2)
        y_ids_all = np.repeat(np.expand_dims(y_ids, axis=2), n, axis=2)
        points_x = (points[:, 0]).reshape(n).astype(int)
        points_y = (points[:, 1]).reshape(n).astype(int)
        return x_ids_all + points_x, y_ids_all + points_y

    @staticmethod
    def fix_border_ids(ids: np.array, max_id: int):
        ids[ids < 0] = 0
        ids[ids >= max_id] = max_id-1
        return ids

    def patches_for_points(self, moved: bool) -> np.array:
        if moved:
            image = self._moved_image
            points = self._moved_points
        else:
            image = self._orig_image
            points = self._orig_points

        n, m = image.shape
        x_ids_all, y_ids_all = self.patch_ids_for_points(points, self._window_size)
        x_ind_all = self.fix_border_ids(x_ids_all, n)
        y_ind_all = self.fix_border_ids(y_ids_all, m)
        return image[x_ind_all, y_ind_all]

    def calc_distance(self):
        orig_patches = self.patches_for_points(moved=False)
        moved_patches = self.patches_for_points(moved=True)
        diff = np.abs(orig_patches-moved_patches)**2
        return np.sqrt(np.sum(diff, axis=(0, 1))/self._window_size**2)


def create_mask_by_corners(image, corners, compress_rate=1):
    mask = np.full(image.shape, 255).astype(np.uint8)
    points = (corners.points//compress_rate).astype(int)
    for i in range(points.shape[0]):
        cv2.circle(mask, (points[i, 0], points[i, 1]), corners.sizes[i, 0]//compress_rate, 0, -1)
    return mask


def add_corners(image, feature_params, max_corner_id, corners=None, max_pyramid_lvl=4):
    if corners is not None:
        mask = create_mask_by_corners(image, corners)
    else:
        mask = None
        corners = FrameCorners.empty_frame()

    compress_rate = 1
    small_img = image
    while compress_rate <= (1 << max_pyramid_lvl):
        new_p = cv2.goodFeaturesToTrack(small_img, mask=mask, **feature_params)
        if new_p is not None:
            new_p = new_p.reshape((-1, 2)) * compress_rate
            n = new_p.shape[0]
            new_sizes = np.full(n, feature_params['blockSize']*compress_rate).reshape((-1, 1))
            new_ids = np.arange(max_corner_id, max_corner_id + n)
            max_corner_id += n
            corners.add_points(new_ids, new_p, new_sizes, np.ones((n, 1)), np.zeros((n, 1)))

        compress_rate *= 2
        small_img = cv2.pyrDown(small_img)
        mask = create_mask_by_corners(small_img, corners, compress_rate)

    return corners, max_corner_id


def lk_params_for_pyramid_lvl(lk_params, compress_rate):
    lk_p_params = lk_params.copy()
    lk_p_params['minEigThreshold'] = lk_params['minEigThreshold'] * compress_rate
    return lk_p_params


def track_corners(image_0, image_1, corners, lk_params, max_pyramid_lvl=4):
    compress_rate = 1
    small_img0 = image_0
    small_img1 = image_1
    new_corners = FrameCorners.empty_frame()

    while compress_rate <= (1 << max_pyramid_lvl):
        mask = (corners.sizes == lk_params['winSize'][0]//2 * compress_rate)
        curr_ids = corners.ids[mask].reshape((-1, 1))
        curr_points = corners.points[np.hstack((mask, mask)) == 1].reshape((-1, 2))
        curr_sizes = corners.sizes[mask].reshape((-1, 1))

        lk_p_params = lk_params_for_pyramid_lvl(lk_params, compress_rate)

        new_p, st, err = cv2.calcOpticalFlowPyrLK(small_img0, small_img1, curr_points/compress_rate,
                                                  cv2.OPTFLOW_LK_GET_MIN_EIGENVALS, **lk_p_params)

        #p2, s2, err2 = cv2.calcOpticalFlowPyrLK(small_img0, small_img1, curr_points/compress_rate,
        #                                        None, **lk_params)

        #err2 = err2[s2==1].reshape(-1)

        #print(f"here:\n err2 ~= {err2[0:5]},\n our ~= {l1_dist[0:5]},\n{np.sum(l1_dist != err2)}")

        if new_p is not None:
            new_p = new_p[np.hstack((st, st)) == 1].reshape((-1, 2))
            l2_dist = L2DistanceCalculator(small_img0, small_img1,
                                           curr_points[np.hstack((st, st)) == 1].reshape((-1, 2)), new_p,
                                           lk_params['winSize'][0]).distance
            new_corners.add_points(curr_ids[st == 1], new_p * compress_rate,
                                   curr_sizes[st == 1], err[st == 1], l2_dist)

        compress_rate *= 2
        small_img0 = cv2.pyrDown(small_img0)
        small_img1 = cv2.pyrDown(small_img1)

    return new_corners


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.1, #0.2
                          minDistance=7,
                          blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.01),
                     minEigThreshold=1e-4)  # 1e-3

    image_0 = (frame_sequence[0] * 255.0).astype(np.uint8)
    max_corner_id = 0
    corners, max_corner_id = add_corners(image_0, feature_params, max_corner_id)
    builder.set_corners_at_frame(0, corners)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        image_1 = (image_1 * 255.0).astype(np.uint8)
        corners = track_corners(image_0, image_1, corners, lk_params)
        
        if frame % 5 == 0:
            corners, max_corner_id = add_corners(image_1, feature_params, max_corner_id, corners)

        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.
    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
