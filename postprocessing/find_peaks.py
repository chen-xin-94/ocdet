import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from scipy.spatial import cKDTree, distance
from warnings import warn
import numpy as np
import scipy.ndimage as ndi


class CentroidCounterpointModule(nn.Module):
    """
    My implementation of https://doi.org/10.1016/j.jocs.2022.101760,
    proved to be slower and worse than our postprocessing below
    """

    def __init__(self, kernel_size=10, threshold=0.65, stride=1):
        super(CentroidCounterpointModule, self).__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.stride = stride

        # Calculate pad to keep the same size
        p = (kernel_size[0] - stride) // 2
        if kernel_size[0] % 2 == 0:
            padding_4 = (p, p + 1, p, p + 1)
        else:
            padding_4 = (p, p, p, p)

        self.pad = partial(F.pad, pad=padding_4)

    def _get_map(self, localization_map):
        # Step 1: Apply maximum filter to emphasize centroids
        max_filtered = F.max_pool2d(
            self.pad(localization_map), kernel_size=self.kernel_size, stride=self.stride
        )

        # Step 2: Compare max_filtered with the original localization map to find local maxima
        maxima_map = (max_filtered == localization_map).float()

        # Step 3: Create a mask to suppress irrelevant regions
        mask = (localization_map > 0).float()

        # Step 4: Erode the mask to remove noise
        eroded_mask = F.max_pool2d(
            self.pad(mask), kernel_size=self.kernel_size, stride=self.stride
        )

        # Step 5: Apply exclusive disjunction (XOR) operation between the eroded mask and maxima_map
        centroid_map = torch.logical_xor(maxima_map.bool(), (1 - eroded_mask).bool())

        # Step 6: Suppress centroids that don't meet the threshold in the localization map
        centroid_map = centroid_map & (localization_map >= self.threshold)

        return centroid_map

    def _get_coordinates(self, localization_map):
        centroid_map = self._get_map(localization_map)
        return torch.nonzero(centroid_map.squeeze())

    def forward(self, localization_map):
        return self._get_coordinates(localization_map)


## The following code is our postprocessing, modified from scikit-image


def _ensure_spacing(coord, spacing, p_norm, max_out):
    """Returns a subset of coord where a minimum spacing is guaranteed.

    Parameters
    ----------
    coord : ndarray
        The coordinates of the considered points.
    spacing : float
        the maximum allowed spacing between the points.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.
    max_out: int
        If not None, at most the first ``max_out`` candidates are
        returned.

    Returns
    -------
    output : ndarray
        A subset of coord where a minimum spacing is guaranteed.

    """

    # Use KDtree to find the peaks that are too close to each other
    tree = cKDTree(coord)

    indices = tree.query_ball_point(coord, r=spacing, p=p_norm)
    rejected_peaks_indices = set()
    naccepted = 0
    for idx, candidates in enumerate(indices):
        if idx not in rejected_peaks_indices:
            # keep current point and the points at exactly spacing from it
            candidates.remove(idx)
            dist = distance.cdist(
                [coord[idx]], coord[candidates], distance.minkowski, p=p_norm
            ).reshape(-1)
            candidates = [c for c, d in zip(candidates, dist) if d < spacing]

            # candidates.remove(keep)
            rejected_peaks_indices.update(candidates)
            naccepted += 1
            if max_out is not None and naccepted >= max_out:
                break

    # Remove the peaks that are too close to each other
    output = np.delete(coord, tuple(rejected_peaks_indices), axis=0)
    if max_out is not None:
        output = output[:max_out]

    return output


def ensure_spacing(
    coords,
    spacing=1,
    p_norm=np.inf,
    min_split_size=50,
    max_out=None,
    *,
    max_split_size=2000,
):
    """Returns a subset of coord where a minimum spacing is guaranteed.

    Parameters
    ----------
    coords : array_like
        The coordinates of the considered points.
    spacing : float
        the maximum allowed spacing between the points.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.
    min_split_size : int
        Minimum split size used to process ``coords`` by batch to save
        memory. If None, the memory saving strategy is not applied.
    max_out : int
        If not None, only the first ``max_out`` candidates are returned.
    max_split_size : int
        Maximum split size used to process ``coords`` by batch to save
        memory. This number was decided by profiling with a large number
        of points. Too small a number results in too much looping in
        Python instead of C, slowing down the process, while too large
        a number results in large memory allocations, slowdowns, and,
        potentially, in the process being killed -- see gh-6010. See
        benchmark results `here
        <https://github.com/scikit-image/scikit-image/pull/6035#discussion_r751518691>`_.

    Returns
    -------
    output : array_like
        A subset of coord where a minimum spacing is guaranteed.

    """

    output = coords
    if len(coords):
        coords = np.atleast_2d(coords)
        if min_split_size is None:
            batch_list = [coords]
        else:
            coord_count = len(coords)
            split_idx = [min_split_size]
            split_size = min_split_size
            while coord_count - split_idx[-1] > max_split_size:
                split_size *= 2
                split_idx.append(split_idx[-1] + min(split_size, max_split_size))
            batch_list = np.array_split(coords, split_idx)

        output = np.zeros((0, coords.shape[1]), dtype=coords.dtype)
        for batch in batch_list:
            output = _ensure_spacing(
                np.vstack([output, batch]), spacing, p_norm, max_out
            )
            if max_out is not None and len(output) >= max_out:
                break

    return output


def _get_high_intensity_peaks(
    image, mask, num_peaks, min_distance, p_norm, ensure_min_distance=True
):
    """
    Return the highest intensity peak coordinates.
    """
    # get coordinates of peaks
    coord = np.nonzero(mask)
    intensities = image[coord]
    # Highest peak first
    idx_maxsort = np.argsort(-intensities, kind="stable")
    coord = np.transpose(coord)[idx_maxsort]

    if np.isfinite(num_peaks):
        max_out = int(num_peaks)
    else:
        max_out = None

    if ensure_min_distance:
        coord = ensure_spacing(
            coord, spacing=min_distance, p_norm=p_norm, max_out=max_out
        )

    if len(coord) > num_peaks:
        coord = coord[:num_peaks]

    return coord


def _get_peak_mask(image, footprint, threshold, mask=None):
    """
    Return the mask containing all peak candidates above thresholds.
    """
    if footprint.size == 1 or image.size == 1:
        return image > threshold

    image_max = ndi.maximum_filter(image, footprint=footprint, mode="nearest")

    out = image == image_max

    # no peak for a trivial image
    image_is_trivial = np.all(out) if mask is None else np.all(out[mask])
    if image_is_trivial:
        out[:] = False
        if mask is not None:
            # isolated pixels in masked area are returned as peaks
            isolated_px = np.logical_xor(mask, ndi.binary_opening(mask))
            out[isolated_px] = True

    out &= image > threshold
    return out


def _exclude_border(label, border_width):
    """Set label border values to 0."""
    # zero out label borders
    for i, width in enumerate(border_width):
        if width == 0:
            continue
        label[(slice(None),) * i + (slice(None, width),)] = 0
        label[(slice(None),) * i + (slice(-width, None),)] = 0
    return label


def _get_threshold(image, threshold_abs, threshold_rel):
    """Return the threshold value according to an absolute and a relative
    value.

    """
    threshold = threshold_abs if threshold_abs is not None else image.min()

    if threshold_rel is not None:
        threshold = max(threshold, threshold_rel * image.max())

    return threshold


def _get_excluded_border_width(image, min_distance, exclude_border):
    """Return border_width values relative to a min_distance if requested."""

    if isinstance(exclude_border, bool):
        border_width = (min_distance if exclude_border else 0,) * image.ndim
    elif isinstance(exclude_border, int):
        if exclude_border < 0:
            raise ValueError("`exclude_border` cannot be a negative value")
        border_width = (exclude_border,) * image.ndim
    elif isinstance(exclude_border, tuple):
        if len(exclude_border) != image.ndim:
            raise ValueError(
                "`exclude_border` should have the same length as the "
                "dimensionality of the image."
            )
        for exclude in exclude_border:
            if not isinstance(exclude, int):
                raise ValueError(
                    "`exclude_border`, when expressed as a tuple, must only "
                    "contain ints."
                )
            if exclude < 0:
                raise ValueError("`exclude_border` can not be a negative value")
        border_width = exclude_border
    else:
        raise TypeError(
            "`exclude_border` must be bool, int, or tuple with the same "
            "length as the dimensionality of the image."
        )

    return border_width


def peak_local_max(
    image,
    min_distance=1,
    threshold_abs=None,
    threshold_rel=None,
    exclude_border=True,
    num_peaks=np.inf,
    footprint=None,
    labels=None,
    num_peaks_per_label=np.inf,
    p_norm=np.inf,
    ensure_min_distance=True,
):
    """Find peaks in an image as coordinate list.

    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).

    If both `threshold_abs` and `threshold_rel` are provided, the maximum
    of the two is chosen as the minimum intensity threshold of peaks.

    .. versionchanged:: 0.18
        Prior to version 0.18, peaks of the same height within a radius of
        `min_distance` were all returned, but this could cause unexpected
        behaviour. From 0.18 onwards, an arbitrary peak within the region is
        returned. See issue gh-2592.

    Parameters
    ----------
    image : ndarray
        Input image.
    min_distance : int, optional
        The minimal allowed distance separating peaks. To find the
        maximum number of peaks, use `min_distance=1`.
    threshold_abs : float or None, optional
        Minimum intensity of peaks. By default, the absolute threshold is
        the minimum intensity of the image.
    threshold_rel : float or None, optional
        Minimum intensity of peaks, calculated as
        ``max(image) * threshold_rel``.
    exclude_border : int, tuple of ints, or bool, optional
        If positive integer, `exclude_border` excludes peaks from within
        `exclude_border`-pixels of the border of the image.
        If tuple of non-negative ints, the length of the tuple must match the
        input array's dimensionality.  Each element of the tuple will exclude
        peaks from within `exclude_border`-pixels of the border of the image
        along that dimension.
        If True, takes the `min_distance` parameter as value.
        If zero or False, peaks are identified regardless of their distance
        from the border.
    num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` peaks based on highest peak intensity.
    footprint : ndarray of bools, optional
        If provided, `footprint == 1` represents the local region within which
        to search for peaks at every point in `image`.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.

    Returns
    -------
    output : ndarray
        The coordinates of the peaks.

    """
    if (footprint is None or footprint.size == 1) and min_distance < 1:
        warn(
            "When min_distance < 1, peak_local_max acts as finding "
            "image > max(threshold_abs, threshold_rel * max(image)).",
            RuntimeWarning,
            stacklevel=2,
        )

    border_width = _get_excluded_border_width(image, min_distance, exclude_border)

    threshold = _get_threshold(image, threshold_abs, threshold_rel)

    if footprint is None:
        size = 2 * min_distance + 1
        footprint = np.ones((size,) * image.ndim, dtype=bool)
    else:
        footprint = np.asarray(footprint)

    # Non maximum filter
    mask = _get_peak_mask(image, footprint, threshold)

    mask = _exclude_border(mask, border_width)

    # Select highest intensities (num_peaks)
    coordinates = _get_high_intensity_peaks(
        image, mask, num_peaks, min_distance, p_norm, ensure_min_distance
    )

    return coordinates
