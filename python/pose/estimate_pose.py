"""
Pose predictions in Python.

Caffe must be available on the Pythonpath for this to work. The methods can
be imported and used directly, or the command line interface can be used. In
the latter case, adjust the log-level to your needs. The maximum image size
for one prediction can be adjusted with the variable _MAX_SIZE so that it
still fits in GPU memory, all larger images are split in sufficiently small
parts.

Authors: Christoph Lassner, based on the MATLAB implementation by Eldar
  Insafutdinov.
"""

import numpy as _np
import scipy as _scipy
import logging as _logging
import caffe as _caffe


_LOGGER = _logging.getLogger(__name__)

# Constants.
# Image mean to use.
_MEAN = _np.array([104., 117., 123.])
# Scale factor for the CNN offset predictions.
_LOCREF_SCALE_MUL = _np.sqrt(53.)
# Maximum size of one tile to process (to limit the required GPU memory).
_MAX_SIZE = 700

_STRIDE = 8.

# CNN model store.
_MODEL = None


def estimate_pose(image, model_def, model_bin, scales=None):  # pylint: disable=too-many-locals
    """
    Get the estimated pose for an image.

    Uses the CNN pose estimator from "Deepcut: Joint Subset Partition and
    Labeling for Multi Person Pose Estimation" (Pishchulin et al., 2016),
    "DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation
    Model" (Insafutdinov et al., 2016).


    Parameters
    ==========

    :param image: np.array(3D).
      The image in height X width X BGR format.

    :param scales: list(float) or None.
      Scales on which to apply the estimator. The pose with the best confidence
      on average will be returned.

    Returns
    =======

    :param pose: np.array(2D).
      The pose in 5x14 layout. The first axis is along per-joint information,
      the second the joints. Information is:
        1. position x,
        2. position y,
        3. CNN confidence,
        4. CNN offset vector x,
        5. CNN offset vector y.
    """
    global _MODEL  # pylint: disable=global-statement
    if scales is None:
        scales = [1.]
    if _MODEL is None:
        _LOGGER.info("Loading pose model...")
        _MODEL = _caffe.Net(model_def, model_bin, _caffe.TEST)
        _LOGGER.info("Done!")
    _LOGGER.debug("Processing image...")
    im_orig = image.copy()
    _LOGGER.debug("Image shape: %s.", im_orig.shape)
    best_pose = None
    highest_confidence = 0.
    for scale_factor in scales:
        _LOGGER.debug("Scale %f...", scale_factor)
        image = im_orig.copy()
        # Create input for multiples of net input/output changes.
        im_bg_width = int(_np.ceil(
            float(image.shape[1]) * scale_factor / _STRIDE) * _STRIDE)
        im_bg_height = int(_np.ceil(
            float(image.shape[0]) * scale_factor / _STRIDE) * _STRIDE)
        pad_size = 64
        im_bot_pixels = image[-1:, :, :]
        im_bot = _np.tile(im_bot_pixels, (pad_size, 1, 1))
        image = _np.vstack((image, im_bot))
        im_right_pixels = image[:, -1:, :]
        im_right = _np.tile(im_right_pixels, (1, pad_size, 1))
        image = _np.hstack((image, im_right))
        image = _scipy.misc.imresize(image, scale_factor, interp='bilinear')
        image = image.astype('float32') - _MEAN

        net_input = _np.zeros((im_bg_height, im_bg_width, 3), dtype='float32')
        net_input[:min(net_input.shape[0], image.shape[0]),
                  :min(net_input.shape[1], image.shape[1]), :] =\
            image[:min(net_input.shape[0], image.shape[0]),
                  :min(net_input.shape[1], image.shape[1]), :]

        _LOGGER.debug("Input shape: %d, %d.",
                      net_input.shape[0], net_input.shape[1])
        unary_maps, locreg_pred = _process_image_tiled(_MODEL, net_input, _STRIDE)

        """
        import matplotlib.pyplot as plt
        plt.figure()
        for map_idx in range(unary_maps.shape[2]):
            plt.imshow(unary_maps[:, :, map_idx], interpolation='none')
            plt.imsave('map_%d.png' % map_idx,
                       unary_maps[:, :, map_idx])
            plt.show()
        """

        pose = _pose_from_mats(unary_maps, locreg_pred, scale=scale_factor)

        minconf = _np.min(pose[2, :])
        if minconf > highest_confidence:
            _LOGGER.debug("New best scale detected: %f (scale), " +
                          "%f (min confidence).", scale_factor, minconf)
            highest_confidence = minconf
            best_pose = pose
    _LOGGER.debug("Pose estimated.")
    return best_pose


def _pose_from_mats(scoremat, offmat, scale):
    """Combine scoremat and offsets to the final pose."""
    pose = []
    for joint_idx in range(14):
        maxloc = _np.unravel_index(_np.argmax(scoremat[:, :, joint_idx]),
                                   scoremat[:, :, joint_idx].shape)
        offset = _np.array(offmat[maxloc][joint_idx])[::-1]
        pos_f8 = (_np.array(maxloc).astype('float') * _STRIDE + 0.5 * _STRIDE +
                  offset * _LOCREF_SCALE_MUL)
        pose.append(_np.hstack((pos_f8[::-1] / scale,
                                [scoremat[maxloc][joint_idx]],
                                offset * _LOCREF_SCALE_MUL / scale)))
    return _np.array(pose).T


def _get_num_tiles(length, max_size, rf):
    """Get the number of tiles required to cover the entire image."""
    if length <= max_size:
        return 1
    k = 0
    while True:
        new_size = (max_size - rf) * 2 + (max_size - 2*rf) * k
        if new_size > length:
            break
        k += 1
    return 2 + k


# pylint: disable=too-many-locals
def _process_image_tiled(model, net_input, stride):
    """Get the CNN results for the tiled image."""
    rf = 224  # Standard receptive field size.
    cut_off = rf / stride

    num_tiles_x = _get_num_tiles(net_input.shape[1], _MAX_SIZE, rf)
    num_tiles_y = _get_num_tiles(net_input.shape[0], _MAX_SIZE, rf)
    if num_tiles_x > 1 or num_tiles_y > 1:
        _LOGGER.info("Tiling the image into %d, %d (w, h) tiles...",
                     num_tiles_x, num_tiles_y)

    scoremaps = []
    locreg_pred = []
    for j in range(num_tiles_y):
        start_y = j * (_MAX_SIZE - 2*rf)
        if j == num_tiles_y:
            end_y = net_input.shape[0]
        else:
            end_y = start_y + _MAX_SIZE
        scoremaps_line = []
        locreg_pred_line = []
        for i in range(num_tiles_x):
            start_x = i * (_MAX_SIZE - 2*rf)
            if i == num_tiles_x:
                end_x = net_input.shape[1]
            else:
                end_x = start_x + _MAX_SIZE
            input_tile = net_input[start_y:end_y, start_x:end_x, :]
            _LOGGER.debug("Tile info: %d, %d, %d, %d.",
                          start_y, end_y, start_x, end_x)
            scoremaps_tile, locreg_pred_tile = _cnn_process_image(model,
                                                                  input_tile)
            _LOGGER.debug("Tile out shape: %s, %s.",
                          str(scoremaps_tile.shape),
                          str(locreg_pred_tile.shape))
            scoremaps_tile = _cutoff_tile(scoremaps_tile,
                                          num_tiles_x, i, cut_off, True)
            locreg_pred_tile = _cutoff_tile(locreg_pred_tile,
                                            num_tiles_x, i, cut_off, True)
            scoremaps_tile = _cutoff_tile(scoremaps_tile,
                                          num_tiles_y, j, cut_off, False)
            locreg_pred_tile = _cutoff_tile(locreg_pred_tile,
                                            num_tiles_y, j, cut_off, False)
            _LOGGER.debug("Cutoff tile out shape: %s, %s.",
                          str(scoremaps_tile.shape), str(locreg_pred_tile.shape))
            scoremaps_line.append(scoremaps_tile)
            locreg_pred_line.append(locreg_pred_tile)
        scoremaps_line = _np.concatenate(scoremaps_line, axis=1)
        locreg_pred_line = _np.concatenate(locreg_pred_line, axis=1)
        scoremaps_line = _cutoff_tile(scoremaps_line,
                                      num_tiles_y, j, cut_off, False)
        locreg_pred_line = _cutoff_tile(locreg_pred_line,
                                        num_tiles_y, j, cut_off, False)
        _LOGGER.debug("Line tile out shape: %s, %s.",
                      str(scoremaps_line.shape), str(locreg_pred_line.shape))
        scoremaps.append(scoremaps_line)
        locreg_pred.append(locreg_pred_line)
    scoremaps = _np.concatenate(scoremaps, axis=0)
    locreg_pred = _np.concatenate(locreg_pred, axis=0)
    _LOGGER.debug("Final tiled shape: %s, %s.",
                  str(scoremaps.shape), str(locreg_pred.shape))
    return scoremaps[:, :, 0, :], locreg_pred.transpose((0, 1, 3, 2))


def _cnn_process_image(model, net_input):
    """Get the CNN results for a fully prepared image."""
    net_input = net_input.transpose((2, 0, 1))
    model.blobs['data'].reshape(1, 3, net_input.shape[1], net_input.shape[2])
    model.blobs['data'].data[0, ...] = net_input[...]
    model.forward()

    for outp_name in ['loc_pred', 'prob']:
        out_value = model.blobs[outp_name].data
        if out_value.shape[1] == 14:
            feat_prob = out_value.copy().transpose((2, 3, 0, 1))
            continue
        else:
            out_value = out_value.reshape((14, 2,
                                           out_value.shape[2],
                                           out_value.shape[3]))
            out_value = out_value.transpose((2, 3, 1, 0))
            locreg_pred = out_value
    return feat_prob, locreg_pred


def _cutoff_tile(sm, num_tiles, idx, cut_off, is_x):
    """Cut the valid parts of the CNN predictions for a tile."""
    if is_x:
        sm = sm.transpose((1, 0, 2, 3))
    if num_tiles == 1:
        pass
    elif idx == 1:
        sm = sm[:-cut_off, ...]
    elif idx == num_tiles:
        sm = sm[cut_off:, ...]
    else:
        sm = sm[cut_off:-cut_off, ...]
    if is_x:
        sm = sm.transpose((1, 0, 2, 3))
    return sm
