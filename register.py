import math

from scipy import interpolate
import numpy as np
import scipy.optimize
import skimage.transform
import utils


# TODO - consider gaussian smoothing
# TODO - other loss options than SSD - cross-correlation, mutual information.
# TODO - the transformation can also be made part of lambda?

def im_register_scale(im_fixed, im_moving,
                      pad_size=None, options={}):
    """For registering only by re-scaling."""
    if pad_size is None:
        pad_size = (im_moving.shape[0] * 2, im_moving.shape[1] * 2)

    orig_size = (im_moving.shape[0], im_moving.shape[1])

    im_fixed = utils.resize_image(im_fixed, pad_size)
    im_moving = utils.resize_image(im_moving, pad_size) / 255

    im_f_shape = im_fixed.shape
    im_m_shape = im_moving.shape

    assert im_f_shape == im_m_shape

    # random initialization and assignment
    scale_init = 1.25
    error_function = lambda x: utils.ssd_scale(
        x, im_fixed, im_moving, pad_size,
        disp=options.get('disp', True))
    scale_optimized = scipy.optimize.minimize(
        error_function, scale_init, method='Nelder-Mead',
        options=options, jac=False)

    # apply solved registration
    im_registered = skimage.transform.rescale(
        im_moving, scale_optimized.x, multichannel=True,
        mode='constant', anti_aliasing=True
    )

    return utils.resize_image(im_registered, orig_size)


def im_register_rotate(im_fixed, im_moving,
                       pad_size=None, options={}, show_figs=False):
    """For registering only by rotating.
    """
    if pad_size is None:
        pad_size = (im_moving.shape[0] * 2, im_moving.shape[1] * 2)

    orig_size = (im_moving.shape[0], im_moving.shape[1])

    im_fixed = utils.resize_image(im_fixed, pad_size)
    im_moving = utils.resize_image(im_moving, pad_size) / 255

    im_f_shape = im_fixed.shape
    im_m_shape = im_moving.shape

    assert im_f_shape == im_m_shape

    # random initialization and assignment
    rotate_init = 0.2
    error_function = lambda x: utils.ssd_rotate(
        x, im_fixed, im_moving, pad_size, show_figs=show_figs,
        disp=options.get('disp', True))
    scale_optimized = scipy.optimize.minimize(
        error_function, rotate_init, method='Nelder-Mead',
        options=options, jac=False)

    # apply solved registration
    im_registered = skimage.transform.rotate(
        im_moving, angle=scale_optimized.x * 180 / (math.pi),
        resize=True, mode='constant'
    )

    return utils.resize_image(im_registered, orig_size)


def im_register_affine(im_fixed, im_moving,
                       pad_size=None, options={}, show_figs=False):
    """For registering using affine transformation.
    """
    if pad_size is None:
        pad_size = (int(im_moving.shape[0] * 2),
                    int(im_moving.shape[1] * 2))

    orig_size = (im_moving.shape[0], im_moving.shape[1])

    im_fixed = utils.resize_image(im_fixed, pad_size)
    im_moving = utils.resize_image(im_moving, pad_size) / 255

    im_f_shape = im_fixed.shape
    im_m_shape = im_moving.shape

    assert im_f_shape == im_m_shape

    # random initialization and assignment
    init_scale = (1., 1.)
    init_rotation = .1
    init_shear = 0.0
    init_translation = (0, 0)
    params_init = (init_scale[0], init_scale[1], init_rotation,
                   init_shear, init_translation[0], init_translation[1])
    error_function = lambda x: utils.ssd_affine(
        x, im_fixed, im_moving, pad_size, show_figs=show_figs)
    affine_optimized = scipy.optimize.minimize(
        error_function, params_init, method='Nelder-Mead',
        options=options, jac=False)

    # apply solved registration
    regis_tfm = skimage.transform.AffineTransform(
        scale=(affine_optimized.x[0], affine_optimized.x[1]),
        rotation=affine_optimized.x[2],
        shear=affine_optimized.x[3],
        translation=(affine_optimized.x[4], affine_optimized.x[5]))
    im_registered = utils.apply_matrix_tform(
            im_moving, regis_tfm)

    return utils.resize_image(im_registered, orig_size)


def _image_interp_rbf(Image, dx, dy, pad_pct=0, upsample_factor=1):
    """
    Image - 2d array (no channel).
    dx, dy - 2d displacement fields.
    pad_pct - padding % (applies to both sides)
    upsample_factor - image size i
    """
    x_range = np.arange(0, Image.shape[1])
    y_range = np.arange(0, Image.shape[0])

    xx, yy = np.meshgrid(x_range, y_range)
    x_in, y_in = xx.flatten(), yy.flatten()

    shift = np.array([xx.shape[0], xx.shape[1], 0]).T / 2
    shift = np.reshape(shift, (len(shift), 1))

    x_out = x_in + dx - shift[0]
    y_out = y_in + dy - shift[1]

    im_max = Image.max()
    Image = Image / im_max

    # TODO: implement this for speed
    rbfi = interpolate.Rbf(
        x_out, y_out, Image.flatten(),
        function='gaussian', smooth=0.005, epsilon=1.2)

    # pad, upsample and predict
    x_pad = int(pad_pct * Image.shape[1])
    y_pad = int(pad_pct * Image.shape[0])

    x_range = np.arange(
        -x_pad, Image.shape[1] + x_pad, 1. / upsample_factor)
    y_range = np.arange(
        -y_pad, Image.shape[0] + y_pad, 1. / upsample_factor)

    xx, yy = np.meshgrid(x_range, y_range)
    x_in, y_in = xx.flatten(), yy.flatten()

    im_interped = rbfi(x_in, y_in)

    im_interped = np.clip(im_interped, 0, 1)
    im_interped = np.uint8(im_max * im_interped)
    im_interped = im_interped.reshape(xx.shape)

    return im_interped


# TODO - add warping based image_interp
def image_interp_rbf(Image, dx, dy, pad_pct=0, upsample_factor=1):
    """
    Image - 2d array (single channel or 3 channel).
    see _image_interp
    """

    if pad_pct < 0:
        raise ValueError('pad percentage has to be' +
                         'greater or equal than 0')
    if upsample_factor < 1:
        raise ValueError('upsample factor has to be' +
                         'greater or equal to 1')

    assert pad_pct >= 0
    assert upsample_factor >= 1

    if len(Image.shape) == 2:
        return _image_interp_rbf(
            Image, dx, dy, pad_pct, upsample_factor)
    elif len(Image.shape) == 3 and Image.shape[2] == 3:
        Image_out = []
        for i in range(3):
            Image_out.append(_image_interp_rbf(
                Image[:, :, i], dx, dy, pad_pct, upsample_factor))
        Image_out = np.array(Image_out)
        Image_out = Image_out.swapaxes(0, 1)
        Image_out = Image_out.swapaxes(1, 2)
        return Image_out
    else:
        raise ValueError('Images should be grayscaleor 3 channel')
