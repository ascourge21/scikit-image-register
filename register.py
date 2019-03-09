import math

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
