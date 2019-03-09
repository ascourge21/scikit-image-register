import math

import matplotlib.pyplot as plt
import numpy as np
import skimage.util
import skimage.transform


# TODO (nripesh): add tests


def resize_image(im, new_size):
    """Resize image to given size with appropriate padding/cropping.

    TODO: Currently assumes x/y axes are scaled by the same factor.
    """
    orig_row, orig_col = im.shape[0], im.shape[1]
    assert orig_row > 0
    assert orig_col > 0
    new_row, new_col = new_size

    if new_row > orig_row:  # assuming rows/cols scale similarly
        row_pad_before = math.ceil((new_row - orig_row) / 2)
        row_pad_after = new_row - orig_row - row_pad_before

        col_pad_before = math.ceil((new_col - orig_col) / 2)
        col_pad_after = new_col - orig_col - col_pad_before

        if len(im.shape) == 3:
            im_resized = skimage.util.pad(
                im, ((row_pad_before, row_pad_after),
                     (col_pad_before, col_pad_after),
                     (0, 0)), mode='constant',
            )
        else:
            im_resized = skimage.util.pad(
                im, ((row_pad_before, row_pad_after),
                     (col_pad_before, col_pad_after)),
                mode='constant',
            )
    else:
        row_crop_before = math.ceil((orig_row - new_row) / 2)
        row_crop_after = orig_row - new_row - row_crop_before

        col_crop_before = math.ceil((orig_col - new_col) / 2)
        col_crop_after = orig_col - new_col - col_crop_before

        if len(im.shape) == 3:
            im_resized = skimage.util.crop(
                im, ((row_crop_before, row_crop_after),
                     (col_crop_before, col_crop_after),
                     (0, 0))
            )
        else:
            im_resized = skimage.util.crop(
                im, ((row_crop_before, row_crop_after),
                     (col_crop_before, col_crop_after)),
            )
    assert im_resized.shape[0] == new_size[0]
    assert im_resized.shape[1] == new_size[1]
    return im_resized


def ssd_scale(scale_factor, im_fixed, im_moving, im_pad_size,
              xx=None, yy=None, gradient=False):
    """SSD function only for scaling transformation.

        TODO (nripesh) : separate the image difference to different function
        with other differences also avaialble.
        TODO (nripesh): refactor to use the same transformation as in affine.

    """
    # re-initialize if problematic
    if scale_factor < 0:
        scale_factor = [np.abs(np.random.randn())]
        print(scale_factor)

    im_moving_regis = skimage.transform.rescale(
        im_moving, scale=scale_factor, multichannel=True,
        mode='constant', anti_aliasing=True
    )

    im_movig_regis = resize_image(im_moving_regis, im_pad_size)
    row, col, channel = im_movig_regis.shape

    q1 = im_fixed - im_movig_regis
    ssd = np.sum(np.square(q1)) / (row * col * channel)

    if xx is None:
        xx, yy = np.meshgrid(
            np.arange(im_pad_size[0]), np.arange(im_pad_size[1]))

    if not gradient:
        print(scale_factor, ssd)
        return ssd

    grad = 0
    for c in range(channel):
        # grad along single dimension should be enough
        grad = np.gradient(im_movig_regis[:, :, c])
        grad_x, grad_y = grad[0], grad[1]

        # this is specific to scaling
        grad_x = np.multiply(grad_x / scale_factor[0], xx)
        grad_y = np.multiply(grad_y / scale_factor[0], yy)

        grad += (
            - 2 * np.sum(np.multiply(
                q1[:, :, c], grad_x + grad_y)) / (row * col * channel))

    print(scale_factor, ssd, grad)
    return ssd, grad


def ssd_rotate(angle, im_fixed, im_moving, im_pad_size,
               show_figs=False):
    """Rotate SSD.

    @param angle: angle to rotate in radians.
    """
    # re-initialize if problematic
    if angle < - math.pi:
        angle = 0
    if angle > math.pi:
        angle = 0

    im_moving_regis = skimage.transform.rotate(
        im_moving, angle=angle * 180 / math.pi,
        resize=True, mode='constant'
    )

    im_movig_regis = resize_image(im_moving_regis, im_pad_size)
    row, col, channel = im_movig_regis.shape

    q1 = im_fixed - im_movig_regis
    ssd = np.sum(np.square(q1)) / (row * col * channel)

    print(angle, ssd)

    if show_figs:
        plt.figure(1)
        plt.title('target (transformed)')
        plt.imshow(im_fixed)
        plt.figure(2)
        plt.title('moving (orig)')
        plt.imshow(im_moving)
        plt.figure(3)
        plt.title('registered')
        plt.imshow(im_moving_regis)
        plt.show()
    return ssd


def apply_matrix_tform(im, tform):
    """Affine transforms are applied by calling the wrap function.

    Since the wrap function doesn't transform images about the center,
    a combination of a translation and its inverse needs to be applied to
    the tform before and after respectively.
    """
    shift_y, shift_x = np.array(im.shape[:2]) / 2.
    tf_shift = skimage.transform.AffineTransform(
        translation=[-shift_x, -shift_y])
    tf_shift_inv = skimage.transform.AffineTransform(
        translation=[shift_x, shift_y])
    image_tformed = skimage.transform.warp(
        im, (tf_shift + (tform + tf_shift_inv)).inverse)
    return image_tformed


def ssd_affine(params, im_fixed, im_moving, im_pad_size,
               show_figs=False):
    """SSD is calculated between im_fixed and transformed moving image.

    TODO (nripesh): calculate gradient (numeric or mathematical)
    TODO : callbacks to view progress as images?
    """
    # get individual params
    scale_x, scale_y, rotation, shear, trans_x, trans_y = params

    # re-initialize bad scales
    if scale_x < 0:
        scale_x = [np.abs(np.random.randn())]
        print(scale_x)
    if scale_x < 0:
        scale_x = [np.abs(np.random.randn())]
        print(scale_x)

    moving_tfm = skimage.transform.AffineTransform(
        scale=(scale_x, scale_y),
        rotation=rotation,
        shear=shear,
        translation=(trans_x, trans_y))

    im_moving_regis = apply_matrix_tform(
        im_moving, moving_tfm)

    row, col, channel = im_moving_regis.shape

    q1 = im_fixed - im_moving_regis
    ssd = np.sum(np.square(q1)) / (row * col * channel)

    print('scale_x: {:2.2f}, scale_y: {:2.2f}, rotation: {:2.2f}, '.format(
        scale_x, scale_y, rotation) +
          'shear: {:2.2f}, t_x: {:2.2f}, t_y: {:2.2f}, SSD: {:2.3f}'.format(
        shear, trans_x, trans_y, ssd))

    if show_figs:
        plt.figure(1)
        plt.title('target (transformed)')
        plt.imshow(im_fixed)
        plt.figure(2)
        plt.title('moving (orig)')
        plt.imshow(im_moving)
        plt.figure(3)
        plt.title('registered')
        plt.imshow(im_moving_regis)
        plt.show()

    return ssd
