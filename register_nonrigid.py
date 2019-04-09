from scipy import interpolate
import numpy as np
import skimage.transform


def _image_interp_rbf(Image, dx, dy, pad_pct=0, upsample_factor=1):
    """
    Image - 2d array (no channel).
    dx, dy - 2d displacement fields.
    pad_pct - padding % (applies to both sides)
    upsample_factor - factor to improve RBF calculation resolution
    """
    x_range = np.arange(0, Image.shape[1])
    y_range = np.arange(0, Image.shape[0])

    xx, yy = np.meshgrid(x_range, y_range)
    x_in, y_in = xx.flatten(), yy.flatten()

    shift = np.array([xx.shape[0], xx.shape[1], 0]).T / 2
    shift = np.reshape(shift, (len(shift), 1))

    x_out = x_in + dx.flatten() - shift[0]
    y_out = y_in + dy.flatten() - shift[1]

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


def image_interp_rbf(Image, dx, dy, pad_pct=0, upsample_factor=1):
    """
    Image - 2d array (single channel or 3 channel).
    see _image_interp_rbf
    """

    if pad_pct < 0:
        raise ValueError('pad percentage has to be' +
                         'greater or equal than 0')
    if upsample_factor < 1:
        raise ValueError('upsample factor has to be' +
                         'greater or equal to 1')

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
        raise ValueError('Images should be grayscale or 3 channel')


def _image_interp_warp(Image, dx, dy):
    """
    Image - 2d array (no channel).
    dx, dy - reverse transformation
        (disps that map image_out back to image)

    Image dimensions and (dx, dy) dimensionsh have to match;
        i.e., len(dx) = row(image) * col(image)
    """
    x_range = np.arange(0, Image.shape[1])
    y_range = np.arange(0, Image.shape[0])

    xx, yy = np.meshgrid(x_range, y_range)
    x_in, y_in = xx.flatten(), yy.flatten()

    # coords in output space
    x_out = x_in + dx.flatten()
    y_out = y_in + dy.flatten()

    x_out = np.expand_dims(x_out, 1),
    y_out = np.expand_dims(y_out, 1)

    # something like coord??
    xy_out_grid = np.zeros((2, xx.shape[0], yy.shape[1]))
    xy_out_grid[0] = np.reshape(y_out, (xx.shape[0], yy.shape[1]))
    xy_out_grid[1] = np.reshape(x_out, (xx.shape[0], yy.shape[1]))

    im_max = Image.max()
    Image = Image / im_max

    im_new = skimage.transform.warp(
        Image, xy_out_grid,
        order=3, preserve_range=True, mode='constant',
        clip=True
    )

    im_new = np.clip(im_new, 0, 1)
    im_new = np.uint8(im_max * im_new)

    return im_new


def image_interp_warp(Image, dx, dy):
    """
    Image - 2d array (grayscale or rgb).
    dx, dy - reverse transformation
        (disps that map image_out back to image)
    """
    num_pixels = Image.shape[0] * Image.shape[1]
    if (len(dx) != num_pixels or len(dy) != num_pixels):
        raise ValueError('dx and dy length have to equal the' +
                         ' product of image rows and columns')

    if len(Image.shape) == 2:
        return _image_interp_warp(Image, dx, dy)
    elif len(Image.shape) == 3 and Image.shape[2] == 3:
        Image_out = []
        for i in range(3):
            Image_out.append(_image_interp_warp(
                Image[:, :, i], dx, dy))
        Image_out = np.array(Image_out)
        Image_out = Image_out.swapaxes(0, 1)
        Image_out = Image_out.swapaxes(1, 2)
        return Image_out
    else:
        raise ValueError('Images should be grayscale or 3 channel')
