from scipy import interpolate
import numpy as np
import skimage.transform
import scipy.spatial


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


def get_rbf_design_matrix(x_samp, x_center, sigma=1):
    """
    Function that returns rbf interpolation matrix.
    Note: x_samp/x_center is based off of matrix row/col format and not
         and not image x/y axis format.

    params:
    ===============================
    x_samp: where data is available
    x_center: where RBF interpolants are placed
    sigma: RBG gaussian spread's control factor
    """
    x_s_d_dist_sq = np.power(
        scipy.spatial.distance.cdist(x_samp, x_center), 2)
    x_s_d_exp = np.exp(- x_s_d_dist_sq / sigma ** 2)

    return x_s_d_exp


def solve_lst_sq_wt(H, y, lam=.01):
    """
    solution to argmin_(w) (||Hw - y||^2)

    params:
    ========================
    H - design matrix
    y - data to solve for
    lam - regularization factor
    """
    # solve least squares
    H_t_H = np.dot(H.T, H)
    lam_I = lam * np.eye(H_t_H.shape[0])
    H_t_H_inv = np.linalg.inv(H_t_H + lam_I)
    w = np.dot(H_t_H_inv, np.dot(H.T, y))
    return w


def fit_rbf_interp(x_samp, x_center, x_data_large, y_samp,
                   sigma=1, lam=.01):
    """
    Function that fits to new data points, using values from
    old data points.
    Note: x_samp/x_center is based off of matrix row/col format and not
         and not image x/y axis format.

    params:
    ===============================
    x_samp: where data is available
    x_center: where RBF interpolants are placed
    x_data_large: where interpolation needs to be done
    y_samp: data at x_samp
    sigma: RBG gaussian spread's control factor
    lam: regularization factor (to avoid matrix singularity)
    """
    H_samp = get_rbf_design_matrix(x_samp, x_center, sigma=sigma)
    w = solve_lst_sq_wt(H_samp, y_samp, lam=lam)

    # apply to larger x sample
    H_all = get_rbf_design_matrix(x_data_large, x_center, sigma=sigma)
    return np.dot(H_all, w)


class ImageDeformer:
    def __init__(self, Image, samp_space=5, sigma=10, lam=.1):
        self.Image = Image
        self.sigma = sigma
        self.lam = lam

        # large image grid
        x_range = np.arange(0, Image.shape[1])
        y_range = np.arange(0, Image.shape[0])

        xx, yy = np.meshgrid(x_range, y_range)
        xin, yin = xx.flatten(), yy.flatten()
        x_data_all = np.array([yin, xin]).T

        # small sampled grid
        x_range_samp = np.linspace(
            0, Image.shape[1] - 1, Image.shape[1] / samp_space)
        y_range_samp = np.linspace(
            0, Image.shape[0] - 1, Image.shape[0] / samp_space)

        xx_samp, yy_samp = np.meshgrid(x_range_samp, y_range_samp)
        xin_samp, yin_samp = xx_samp.flatten(), yy_samp.flatten()
        xx_samp, yy_samp = np.meshgrid(x_range_samp, y_range_samp)
        xin_samp, yin_samp = xx_samp.flatten(), yy_samp.flatten()
        x_samp = np.array([yin_samp, xin_samp]).T
        self.x_samp = x_samp

        # design matrix
        self.H_samp = get_rbf_design_matrix(x_samp, x_samp, sigma=sigma)
        self.H_all = get_rbf_design_matrix(x_data_all, x_samp, sigma=sigma)

    def apply_random_disps(self):
        x_disp_samp = 5 * np.random.randn(len(self.x_samp), 1)
        y_disp_samp = 5 * np.random.randn(len(self.x_samp), 1)

        w_x = solve_lst_sq_wt(self.H_samp, x_disp_samp, lam=self.lam)
        w_y = solve_lst_sq_wt(self.H_samp, y_disp_samp, lam=self.lam)

        dx = np.dot(self.H_all, w_x)
        dy = np.dot(self.H_all, w_y)

        return self.apply_disps(dx, dy)

    def apply_disps(self, dx, dy):
        im_interp = image_interp_warp(self.Image, dx, dy)
        return im_interp
