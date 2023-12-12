import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


def GenerateGAF(all_ts, window_size, rolling_length, fname, normalize_window_scaling=1.0, method='summation', scale='[0,1]'):

    # get length of time series
    n = len(all_ts)

    # to make sure the normalizing size for moving window is considered
    moving_window_size = int(window_size * normalize_window_scaling)

    # calculate the steps of rolling
    n_rolling_data = (n - moving_window_size) // rolling_length + 1

    # record the Gramian Angular Field
    gramian_field = []

    # start to generate GAF
    for i_rolling_data in trange(n_rolling_data, desc="Generating...", ascii=True):

        # start position index
        start_flag = i_rolling_data*rolling_length

        # get the data in the window
        full_window_data = list(
            all_ts[start_flag: start_flag + moving_window_size])

        # normalize the data in the window to [0,1] or [-1,1]
        rescaled_ts = np.zeros((moving_window_size, moving_window_size), float)
        min_ts, max_ts = np.min(full_window_data), np.max(full_window_data)
        diff = max_ts - min_ts
        if scale == '[0,1]':
            if diff != 0:
                rescaled_ts = (full_window_data - min_ts) / diff
        if scale == '[-1,1]':
            if diff != 0:
                rescaled_ts = (2 * full_window_data - diff) / diff

        # if normalize_window_scaling > 1, ignore the data out of the original window size
        rescaled_ts = rescaled_ts[-int(window_size *
                                       (normalize_window_scaling-1)):]

        # calculate the Gramian Angular Matrix
        this_gam = np.zeros((window_size, window_size), float)
        sin_ts = np.sqrt(np.clip(1 - rescaled_ts**2, 0, 1))
        if method == 'summation':
            # cos(x1+x2) = cos(x1)cos(x2) - sin(x1)sin(x2)
            this_gam = np.outer(rescaled_ts, rescaled_ts) - \
                np.outer(sin_ts, sin_ts)
        if method == 'difference':
            # sin(x1-x2) = sin(x1)cos(x2) - cos(x1)sin(x2)
            this_gam = np.outer(sin_ts, rescaled_ts) - \
                np.outer(rescaled_ts, sin_ts)

        gramian_field.append(this_gam)

        # garbage collection
        del this_gam

    # dump Gramian Angular Field to pickle file
    np.asarray(gramian_field, dtype="float").dump('%s_gaf.pkl' % fname)

    # garbage collection
    del gramian_field


def PlotHeatmap(all_img, save_dir='output_img'):

    # create output directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # get total length of images
    total_length = all_img.shape[0]

    # get zero padding size for naming
    fname_zero_padding_size = int(np.ceil(np.log10(total_length)))

    # start to plot
    for img_no in trange(total_length, desc="Output Heatmaps...", ascii=True):
        this_fname = str(img_no).zfill(fname_zero_padding_size)
        plt.imshow(all_img[img_no], cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.savefig("%s/%s.png" % (save_dir, this_fname),
                    bbox_inches='tight', pad_inches=0, transparent=True)
        plt.clf()


#
#
# DEMO
#
#
if __name__ == '__main__':

    random_series = np.random.uniform(low=110.0, high=150.0, size=(200,))

    timeSeries = list(random_series)
    windowSize = 50
    rollingLength = 50
    fileName = 'demo_%02d_%02d' % (windowSize, rollingLength)
    GenerateGAF(all_ts=timeSeries,
                window_size=windowSize,
                rolling_length=rollingLength,
                fname=fileName,
                normalize_window_scaling=1.0)

    ts_img = np.load(f'{fileName}_gaf.pkl', allow_pickle=True)
    PlotHeatmap(ts_img)
