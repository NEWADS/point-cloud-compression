"""
Author: Wilson ZHANG
Date: 2020/04/26
Class pointcloud as baseline for data compression
"""
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # You can try removing this lol
import numpy as np
from scipy.fftpack import dct, idct

from absl import flags
from easydict import EasyDict

DATA_DIR = os.getcwd()
if 'data' not in DATA_DIR[-5:]:
    DATA_DIR = os.path.join(DATA_DIR, 'data')

flags.DEFINE_string('datapath', 'data set', 'Data path to read')
FLAGS = flags.FLAGS


class Pointcloud:
    def __init__(self, name: str, force_positive: bool, reduce_point: bool):
        self.dir = os.path.join(DATA_DIR, FLAGS.datapath)
        self.name = name
        self.scatter = self.load_scatter(force_positive)
        self.data = self.sampling(reduce_point)
        self.tmp = EasyDict(coef=[], decoef=[], quantizer=[], scatter=[], metrics=EasyDict())
        print(' Point Clouds Config '.center(80, '-'))
        print('Dictionary: {}'.format(self.dir))
        print('Name: {}'.format(self.name))
        print('Raw Coordinate Number: {}'.format(self.scatter.shape[0]))
        print('Sampled Matrix size: {}'.format(self.data.shape))
        print('-' * 80)

    def load_scatter(self, force_positive=False):
        fullname = os.path.join(self.dir, self.name)
        x = np.loadtxt(fullname)
        if force_positive:
            for i in range(x.shape[-1]):
                x[..., i] -= np.min(x[..., i])
        return x

    def plot_scatter3(self, comparison=False):
        if comparison:
            fig = plt.figure()
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(self.scatter[..., 0], self.scatter[..., 1], self.scatter[..., 2])
            ax1.set_xlabel('X Label')
            ax1.set_ylabel('Y Label')
            ax1.set_zlabel('Z Label')
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(self.tmp.scatter[..., 0], self.tmp.scatter[..., 1], self.tmp.scatter[..., 2])
            ax2.set_xlabel('X Label')
            ax2.set_ylabel('Y Label')
            ax2.set_zlabel('Z Label')
        else:
            # plot point cloud as three dimension scatter
            fig = plt.figure()
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.scatter(self.scatter[..., 0], self.scatter[..., 1], self.scatter[..., 2])
            ax1.set_xlabel('X Label')
            ax1.set_ylabel('Y Label')
            ax1.set_zlabel('Z Label')

        plt.show()

    def sampling(self, reduce_point=False):
        # this part basically just repeat the work from RAM-LAB's paper
        scatter = self.scatter
        if reduce_point:
            start = self.scatter.shape[0] % 10
            scatter = [[self.scatter[0:start, 0].mean(),
                        self.scatter[0:start, 1].mean(),
                        self.scatter[0:start, 2].mean()]]
            for i in range(start, self.scatter.shape[0], 10):
                scatter.append([self.scatter[i:i + 10, 0].mean(),
                                self.scatter[i:i + 10, 1].mean(),
                                self.scatter[i:i + 10, 2].mean()])
            scatter = np.array(scatter)
        sampled = np.around(scatter).astype(np.int)  # make int
        data = np.zeros([n+1 for n in np.max(sampled, axis=0)], dtype=np.int)
        for i in range(sampled.shape[0]):
            x = sampled[i, :]
            data[x[0], x[1], x[2]] = 255
        return data

    def plot_matrix(self):
        # require mayavi to model the matrix.
        pass

    def dct3(self, quantization_bit: int):
        # 3D DCT for point-cloud compression, decompression and evaluation will be performed.
        data = self.data
        # DCT Compression
        coef = dct(data, axis=0, norm='ortho')
        coef = dct(coef, axis=1, norm='ortho')
        coef = dct(coef, axis=2, norm='ortho')
        # The original paper didn't release the quantization matrix for so we will simply quantize the matrix using
        # linear scales, the compression ratio will depend on the quantization bit.
        q = 1 << quantization_bit
        quantizer = (np.amax(coef) - np.amin(coef)) / q
        coef = np.round(coef/quantizer)
        self.tmp.coef = coef
        # save config for after evaluation
        self.tmp.quantizer.append([n for n in [np.amin(coef), np.amax(coef), quantizer]])
        # IDCT Decompression
        decoef = coef*quantizer
        decoef = idct(decoef, axis=2, norm='ortho')
        decoef = idct(decoef, axis=1, norm='ortho')
        decoef = idct(decoef, axis=0, norm='ortho')
        self.tmp.decoef = np.round(decoef)
        # Restruction, points less than 254 will be removed.
        scatter = np.transpose(np.nonzero(decoef >= (q-2)))
        self.tmp.scatter = scatter
        # Summary
        compression_rate = self.tmp.scatter.shape[0] / self.tmp.scatter[0]
        # error_rate =
        print(' Announcements of Compression Result '.center(80, '-'))
        print('Reconstructed Coordinate Number: {}'.format(scatter.shape[0]))
        print('-' * 80)
        self.plot_scatter3(comparison=True)

    def eval(self):
        pass

    def dpcm_compression(self):
        pass
