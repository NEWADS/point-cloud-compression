"""
Author: Wilson ZHANG
Date: 2020/04/26
Class pointcloud as baseline for data compression
"""
import os
import sys
from src.huffman import HuffmanCoding

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # You can try removing this lol
import numpy as np
from scipy.fftpack import dct, idct
from sklearn.metrics import mean_squared_error
from scipy.spatial import cKDTree  # This is so awesome!!!

from absl import flags
from easydict import EasyDict
from tqdm import trange

DATA_DIR = os.getcwd()
if 'data' not in DATA_DIR[-5:]:
    DATA_DIR = os.path.join(DATA_DIR, 'data')

flags.DEFINE_string('datapath', 'data set', 'Data data to read')
FLAGS = flags.FLAGS


class Pointcloud:
    def __init__(self, name: str, force_positive: bool, reduce_point: bool):
        self.dir = os.path.join(DATA_DIR, FLAGS.datapath)
        self.name = name
        self.scatter = self.load_scatter(force_positive)
        self.data = self.sampling(reduce_point)
        self.tmp = EasyDict(coef=[], decoef=[], quantizer=[], scatter=[], metrics=[], byte_stream=[])
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
            ax1 = fig.add_subplot(121, projection='3d', title='Original Scatters')
            ax1.scatter(self.scatter[..., 0], self.scatter[..., 1], self.scatter[..., 2])
            ax1.set_xlabel('X Label')
            ax1.set_ylabel('Y Label')
            ax1.set_zlabel('Z Label')
            ax2 = fig.add_subplot(122, projection='3d', title='Compressed Scatters')
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

    def dct3(self, quantization_bit: int, compress_size: int):
        # 3D DCT for point-cloud compression, decompression and evaluation will be performed.

        def _compute_ratio(raw_data: np.ndarray, compressed_data: np.ndarray):
            return raw_data.size / compressed_data.size

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
        # save config for after evaluation
        self.tmp.quantizer.append([n for n in [np.amin(coef), np.amax(coef), quantizer, coef.shape]])
        # Compression will be achieved by directly cutting the coef matrix.
        if compress_size >= coef.shape[0] or compress_size >= coef.shape[1]:
            compress_size = 0
            print('Required Transmit Matrix Size too large, will transmit raw matrix.')
        else:
            coef = coef[:compress_size, :compress_size, :]
        self.tmp.coef = coef
        # compute compression ratio:
        self.tmp.metrics.append(_compute_ratio(self.data, self.tmp.coef))
        # IDCT Decompression
        t = np.zeros(self.tmp.quantizer[0][-1])
        if compress_size:
            t[:compress_size, :compress_size, :compress_size] = coef
        else:
            t = coef
        coef = t
        decoef = coef*quantizer
        decoef = idct(decoef, axis=2, norm='ortho')
        decoef = idct(decoef, axis=1, norm='ortho')
        decoef = idct(decoef, axis=0, norm='ortho')
        self.tmp.decoef = np.round(decoef)
        # Restruction, points less than 2**q-2 will be removed.
        scatter = np.transpose(np.nonzero(decoef > (q-2)))
        self.tmp.scatter = scatter
        # Summary
        self.eval()
        self.plot_scatter3(comparison=True)

    def eval(self):
        compression_rate = self.tmp.metrics.pop(0)
        # for i in trange(self.scatter.shape[0], desc='Computing Neareast Points'):
        #     # compute and get the minimum euclidean distance for compreseed point
        #     # to lower operation complexity, squaring is removed
        #     tmp0 = 65536  # just for inialization...
        #     index = 0
        #     for j in range(self.tmp.scatter.shape[0]):
        #         tmp = sum([(a - b) ** 2 for a, b in zip(self.scatter[i], self.tmp.scatter[j])])
        #         if tmp0 <= tmp:
        #             tmp0 = tmp
        #             index = j
        #     nearest_points.append(self.tmp.scatter[index])
        # linear search sucks, please allow me to introduce kdtree!!!
        tree = cKDTree(self.tmp.scatter)
        nearest_points = [self.tmp.scatter[tree.query(self.scatter[i])[1]]
                          for i in trange(self.scatter.shape[0], desc="Computing Nearest Points")]
        rmse = mean_squared_error(y_true=self.scatter, y_pred=np.array(nearest_points), squared=False)
        # Pay attention to squared!
        print(' Announcements of Compression Result '.center(80, '-'))
        print('Reconstructed Coordinate Number: {}'.format(self.tmp.scatter.shape[0]))
        print('Compression Ratio: {}'.format(compression_rate))
        print('Root-mean-square Error Rate: {}'.format(rmse))
        print('-' * 80)

    def linearcoding(self, code_depth=16):
        # Simple Linear Coding compression, decompression and evaluation will be performed!
        def _compute_ratio(compressed, scatter: np.ndarray):
            compressed_size = 0
            scatter_size = scatter.nbytes
            for i in compressed:
                compressed_size += sys.getsizeof(i)
            return compressed_size / scatter_size

        def _d1halfing_fast(pmin, pmax, cd):
            return np.linspace(pmin, pmax, 1 << cd + 1)

        def _resampling(ppoints, cd):
            x_min = np.amin(ppoints[:, 0])
            x_max = np.amax(ppoints[:, 0])
            y_min = np.amin(ppoints[:, 1])
            y_max = np.amax(ppoints[:, 1])
            z_min = np.amin(ppoints[:, 2])
            z_max = np.amax(ppoints[:, 2])
            xletra = _d1halfing_fast(x_min, x_max, cd)
            yletra = _d1halfing_fast(y_min, y_max, cd)
            zletra = _d1halfing_fast(z_min, z_max, cd)
            otcodex = np.searchsorted(xletra, ppoints[:, 0], side='right') - 1  # establish tree using this.
            otcodey = np.searchsorted(yletra, ppoints[:, 1], side='right') - 1
            otcodez = np.searchsorted(zletra, ppoints[:, 2], side='right') - 1
            return [[otcodex, otcodey, otcodez], x_min, x_max, y_min, y_max, z_min, z_max]

        # Octree compression
        data = self.scatter
        occ = _resampling(data, code_depth)
        # coef = np.sort(occ[0])
        coef = occ[0]
        # Huffman Coding
        byte_stream_x = HuffmanCoding(data=coef[0]).compression()
        byte_stream_y = HuffmanCoding(data=coef[1]).compression()
        byte_stream_z = HuffmanCoding(data=coef[2]).compression()
        # compression ratio
        self.tmp.metrics.append(_compute_ratio([byte_stream_x, byte_stream_y, byte_stream_z], data))
        quantizer = np.array([code_depth, occ[1], occ[2], occ[3], occ[4], occ[5], occ[6]])  # depth and boundary
        self.tmp.coef = coef
        self.tmp.quantizer.append(quantizer)

        # Decompression
        x_axis = _d1halfing_fast(quantizer[1], quantizer[2], code_depth)
        y_axis = _d1halfing_fast(quantizer[3], quantizer[4], code_depth)
        z_axis = _d1halfing_fast(quantizer[5], quantizer[6], code_depth)
        koorx = x_axis[coef[0]]
        koory = y_axis[coef[1]]
        koorz = z_axis[coef[2]]
        decoef = np.array([koorx, koory, koorz]).T
        self.tmp.decoef = decoef
        self.tmp.scatter = decoef
        # Summary
        self.eval()
        self.plot_scatter3(comparison=True)

    def dpcm(self, mode='linear', code_depth=16):
        # Simple Predictive Coding compression, decompression and evaluation will be performed!
        def _compute_ratio(compressed, scatter: np.ndarray):
            compressed_size = 0
            scatter_size = scatter.nbytes
            for i in compressed:
                compressed_size += sys.getsizeof(i)
            return compressed_size / scatter_size

        def _d1halfing_fast(pmin, pmax, cd):
            return np.linspace(pmin, pmax, 1 << cd + 1)

        def _predictor(prefix: np.ndarray, md='linear'):
            # predictor for next coordinates mode can be linear or constant
            if md == 'linear':
                output = 2 * prefix[1] - prefix[0]
            else:
                output = prefix[1]
            return output

        def _predictive_coding(scatters: np.ndarray, cd: int, md: str):
            pred = []
            for i in range(scatters.shape[0]):
                if i < 2:
                    pred.append(scatters[i])
                else:
                    pred.append(scatters[i] - _predictor(scatters[i-2:i], md))

            pred = np.array(pred)
            # quantization
            x_min = np.amin(pred[:, 0])
            x_max = np.amax(pred[:, 0])
            y_min = np.amin(pred[:, 1])
            y_max = np.amax(pred[:, 1])
            z_min = np.amin(pred[:, 2])
            z_max = np.amax(pred[:, 2])
            xletra = _d1halfing_fast(x_min, x_max, cd)
            yletra = _d1halfing_fast(y_min, y_max, cd)
            zletra = _d1halfing_fast(z_min, z_max, cd)
            otcodex = np.searchsorted(xletra, pred[:, 0], side='right') - 1  # establish tree using this.
            otcodey = np.searchsorted(yletra, pred[:, 1], side='right') - 1
            otcodez = np.searchsorted(zletra, pred[:, 2], side='right') - 1
            return [[otcodex, otcodey, otcodez], x_min, x_max, y_min, y_max, z_min, z_max]

        def _reverse_coding(pred: np.ndarray):
            # when reconstruct performing, it will reverse from last point, hense first point is original coordinate.
            res = []
            for i in range(pred.shape[0]):
                if i < 2:
                    res.append(pred[i])
                else:
                    p_i = pred[i] + 2*res[i-1] - res[i-2]
                    res.append(p_i)
            return np.array(res)

        # Compression
        data = self.scatter
        occ = _predictive_coding(data, md=mode, cd=code_depth)
        coef = occ[0]
        # Huffman Coding
        byte_stream_x = HuffmanCoding(data=coef[0]).compression()
        byte_stream_y = HuffmanCoding(data=coef[1]).compression()
        byte_stream_z = HuffmanCoding(data=coef[2]).compression()
        # compression ratio
        self.tmp.metrics.append(_compute_ratio([byte_stream_x, byte_stream_y, byte_stream_z], data))
        quantizer = np.array([code_depth, occ[1], occ[2], occ[3], occ[4], occ[5], occ[6]])  # depth and boundary
        self.tmp.coef = coef
        self.tmp.quantizer.append(quantizer)

        # Decompression
        x_axis = _d1halfing_fast(quantizer[1], quantizer[2], code_depth)
        y_axis = _d1halfing_fast(quantizer[3], quantizer[4], code_depth)
        z_axis = _d1halfing_fast(quantizer[5], quantizer[6], code_depth)
        koorx = x_axis[coef[0]]
        koory = y_axis[coef[1]]
        koorz = z_axis[coef[2]]
        koor = np.array([koorx, koory, koorz]).T
        decoef = _reverse_coding(koor)

        self.tmp.decoef = decoef
        self.tmp.scatter = decoef
        # Summary
        self.eval()
        self.plot_scatter3(comparison=True)
