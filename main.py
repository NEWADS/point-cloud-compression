"""
Author: Wilson ZHANG
Date: 2020/04/26
point cloud compression
"""

import os
from absl import app
from absl import flags
from src.data import Pointcloud

FLAGS = flags.FLAGS


def main(argv):
    del argv
    data = Pointcloud(
        name=FLAGS.name,
        force_positive=True,
        reduce_point=True,
    )
    data.plot_scatter3()
    # data.dct3(quantization_bit=6, compress_size=220)
    # data.linearcoding(code_depth=16)
    data.dpcm(mode='linear', code_depth=16)


if __name__ == '__main__':
    flags.DEFINE_string('name', 'kangaroo.dat', 'point cloud file')
    app.run(main)
