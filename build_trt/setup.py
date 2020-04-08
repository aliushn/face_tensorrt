from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy

library_dirs = [
    '/usr/local/cuda/lib64',
    '/usr/local/lib',
    '/root/optimization/TensorRT-7.0.0.11/lib/'
]

libraries = [
    'nvinfer',
    'cudnn',
    'cublas',
    'cudart_static',
    'nvToolsExt',
    'cudart',
    'rt',
]

include_dirs = [
    # in case the following numpy include path does not work, you
    # could replace it manually with, say,
    # '-I/usr/local/lib/python3.6/dist-packages/numpy/core/include',
    '-I' + numpy.__path__[0] + '/core/include',
    '-I/usr/local/cuda/include',
    '-I/usr/local/include',
    '-I/root/optimization/face_tensorrt/include/',
    '-I/root/optimization/face_tensorrt/include/common/',
    '-I/root/optimization/TensorRT-7.0.0.11/include/'
    ]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(
        Extension(
            'pytrt',
            sources=['pytrt.pyx'],
            language='c++',
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=['-O3', '-std=c++11'] + include_dirs
        ),
        compiler_directives={'language_level': '3'}
    )
)