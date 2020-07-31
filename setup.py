from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

from torch.utils import cpp_extension

print(cpp_extension.include_paths())

__version__ = '0.0.1'


extensions = [
    cpp_extension.CppExtension('src.qp_fast',
              ["src/qp_fast.cpp"],
              language='c++',
              extra_compile_args=['-std=c++17'],
    ),
]

setup(name='latent_decision_tree',
      version=__version__,
      author="VZ,MK,VN",
      ext_modules=extensions,
      setup_requires=['pybind11>=2.5.0'],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      zip_safe=False
)
