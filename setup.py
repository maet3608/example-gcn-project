import exgcn

from setuptools import setup, find_packages

setup(
    name='exgcn',
    version=exgcn.__version__,
    author='Stefan Maetschke',
    author_email='stefan.maetschke@gmail.com',
    description='Minimal example for a Graph Convolutional Network',
    install_requires=[
        'nutsml >= 1.0.44',
        'networkx >= 2.4',
        'torch >= 1.0.0',
        'dgl >= 0.4.3',
        'pytest >= 5.3.0',
    ],
    tests_require=['pytest >= 3.0.3'],
    platforms='any',
    packages=find_packages(exclude=['setup']),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
    ],
)
