# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

setuptools.setup(
    name='ts-cash',
    version='1.0.0',
    packages=setuptools.find_packages(),
    python_requires='>=3.10',
    # 包含指定的csv文件 uts_dataset/best_sample_rate.csv
    package_data={
        'pylibs': ['uts_dataset/best_sample_rate.csv'],
    },
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        # 'Operating System :: POSIX :: Linux',
        # 'Operating System :: MacOS',
    ],
    author='gsunwu@163.com',
    author_email='gsunwu@163.com',
    description='',
    license='MIT',
    url=''
)
