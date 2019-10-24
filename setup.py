# -*- coding:utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="Video Test",
    version="0.0.0",
    keywords=("pip", "faceNet", "opencv", "python3"),
    description="家用摄像头人脸捕获、对比实验",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=["opencv-python", "pillow", "facenet", "numpy", "tensorflow", "PyQt5"]
)
