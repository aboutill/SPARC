from setuptools import setup


setup(
    entry_points={
        "console_scripts": [
            "sparc = sparc.bin.dispatch:main",
            "pipeline = sparc.bin.pipeline:main",
            "train = sparc.bin.train:main",
            "test = sparc.bin.test:main",
            ]
        }
)
