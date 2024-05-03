from setuptools import setup, find_packages

setup(
    name="stork",
    version="0.1",
    description="Surrogate gradient library based on pytorch",
    author="Friedemann Zenke",
    author_email="fzenke@gmail.com",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
)
