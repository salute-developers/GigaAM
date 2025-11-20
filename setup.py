import pkg_resources
from setuptools import find_packages, setup

setup(
    name="gigaam",
    py_modules=["gigaam"],
    version="0.1.0",
    description="GigaAM: A package for audio modeling and ASR.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    author="GigaChat Team",
    url="https://github.com/salute-developers/GigaAM/",
    license="MIT",
    packages=find_packages(include=["gigaam"]),
    python_requires=">=3.10",
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open("requirements.txt", "r", encoding="utf-8").read()
        )
    ],
    extras_require={
        "longform": ["torch==2.8.*", "torchaudio==2.8.*", "pyannote.audio==4.0", "torchcodec==0.7", "numba>=0.62"],
        "tests": ["pytest", "pytest-cov", "scipy", "soundfile", "librosa"],
    },
    include_package_data=True,
)
