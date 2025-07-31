from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pixel_classifier_torch',
    version='0.2.1',
    packages=find_packages(),
    long_description=long_description,

    long_description_content_type="text/markdown",
    include_package_data=True,
    author="Alexander Hartelt",
    author_email="alexander.hartelt@informatik.uni-wuerzburg.de",
    url="https://github.com/Gawajn/segmentation-pytorch",
    install_requires=[x for x in open("requirements.txt").read().split() if not x.startswith("git+")] + [
        #"doxapy @ git+https://gitlab2.informatik.uni-wuerzburg.de/nof36hn/binarizer-doxa.git"
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition"

    ],
    keywords=['OCR', 'page segmentation', 'pixel classifier'],
    data_files=[('', ["requirements.txt"])],
    entry_points={
        'console_scripts':[
            'pytorchseg_predict=segmentation.scripts.predict:main'
        ]
    }
)