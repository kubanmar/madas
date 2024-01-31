import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SimilarityDataFramework",
    version="1.0.0rc1",
    author="Martin Kuban",
    author_email="kuban@physik.hu-berlin.de",
    description="A framework for working with materials similarity.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.physik.hu-berlin.de/kuban/similarity-data-framework",
    python_requires='>=3.10',
    install_requires = ['numpy', 
                        'pandas>=2.0', 
                        'ase>=3.22', 
                        'scipy', 
                        'matplotlib', 
                        'tqdm', 
                        'pytest'],
    packages=['simdatframe']
)
