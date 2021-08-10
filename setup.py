import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SimilarityDataFramework",
    version="0.8",
    author="Martin Kuban",
    author_email="kuban@physik.hu-berlin.de",
    description="A framework for working with materials similarity.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.physik.hu-berlin.de/kuban/similarity-data-framework",
    python_requires='>=3.6',
    install_requires = ['numpy', 'pandas', 'scikit-learn', 'lxml', 'requests', 'ase', 'scipy', 'matplotlib', 'bitarray', 'dscribe = 0.3.5', 'tqdm', 'pytest', 'pytest-cov', 'bravado'],
    packages=['simdatframe']
)
