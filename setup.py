import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="similarity_data_framework",
    version="0.8",
    author="Martin Kuban",
    author_email="kuban@physik.hu-berlin.de",
    description="A framework for working with materials similarity.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages()
)
