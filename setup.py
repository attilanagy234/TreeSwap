from setuptools import find_packages, setup

DESCRIPTION = "HU-EN neuralt machine translation"

setup(
    name="hunmt-bert",
    version="0.1",
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    #author_email="dev@",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["*.py"]},
    install_requires=[],
)