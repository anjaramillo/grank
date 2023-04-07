from setuptools import find_packages, setup

setup(
    name="grank",
    version="0.0.1",
    description="Group Rank",
    author="Binsheng Liu",
    author_email="liubinsheng@gmail.com",
    packages=find_packages(exclude=["docs", "tests", "scripts"]),
    include_package_data=True,
    install_requires=["numpy"],
)
