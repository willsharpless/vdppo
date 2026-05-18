from setuptools import setup, find_packages

setup(
    name="vdppo",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # tbd
    ],
    include_package_data=True,
    zip_safe=False,
)
