from setuptools import setup, find_packages

install_requires = []

setup(
    name='idp_utils',
    version='1.0.0',
    packages=find_packages(),
    install_requires=install_requires,
    zip_safe=False,
    include_package_data=True
)