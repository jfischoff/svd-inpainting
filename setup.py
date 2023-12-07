from setuptools import setup, find_packages

# Function to read the list of requirements from requirements.txt
def read_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()

setup(
    name='svd_inpainting',
    version='0.1',
    packages=find_packages(),
    install_requires=read_requirements(),
    # Optional metadata
    author='Jonathan Fischoff',
    author_email='jonathan.g.fischoff@gmail.com',
    description='svd inpainting',
    url='https://github.com/jfischoff/svd-inpainting',
)
