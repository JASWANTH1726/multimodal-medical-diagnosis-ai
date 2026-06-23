from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name='multimodal-medical-diagnosis-ai',
    version='0.1.0',
    description='Advanced multimodal medical diagnosis system with explainable AI',
    long_description=long_description,
    url='https://github.com/JASWANTH1726/multimodal-medical-diagnosis-ai',
    author='JASWANTH1726',
    packages=find_packages(),
    python_requires='>=3.9',
)