from setuptools import setup, find_packages

setup(
    name='TractoPL',
    version='0.1.0',
    author='Nathan Decaux',
    author_email='nathan.decaux@irisa.fr',
    description='Tractometry processing library for diffusion MRI data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/nathandecaux/TractoPL',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)