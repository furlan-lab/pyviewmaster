# setup.py

from setuptools import setup, find_packages

setup(
    name='pyviewmaster',  # Replace with your package name
    version='0.0.1',
    author='Scott Furlan',
    author_email='scott.furlan@example.com',
    description='viewmastR uses machine learning implemented in Rust to perform automated cell type classification for single-cell genomic data.  Currently viewmastR is authored to work with scRNAseq data, but more features are coming.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/furlan-lab/pyviewmaster',  # Replace with your repository URL
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        # Add other classifiers as appropriate
    ],
    install_requires=[
        "numpy",
        "pandas",  
        "scanpy",
        "scikit-learn",
        "scipy" # Add your package dependencies here
    ],
    python_requires='>=3.9',
)

'''
python setup.py sdist bdist_wheel

'''