"""
Setup configuration for feature-engineering-tk package
"""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name='feature-engineering-tk',
    version='2.3.0',
    author='bluelion1999',
    author_email='',
    description='Feature Engineering Toolkit - A comprehensive Python library for feature engineering and data analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/bluelion1999/feature_engineering_tk',
    project_urls={
        'Bug Tracker': 'https://github.com/bluelion1999/feature_engineering_tk/issues',
        'Source Code': 'https://github.com/bluelion1999/feature_engineering_tk',
    },
    packages=find_packages(exclude=['tests', 'tests.*']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.8',
    install_requires=[
        'pandas>=1.5.0',
        'numpy>=1.23.0',
        'scikit-learn>=1.2.0',
        'scipy>=1.9.0',
        'matplotlib>=3.6.0',
        'seaborn>=0.12.0',
        'statsmodels>=0.14.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'black>=23.0',
            'flake8>=6.0',
            'mypy>=1.0',
        ],
    },
    keywords='machine-learning feature-engineering data-analysis preprocessing feature-selection',
    include_package_data=True,
)
