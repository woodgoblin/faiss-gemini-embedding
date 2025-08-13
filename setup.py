#!/usr/bin/env python3
"""
Setup script for FAISS Gemini Embedding System.
"""

import os

from setuptools import find_packages, setup


# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]


setup(
    name="faiss-gemini-embedding",
    version="0.1.0",
    author="AI Assistant",
    author_email="ai@example.com",
    description="A system for generating embeddings using Google's Gemini model and storing them with FAISS",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/faiss-gemini-embedding",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.11.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "safety>=3.0.0",
            "bandit>=1.7.5",
        ]
    },
    entry_points={
        "console_scripts": [
            "faiss-gemini=src.faiss_gemini.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
