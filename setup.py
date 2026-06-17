"""Setup script for the market-risk-engine package."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="market-risk-engine",
    version="0.1.0",
    description=(
        "Institutional-grade market risk engine: Historical/Parametric/Monte Carlo "
        "VaR, Expected Shortfall, Basel III/IV backtesting, stress testing, risk "
        "decomposition and capital allocation."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="market-risk-engine contributors",
    url="https://github.com/rxj0102/market-risk-engine",
    license="MIT",
    packages=find_packages(exclude=("tests", "notebooks", "paper")),
    py_modules=["core", "app"],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
        "statsmodels>=0.14",
        "scikit-learn>=1.3",
    ],
    extras_require={
        "app": ["streamlit>=1.30", "plotly>=5.18", "matplotlib>=3.7"],
        "paper": ["reportlab>=4.0"],
        "dev": ["pytest>=7.4", "black", "isort"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Office/Business :: Financial :: Investment",
        "Intended Audience :: Financial and Insurance Industry",
    ],
)
