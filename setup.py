from setuptools import find_packages, setup

import hela

setup(
    name='hela',
    version=hela.__version__,
    description="Factored hidden Markov modeling.",
    author="Will Vega-Brown",
    author_email="will@tagup.io",
    url="http://tagup.io",
    packages=find_packages())
