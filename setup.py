from setuptools import find_packages, setup

import hela

setup(
    name='hela',
    version=hela.__version__,
    description="Factored and hybrid hidden Markov modeling.",
    author="Anna Haensch",
    author_email="anna.haensch@tufts.edu",
    url="http://annahaensch.com",
    packages=find_packages())
