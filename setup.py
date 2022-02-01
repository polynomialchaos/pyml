from setuptools import setup

setup(
    name='pyml',
    version='1.0',
    description='A simple python package for machine learning development.',
    author='Florian',
    author_email='polynomialchaos@gmail.com',
    packages=['pyml'],
    install_requires=['numpy', 'scipy', 'matplotlib'],
    entry_points={
        "console_scripts": [
            # 'pyML=pyml.bin.pyML:main',
        ]
    }
)
