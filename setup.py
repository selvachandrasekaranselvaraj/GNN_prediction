from setuptools import setup, find_packages

setup(
    name='autopsy',
    version='0.1.0',
    author='Selva Chandrasekaran Selvaraj',
    author_email='selvachandrasekar.s@gmail.com',
    description='A Python package for analyzing molecular dynamics trajectories',
    long_description='''Autopsy is a Python package for analyzing molecular dynamics trajectories, with a focus on computing properties such as mean-squared displacement (MSD) and non-Gaussian parameter (NGP).''',
    url='https://github.com/yourusername/autopsy',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'tqdm',
        'scipy',
        'numba',
        'pip>=22.1.2',
        'wheel>=0.37.1',
        'twine>=4.0.1',
        'ase>=3.22.1',
        # Add other dependencies as needed
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    entry_points={
        'console_scripts': [
            'autopsy=autopsy.cli.main:main',
            'plot_msd=autopsy.util.plot_autopsy_msd:plot_autopsy_msd',
            'plot_arrhenius=autopsy.util.plot_arrhenius:plot_arrhenius',
            'plot_deepPot_accuracy=autopsy.util.plot_deepPot_accuracy:plot_deepPot_accuracy',
            'plot_lammps_md=autopsy.util.plot_lammps_md:plot_lammps_md',
        ],
    },
)

