from setuptools import setup, find_packages

setup(name="manual_fitter",
	version="0.0.1",
	packages = find_packages(),
    python_requires=">=3.7",
	install_requires=[
          'scipy', 
          'matplotlib',
          'numpy >= 1.20',
      ],
	)
