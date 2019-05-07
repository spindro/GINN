from setuptools import setup, find_packages

setup(name='ginn',
      version= '0.1',
      description= 'Missing data imputation with Graph Neural Networks',
      url= 'http://github.com/spindro/GINN',
      author= 'Indro Spinelli',
      author_email= 'indro.spinelli@gmail.com',
      licence= 'Apache',
      packages= find_packages(),
      install_requires=[
          "dgl",
          "numpy",
          "networkx",
          "scikit-learn",
      ],
#      packages=['ginn'],
      zip_safe=False)