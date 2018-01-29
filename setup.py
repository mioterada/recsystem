from setuptools import setup, find_packages

setup(name='recsystem',
      version='1.0',
      description='Package for recommender system',
      author='terada',
      author_email='mio.terada@sci.hokudai.ac.jp',
      install_requires=['numpy','pandas','scipy'],
      url='https://github.com/mioterada/recsystem',
      packages=find_packages()
)
