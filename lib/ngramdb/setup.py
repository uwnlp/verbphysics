from setuptools import setup

setup(name='ngramdb',
      version='0.1.1',
      description='Provides access to the Myria DB of syntactic n-grams.',
      author='lzilles',
      author_email='lzilles@cs.washington.edu',
      packages=['ngramdb'],
      dependency_links=[
        'https://github.com/uwescience/myria-python/archive/master.zip'],
      # install_requires=['myria-python'],
      zip_safe=False)
