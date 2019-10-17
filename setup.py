import re
import ast
from setuptools import setup

_version_re = re.compile(r'__version__\s+=\s+(.*)')

# Parse version from root package version
with open('deephub/__init__.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

# Read requirements from external pip files
with open('requirements.txt', 'r') as fp:
    requirements = [x.strip() for x in fp.readlines()]

with open('requirements_test.txt', 'r') as fp:
    test_requirements = [x.strip() for x in fp.readlines()]

setup(name='deephub',
      version=version,
      packages=[
          'deephub',
          'deephub.common',
          'deephub.utils',
          'deephub.preprocessor.image',
          'deephub.preprocessor.text',
          'deephub.resources',
          'deephub.trainer',
          'deephub.models',
          'deephub.models.feeders',
          'deephub.models.feeders.tfrecords',
          'deephub.models.registry',
          'deephub.variants',
      ],
      include_package_data=True,
      install_requires=requirements,
      tests_require=test_requirements,
      setup_requires=[
          'flake8',
          'pytest-runner'],
      extras_require={
          'test': test_requirements
      },
      entry_points="""
        [console_scripts]
        deep=deephub.__main__:cli
        deep_resources=deephub.resources.__main__:cli
      """
      )
