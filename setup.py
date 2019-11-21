from setuptools import setup, find_packages


setup(
   name='ignite_helpers',
   version='0.0.1',
   description='Utility tools for PyTorch Ignite',
   author='Felix Abrahamsson',
   author_email='FelixAbrahamsson@github.com',
   keywords='ignite pytorch torch helpers utils',
   packages=['ignite_helpers'],
   install_requires=[
       'torch',
       'pytorch-ignite',
   ],
)