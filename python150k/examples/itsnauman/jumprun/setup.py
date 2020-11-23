from setuptools import setup
import sys
import os

if not sys.version_info[0] == 3:
    print("Sorry, Jumprun doesn't support Python 2")
    sys.exit(1)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='jumprun',
    version='1.12',
    py_modules=['jumprun'],
    description='Command-line utility to create shortcuts'
                ' for running scripts',
    url='http://github.com/itsnauman/jumprun',
    author='Nauman Ahmad',
    author_email='nauman-ahmad@outlook.com',
    license='MIT',
    long_description=read('README.md'),
    include_package_data=True,
    install_requires=[
        'termcolor',
        'docopt',
    ],
    entry_points='''
        [console_scripts]
        jr=jumprun:main
    ''',
)
