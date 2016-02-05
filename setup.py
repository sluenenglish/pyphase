from setuptools import setup

setup(
        name='PyPhase',
        description='Tools to use and fit Phase Type Distributions in Python',
        version='0.0',
        author='Sam Luen-English',
        author_email=('sluenenglish@gmail.com'),
        url='http://github.com/sluenenglish/pyphase',
        license='MIT',
        packages=['pyphase'],
        install_requires=[
          'numpy',
        ]
)
