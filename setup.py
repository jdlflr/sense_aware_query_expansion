from setuptools import setup, find_packages

requires = []
with open(r'requirements.txt', 'r') as f:
    lines = f.readlines()
    [requires.append(line.strip().strip('\n').strip()) for line in lines]

setup(
    name='saqe',
    version='0.1.0',
    description='An information retrieval tool for carrying out sense-aware query expansion',
    author='Jude LaFleur',
    url='https://github.com/jdlflr/sense_aware_query_expansion',
    keywords=[
        'information retrieval', 'IR', 'document retrieval', 'query expansion', 'word-sense disambiguation', 'WSD'
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requires
)
