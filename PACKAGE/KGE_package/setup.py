import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="extract_sfm", # Replace with your own username
    version="1.0",
    author="Yueen Ma",
    # author_email="author@example.com",
    description="Knowledge Graph Extraction for SFM dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/security-force-monitor/ian-nlp-2019",
    packages=setuptools.find_packages(),
    package_data={'': ['results/*', 'results/scores/*', 'results/scores/*', 'results/model/*',\
                'SFM_STARTER/*', \
                'jPTDP-master/*', 'jPTDP-master/utils/*', 'jPTDP-master/sample/*', \
                'jPTDP-master/.DyNet/.Python', 'jPTDP-master/.DyNet/bin/*']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'tensorflow',
        'spacy'
    ]
)
