from setuptools import setup, find_packages
setup(
    name = "acab",
    version = "0.1",
    packages = find_packages(),
    # scripts = ["say_hello.py"],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    # install_requires=["docutils>=0.3"],

    # package_data={
    #     # If any package contains *.txt or *.rst files, include them:
    #     "": ["*.txt", "*.rst"],
    #     # And include any *.msg files found in the "hello" package, too:
    #     "hello": ["*.msg"],
    # },

    # metadata to display on PyPI
    author = "RomanczukLab",
    author_email = "???",
    description = "a collection of utility functions for analysing time series data of both behavioural observations and modelling approaches",
    keywords = "tool, time-series, analyze",
    # url = "http://example.com/HelloWorld/",   # project home page, if any
    # project_urls = {
    #     "Bug Tracker": "https://bugs.example.com/HelloWorld/",
    #     "Documentation": "https://docs.example.com/HelloWorld/",
    #     "Source Code": "https://code.example.com/HelloWorld/",
    # },
    classifiers = [
        "License :: OSI Approved :: Python Software Foundation License"
    ]

    # could also include long_description, download_url, etc.
)
