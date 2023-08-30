# Copyright 2022 CVS Health and/or one of its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

# Set constants
NAME = "blockeval"
VERSION = "0.1.0"
DESCRIPTION = "Analyze campaigns with segments derived from a predictive model, an uplift score, or any business rule."
with open("README.md") as f:
    LONG_DESCRIPTION, LONG_DESC_TYPE = f.read(), "text/markdown"
URL = "https://github.com/cvs-health/blockeval"
LICENSE = "Apache 2.0"
AUTHORS = "d'Acremont Mathieu, Audrey Lee"
AUTHOR_EMAILS = "dacremontm@aetna.com, leey@cvshealth.com"
PYTHON_REQ = ">=3.7"
PACKAGES = find_packages(exclude=["*.tests", "*.tests.*"])
REQUIREMENTS = [
    "numpy>=1.18.1",
    "pandas>=1.0.1",
    "scipy>=1.4.1",
    "statsmodels>=0.11.1",
]

# Run setup
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    url=URL,
    license=LICENSE,
    author=AUTHORS,
    author_email=AUTHOR_EMAILS,
    python_requires=PYTHON_REQ,
    packages=PACKAGES,
    install_requires=REQUIREMENTS,
    include_package_data=True,
)
