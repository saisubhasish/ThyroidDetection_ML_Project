from setuptools import find_packages, setup
from typing import List

REQUIREMENT_FILE_NAME="requirements.txt"
HYPHEN_E_DOT = "-e ."    # To execute the setup.py file we use '-e .', so that we can use our project as library from anywhere

def get_requirements()->List[str]:    # Provides information to developer that this function returns list of string
    
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()     # reading each line from the file
    requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list]    # Removing '\n' from the list of libraries
    
    if HYPHEN_E_DOT in requirement_list:
        requirement_list.remove(HYPHEN_E_DOT)      # Removing '-e .' as it is not required
    return requirement_list


setup(
    name = "thyroid",
    version = "0.0.1",
    author = "SaiSubhasish",
    author_email = "saisubhasishrout777@gmail.com",
    packages = find_packages(),       # find_packages() will convert the folder with __init__.py file to library/package
    install_requires=get_requirements())   # Searches and stores the list of required libraries which are needed to be inststalled to run this project

