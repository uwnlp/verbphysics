"""
This ended up being smaller than I expected.

author: mbforbes
"""

import os


def ensure_dir(directory):
    '''
    Makes directory and all needed parent dirs if it doesn't exist.

    Args:
        directory (str)
    '''
    if not os.path.isdir(directory):
        os.makedirs(directory)
