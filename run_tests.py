"""
"""


import os

import nose


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'tests', 'output')


if __name__ == '__main__':

    ## make tests\output folder
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    ## empty tests\output folder
    for fn in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, fn))

    nose.main()
