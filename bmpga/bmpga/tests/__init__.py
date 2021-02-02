# coding=utf-8
"""
All project unittests should be stored here.
I am aiming for quite a high test coverage.
"""
from os.path import abspath

from bmpga.tests import __file__ as test_path

test_path = test_path.split("/")[:-1]
test_path = "/".join(test_path)
test_data_path = abspath(test_path+"/test_data")

