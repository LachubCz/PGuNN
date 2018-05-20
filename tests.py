import os
import re
import sys
import unittest
import subprocess

class TestAppRuns(unittest.TestCase):

    def test_output_check(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env CartPole-v0 -eps 2", shell=True)
        rgx = re.compile(r'''\[Graph\sof\slearning\sprogress\svisualization\swas\smade\.\]''', re.X)
        self.assertTrue(rgx.search(str(output)))

if __name__ == "__main__":
    unittest.main()
