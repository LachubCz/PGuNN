import re
import unittest
import subprocess

class TestAppRuns(unittest.TestCase):
    
    def test_cct_env(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env CartPole-v0 -eps 2", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_atari_env(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env Breakout-ram-v0 -eps 2", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_2048_env(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env 2048-v0 -eps 2", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


if __name__ == "__main__":
    unittest.main()
