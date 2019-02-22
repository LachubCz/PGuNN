import os.path
import re
import unittest
import subprocess

class TestAppRuns(unittest.TestCase):
    def test_train_cct_env_01(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env CartPole-v1 -eps 2 -init -net dueling", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_train_ram_atari_env_01(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env BeamRider-ram-v0 -eps 2 -init -net dueling", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_train_img_atari_env_01(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env BeamRider-v0 -frames 3 -eps 2 -init -net dueling", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_train_2048_env_01(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env 2048-v0 -eps 2 -init -net dueling", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_train_cct_env_02(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env CartPole-v0 -eps 2 -init -mem prioritized", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_train_ram_atari_env_02(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env Breakout-ram-v0 -eps 2 -init -mem prioritized", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_train_img_atari_env_02(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env Breakout-v0 -frames 3 -eps 2 -init -mem prioritized", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_train_2048_env_02(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env 2048-v0 -eps 2 -init -mem prioritized", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_train_cct_env_03(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env MountainCar-v0 -eps 2 -alg DQN+TN -update_f 1", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_train_ram_atari_env_03(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env SpaceInvaders-ram-v0 -eps 2 -init -alg DQN+TN", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_train_img_atari_env_03(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env SpaceInvaders-v0 -frames 4 -eps 2 -init -alg DQN+TN", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_train_2048_env_03(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env 2048-v0 -eps 2 -init -mem prioritized -alg DQN+TN -update_f 1", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_train_cct_env_04(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env Acrobot-v1 -eps 2 -alg DDQN -save_f 1", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_train_ram_atari_env_04(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env Breakout-ram-v0 -eps 2 -init -alg DDQN -mdl_blueprint -save_f 2", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_train_img_atari_env_04(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env Breakout-v0 -frames 2 -eps 2 -init -alg DDQN -mdl_blueprint -save_f 2", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_train_2048_env_04(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode train -env 2048-v0 -eps 2 -init -mem prioritized -alg DDQN -dont_save", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_test_cct_env_05(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode test -env CartPole-v1 -eps 2 -mdl ./models/CartPole-v1_basic.h5", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_test_ram_atari_env_05(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode test -env BeamRider-ram-v0 -eps 2 -mdl ./models/BeamRider-ram-v0_basic.h5", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_test_img_atari_env_05(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode test -env BeamRider-v0 -frames 2 -eps 2 -mdl ./models/BeamRider-v0_basic.h5", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_test_2048_env_05(self):
        output = subprocess.check_output("python3 pgunn/main.py -mode test -env 2048-v0 -eps 2 -mdl ./models/2048-v0_basic.h5", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_visualization_01(self):
        if os.path.isfile("./model-CartPole-v1-last.h5"):
            subprocess.Popen("rm model-CartPole-v1-last.h5", shell=True)
        if os.path.isfile("./model-CartPole-v1-solved.h5"):
            subprocess.Popen("rm model-CartPole-v1-solved.h5", shell=True)
        subprocess.Popen("python3 pgunn/main.py -mode train -env CartPole-v1 -eps 100 > log.out", shell=True)
        while True:
            if os.path.isfile("./model-CartPole-v1-last.h5") or os.path.isfile("./model-CartPole-v1-solved.h5"):
                break
        output = subprocess.check_output("python3 pgunn/visualization.py -filename log.out -graph_name score -idx_val 6", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_visualization_02(self):
        if os.path.isfile("./model-CartPole-v1-last.h5"):
            subprocess.Popen("rm model-CartPole-v1-last.h5", shell=True)
        if os.path.isfile("./model-CartPole-v1-solved.h5"):
            subprocess.Popen("rm model-CartPole-v1-solved.h5", shell=True)
        subprocess.Popen("python3 pgunn/main.py -mode train -env CartPole-v1 -eps 100 > log.out", shell=True)
        while True:
            if os.path.isfile("./model-CartPole-v1-last.h5") or os.path.isfile("./model-CartPole-v1-solved.h5"):
                break
        output = subprocess.check_output("python3 pgunn/visualization.py -filename log.out -graph_name results -idx_val 6 -lines 22 30 -scatter", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


    def test_visualization_03(self):
        if os.path.isfile("./model-CartPole-v1-last.h5"):
            subprocess.Popen("rm model-CartPole-v1-last.h5", shell=True)
        if os.path.isfile("./model-CartPole-v1-solved.h5"):
            subprocess.Popen("rm model-CartPole-v1-solved.h5", shell=True)
        subprocess.Popen("python3 pgunn/main.py -mode train -env CartPole-v1 -eps 100 > log.out", shell=True)
        while True:
            if os.path.isfile("./model-CartPole-v1-last.h5") or os.path.isfile("./model-CartPole-v1-solved.h5"):
                break
        output = subprocess.check_output("python3 pgunn/visualization.py -filename log.out -graph_name moves -idx_val 4 -coordinate_x 50", shell=True)
        rgx = re.compile(r'''\[SUCCESSFUL\sRUN\]''', re.X)
        self.assertTrue(rgx.search(str(output)))


if __name__ == "__main__":
    unittest.main()
