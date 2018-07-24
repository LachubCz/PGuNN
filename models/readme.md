# Trained Models

The score from each environment depends heavily on the setting of hyperparameters, which was different for almost every environment.

## Control theory problems environments

### Score:
Scores were obtained from 25 same training sessions for each variant of DQN. Score of CartPole-v0, CartPole-v1, MountainCar-v0 means average number of episodes required to solve an environment. Score of Acrobot-v1 means average score from 100 testing episodes after 100 episodes of training.

|                              | CartPole-v0    | CartPole-v1  | MountainCar-v0 | Acrobot-v1   |
|------------------------------|:--------------:|:------------:|:--------------:|:------------:|
| DQN                          |       281      |    890       |     405        |     -128     |
| DQN+TN                       |       124      |    399       |     347        |     -128     |
| DDQN                         |       73       |    451       |     309        |     -125     |
| DQN, Priority ER             |       202      |    693       |     931        |     -149     |
| DQN+TN, Priority ER          |       69       |    343       |     1438       |     -174     |
| DDQN, Priority ER            |       96       |    359       |     1625       |     -214     |
| Dueling DQN                  |       272      |    852       |     **_266_**  |     -137     |
| Dueling DQN+TN               |       84       |    445       |     299        |     -106     |
| Dueling DDQN                 |       51       |    329       |     342        |     **_-93_**|
| Dueling DQN, Priority ER     |       178      |    912       |     588        |     -139     |
| Dueling DQN+TN, Priority ER  |       43       |    307       |     1828       |     -175     |
| Dueling DDQN, Priority ER    |       **_39_** |    **_270_** |     2249       |     -137     |

## 2048 game
Training lasted 165 000 episodes with 37 500 000 steps. Average score of random agent is 1011 with highest tile 512. Average score of trained agent is 3700 with highest tile 1024. 

<img src="https://raw.githubusercontent.com/LachubCz/PGuNN/master/images/2048-v0_learning_curve.png" height="300"/>

## Atari games

### Score

|                 | Space Invaders | Breakout | Beam Rider |
|-----------------|:--------------:|:--------:|:----------:|
| Random Player   |       153      |    1.3   |     361    |
| RAM (average)   |       398      |    3.8   |     674    |
| RAM (best)      |       955      |    12    |    1284    |
| Image (average) |       280      |    3.1   |     433    |
| Image (best)    |       830      |     8    |     804    |

### Training stats

|                       | Number of episodes |       Steps      | Time (hours)  |
|-----------------------|:------------------:|:----------------:|:-------------:|
| SpaceInvaders-ram-v0  |       34 500       |    30 000 000    |     260       |
| SpaceInvaders-v0      |       13 500       |    11 500 000    |     160       |
| Breakout-ram-v0       |       74 500       |    30 000 000    |     260       |
| Breakout-v0           |       33 500       |    11 500 000    |     160       |
| BeamRider-ram-v0      |       16 000       |    30 000 000    |     260       |
| BeamRider-v0          |        6 100       |    11 500 000    |     160       |

### Learning curves

<img src="https://raw.githubusercontent.com/LachubCz/PGuNN/master/images/SpaceInvaders-ram-v0_learning_curve.png" height="300"/><img src="https://raw.githubusercontent.com/LachubCz/PGuNN/master/images/SpaceInvaders-v0_learning_curve.png" height="300"/>
<img src="https://raw.githubusercontent.com/LachubCz/PGuNN/master/images/Breakout-ram-v0_learning_curve.png" height="300"/><img src="https://raw.githubusercontent.com/LachubCz/PGuNN/master/images/Breakout-v0_learning_curve.png" height="300"/>
<img src="https://raw.githubusercontent.com/LachubCz/PGuNN/master/images/BeamRider-ram-v0_learning_curve.png" height="300"/><img src="https://raw.githubusercontent.com/LachubCz/PGuNN/master/images/BeamRider-v0_learning_curve.png" height="300"/>
