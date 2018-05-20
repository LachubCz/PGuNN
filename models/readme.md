# Trained Models

## Control theory problems environments

## 2048 game
Training lasted 165 000 episodes with 37 500 000 steps. Average score of random agent is 1011 with highest tile 512. Average score of trained agent is 3700 with highest tile 1024. Model is saved in **2048-v0_basic.h5**.

<img src="https://raw.githubusercontent.com/LachubCz/PGuNN/master/images/2048-v0_learning_curve.png" height="300"/>

## Atari games

|                 | Space Invaders | Breakout | Beam Rider |
|-----------------|:--------------:|:--------:|:----------:|
| Random Player   |       153      |    1.3   |     361    |
| RAM (average)   |       398      |    3.8   |     674    |
| RAM (best)      |       955      |    12    |    1284    |
| Image (average) |       280      |    3.1   |     433    |
| Image (best)    |       830      |     8    |     804    |
