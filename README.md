# Fictitious Cross-Play

## Install

test on CUDA == 11.6   

``` Bash
   conda create -n marl
   conda activate marl
   pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
   pip install -e . 
   pip install -r requirements.txt
```

## Matrix Games

In folder `matrix_games`: we consider three matrix games: an example game `example`, the Seek-Attack-Defend game `sad`, and the team Rock-Paper-Scissors game `team_rps`. 

Run experiments by executing
```
cd matrix_games/<game_name>
python run.py
```

Then plot results by executing
```
python plot.py
```

## Gridworld Game

In folder `gridworld`: we consider the MAgent Battle 3-vs-3 game.

Run experiments by executing
```
cd gridworld/scripts
./train_magent_<alg_name>.py
```

## Google Reserach Football

It is internal code and cannot be opensourced due to copyright issues.
