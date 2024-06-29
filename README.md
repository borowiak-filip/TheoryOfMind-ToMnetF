# Introduction 

ToMnetF is a DeepLearning baseline of the Theory of Mind family frameworks like ToMent, ToMnet+, ToMnetN. It inherits from ToMnet, ToMnet+, and ToMnet-N and showcases a new baseline for Theory of Mind experiments using Deep Neural Networks.

The framework introduces Grid World enviroment in which Agent, tries to collect one of the four different reward objects.
The Neural Netwok, analyses current location of the player in the given environment and predicts where the agent is going to go and which object is it going to collect.

The original research paper can be found here: https://arxiv.org/abs/1802.07740

## Project setup

### Creating data
To create training games run --> /Game_Engine/runAgent.py
each iterations represents a single game. 
set 'no_walls' parameter to determine complexity of the game.
at the end of the file specify folder name in data to which you want data to be saved, e.g., 'demo' will save games to TomnetF/data/demo

### Train network
To train the model run ToMnetF/ToMnetF_CNN/TrainToMnetF.py
Specify data directory, experiment name, and hyperparamters if other that default.

### Run Predictions
In ToMnetF/ToMnetF_CNN/Predict.py speicfy which model to run and which games. The model analyses 1 previous game of length 10, and iteratively predicts new game step at a time.
For each step simulation will run, click 'q' to move to next step.

## Software Acknowledgement
The baseline uses Labmaze generator https://github.com/google-deepmind/labmaze
The Game Engine is inherited from https://github.com/Nik-Kras/ToMnet-N
Some Network components inherit from https://github.com/yunshiuan/tomnet-project

