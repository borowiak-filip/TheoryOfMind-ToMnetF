# Introduction 

ToMnetF is a DeepLearning baseline of the Theory of Mind family frameworks like ToMent, ToMnet+, ToMnetN. It inherits from ToMnet, ToMnet+, and ToMnet-N and showcases a new baseline for Theory of Mind experiments using Deep Neural Networks based on convolutional blocks.

The framework introduces grid world enviroment in which agent, tries to collect one of the four different reward objects.
The Neural Netwok - ToMnetF, analyses current location of the player in the given environment and predicts where the agent is going to go and which object is it going to collect (its intention and desire).

The original research paper can be found here: https://arxiv.org/abs/1802.07740

## Project setup
Game is setup for 13x13 grid, however can be run for larger or smaller grids. changing the size will require adjusting some hardocded parameters in neural network and predict.py file.

### Creating data
To create training games run Game_Engine/runAgent.py
each iterations represents a single game. 
set 'no_walls' parameter to determine complexity of the game.
at the end of the file specify folder name in data to which you want data to be saved, e.g., 'demo' will save games to TomnetF/data/demo
Its recommended to create 5000 data samples. For training netowrk will only take a fraction (can be changed).
Sample data is provided in data/Saved Games/Experiment1, Experiment2, demo. Experiment1 is the environment without walls, Experiment2 is the environment with walls, and demo is for testing.

### Train network
To train the model run ToMnetF/ToMnetF_CNN/TrainToMnetF.py
Specify data directory, experiment name, and hyperparamters if other that default.
Folders already containt sample results and setup configuration in setup.txt. The model dict is not included.

### Run Predictions
In ToMnetF/ToMnetF_CNN/Predict.py speicfy which model to run and on which games. The model analyses 1 previous game of length 10, and iteratively predicts new game step at a time.
The simulation will run for each predicitons. Click 'q' to move to the next step.
Some results from the network point just to one direction (potential error here) hence, those are skipped.

## Software Acknowledgement
The baseline uses Labmaze generator https://github.com/google-deepmind/labmaze
The Game Engine is inherited from https://github.com/Nik-Kras/ToMnet-N
Some Network components inherit from https://github.com/yunshiuan/tomnet-project

