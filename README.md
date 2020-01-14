# Self-Driving-system-for-Racing-games-using-Pytorch
This repository contains the code that records your screen when you are playing a racing game and also records the keys you 
press and then uses it to create a training data, which will then be used to train out Convolutional Neural Network written in pytorch.
read_screen.py file record your screen and the keys you press while you are playing your game, it would then save the images as 
numpy arrays and the keys pressed as one-hot in one numpy array.
balancing_data.py file would balance the data so that there are now equal number of all of kthe events.
model.py and model_01.py are the two models that we will be training.
play.py file is where the trained model is used to actually play the game.

