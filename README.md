# Self Driving system for Racing games using Pytorch

This repository contains the code to collect data while you drive a car in the game, the keys you press are recorded as you drive the car along with every frame. This data is then used to train a Neural Network to imitate your driving style.

<p align="center">
<img src="NFS_vid_Trimg.gif" width="100%"/>
</p>



`read_screen.py` records the screen and the keys you press and same them as a one-hot array in a numpy array.
`model_01.py` trains the model using the data saved in the numpy array, i used a random not to deep model, feel free to change the model or use a different model.
