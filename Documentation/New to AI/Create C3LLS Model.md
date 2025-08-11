## Creating a C3LLS Model

If you have not already made training data for your model, click [here](https://github.com/hbakhtiar/C3LLS/blob/main/Documentation/New%20to%20AI/Create%20Training%20Data.md)

There are three parts to building a C3LLS model. Preprocessing, training, and predicting. 

## Pre-processing

1. Once you have reviewed all your training images, place all the original images and the labeled images in one folder.
2. From the run_c3lls main menu select 2. "Make a Model (needs training data)"
3. Select 1. "Create New Model"
4. Enter a base directory - this is where all of the model data will be stored
5. Set ID - This is a number used to indicate the model
6. Set name - This is a unique name for the model
7. Number of processors - the number of processors on your server you would like to use for preprocessing the data.

The program should then run preprocessing on your images. Once completed it should return to the menu. 

## Training

Once the model has finished preprocessing

1. Select 'Train previously preprocessed model'
2. Enter the same set name and set id as above
3. Follow the remaining prompts
4. If uncertain on the 'fold' simply enter 0.

Congrats! The model is writing and learning on it's own. Depending on the size of your data/server this might take several hours to a few days to complete. 

## Predicting

