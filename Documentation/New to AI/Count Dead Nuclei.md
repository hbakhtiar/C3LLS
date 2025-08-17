# Using C3LLS to Count Dead Nuclei

You first need a trained model that has run predictions. 

If you haven't started making a model, click [here](https://github.com/hbakhtiar/C3LLS/blob/main/Documentation/New%20to%20AI/Create%20Training%20Data.md) to begin

## Counting Dead Nuclei (standard approach)

Once you have trained and predicted model, follow these steps to count dead nuclei

1. Launch run_c3lls main menu
2. Select option 4. Count Dead Nuclei (needs a trained nuclear model)
3. Follow the prompts for the folders containing the original and segmented organoid images
4. The JSON file will save the data with an organoid ID and the number of dead cells identified
5. 'Percent Covered' indicates what percent volume of a nucleus need the death marker to be marked as dead

You can *technically* use this to quantify the number of cells with *any* type of nuclear stain. However, this has only been validated on nuclear death marker stains
