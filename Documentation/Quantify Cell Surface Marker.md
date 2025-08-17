# Using C3LLS to Quantify Cell Surface Marker Expression

You first need a trained model that has run predictions. 

If you haven't started making a model, click [here](https://github.com/hbakhtiar/C3LLS/blob/main/Documentation/New%20to%20AI/Create%20Training%20Data.md) to begin

## Quantifying Cell Surface Marker Expression

Once you have a trained model that has run predictions follow the below steps 

1. Launch the C3LLS main menu by running run_c3lls
2. Select option 5. Compute Cell Surface Score (needs a trained nuclear model)
3. Follow the prompts, entering the file paths for the original and segmented images
4. The json file will be saved with the nuclear IDs, raw surface marker score for that nucleus, and associated nuclear volume

We chose to normalize surface marker scores by dividing by their nuclear volume. However, you can normalize values in whatever way makes sense to your data. 
