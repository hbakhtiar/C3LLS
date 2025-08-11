## What is "Training Data"

Many AI models are built using 'training data'. These are examples used to 'teach' the system what patterns to identify. In the case of organoids/microscopy images, this means giving the model paired images, one original, and a second 'labeled' image, where someone has indicated the organoid/cells they want the model to identify. Traditionally, this means manually demarcating the boundaries of organoids/cells for thousands of cells (tons of time!). 

C3LLS takes care of this by allowing you to make your own training data, specific to your dataset. 

## Step By Step

1. Select a group of example images. I would recommend at least one image from each experimental condition, or at least one image from each cell line/patient you would like to count.
2. Place all these images into a folder of your choice
3. In the terminal, enter run_c3lls and select option 1. 'Create Training Data'
4. Depending on what you like to make your model for, select one of the following:

	a. [Model that identifies organoids](#Training-Data-for-Finding-Organoids)
	b. Model that identifies nuclei within organoids
	c. Model that identifies individual cells (not within an organoid)

## Training Data for Finding Organoids
