# C3LLS: Deep Learning Cellular 3D Laboratory Labeling System
Welcome to C3LLS, an open-source semi-automatated approach for researchers with limited AI expertise to create their own model for organoid, nuclear, and cell segmentation! 

C3LLS can 
1. Identify organoid in an image 
2. Count nuclei in an organoid 
3. Count the percentage of dead nuclei in an organoid 
4. Quantify the cell surface marker expression of an independent cell, a cell in an organoid, or an entire organoid
5. Identify invdividual cells in a 3D images

## How Does C3LLS Work?

The C3LLS pipeline has two main parts

1. Human-reviewed auto-segmentation, which allows for quick creation of datasest-specific training data
2. Automated CNN model construction and training to create a dataset specific segmentation model

To review a step-by-step guide, select one of the below options based on your experience level:

1. [I am new to AI](/Documentation/New to AI)
2. [I have experience in AI](/Documentation/AI Experienced)
