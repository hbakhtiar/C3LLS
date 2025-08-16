# C3LLS: Deep Learning Cellular 3D Laboratory Labeling System
Welcome to C3LLS, an open-source semi-automatated approach for researchers with limited AI expertise to create their own model for organoid, nuclear, and cell segmentation! 

C3LLS can 
1. Identify organoid in an image 
2. Count nuclei in an organoid 
3. Count the percentage of dead nuclei in an organoid 
4. Quantify the cell surface marker expression of an independent cell, a cell in an organoid, or an entire organoid
5. Identify invdividual cells in a 3D images

## Installation Steps

### 1. Install Python
* You need Python 3.10.2 or newer installed
* To see your version of python, open the Terminal (Mac or Linux) or Command Prompt (Windows) and enter

```bash
python --version
```

or 

```bash
python3 --version
```

If you don't have python installed, you can download it here
[https://www.python.org/downloads/](https://www.python.org/downloads/)

### 2. Create a Folder for the project

Select a place on your computer where you want to put the project

```bash
mkdir myproject
cd myproject
```
You can call it anything you would like.

### 3. Create a Virtual Environment

This will keep the project self-contained so that it doesn't mess with other programs 

```bash
python -m venv venv
```

on some systems you may need to use python3 instead of python

## How Does C3LLS Work?

The C3LLS pipeline has two main parts

1. Human-reviewed auto-segmentation, which allows for quick creation of datasest-specific training data
2. Automated CNN model construction and training to create a dataset specific segmentation model

To review a step-by-step guide, select one of the below options based on your experience level:

1. [I am new to AI](https://github.com/hbakhtiar/C3LLS/tree/main/Documentation/New%20to%20AI)
2. [I have experience in AI](https://github.com/hbakhtiar/C3LLS/tree/main/Documentation/AI%20Experienced)








