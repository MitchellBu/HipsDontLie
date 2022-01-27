# Lips Don't Lie

<h1 align="center">

Lip reading using Deep Learning
  <br>
  <img src="https://github.com/rizkiarm/LipNet/blob/master/assets/lipreading.gif?raw=true" height="300">
</h1>
  <p align="center">
    <a href="https://github.com/MitchellBu">Mitchell Butovsky</a> , <a href="https://github.com/TomBekor">Tom Bekor</a> 
  </p>

Final project as a part of Technion's EE 046211 course "Deep Learning"
* Animation by <a href = https://github.com/rizkiarm> @rizkiarm </a>.

  * [Description](#agenda)
  * [Running the project](#running-the-project)
    + [Training](#training)
    + [Running Locally](#running-locally)
  * [Installation Instructions](#installation-instructions)
    + [Libraries to Install](#libraries-to-install)


## Description


## Running The Project
You can view the tutorials online or download and run locally.

### Training :weight_lifting:
In order to train the models from scratch:
1. Download the desired videos to train on from the GRID corpus which can be found [here](http://spandh.dcs.shef.ac.uk/gridcorpus/ "here"). Make sure that you download the **high quality videos** and the corresponding word alignments.

2. Put the videos in the project directory according to the following path format: ``videos/[speaker_id]/[video.mpg]``. 

    Put the alignments according to the following path format: ``alignments/[speaker_id]/[alignment.align]``.  

3. Change the ``SPEAKERS`` attribute in the ``config.py`` file to a list containing all the speaker ids to train on. 

4. Run ``preprocess.py``. This might take a while. 

5. Run ``run.py``.

Note: creating the Binder instance takes about ~5-10 minutes, so be patient

### Running Locally

Press "Download ZIP" under the green button `Clone or download` or use `git` to clone the repository using the 
following command: `git clone https://github.com/taldatech/ee046211-deep-learning.git` (in cmd/PowerShell in Windows or in the Terminal in Linux/Mac)

Open the folder in Jupyter Notebook (it is recommended to use Anaconda). Installation instructions can be found in `Setting Up The Working Environment.pdf`.


## Installation Instructions

1. Get Anaconda with Python 3, follow the instructions according to your OS (Windows/Mac/Linux) at: https://www.anaconda.com/products/individual
2. Install the basic packages using the provided `environment.yml` file by running: `conda env create -f environment.yml` which will create a new conda environment named `deep_learn`. If you did this, you will only need to install PyTorch, see the table below.
3. Alternatively, you can create a new environment for the course and install packages from scratch:
In Windows open `Anaconda Prompt` from the start menu, in Mac/Linux open the terminal and run `conda create --name deep_learn`. Full guide at https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands
4. To activate the environment, open the terminal (or `Anaconda Prompt` in Windows) and run `conda activate deep_learn`
5. Install the required libraries according to the table below (to search for a specific library and the corresponding command you can also look at https://anaconda.org/)

### Libraries to Install

|Library         | Command to Run |
|----------------|---------|
|`Jupyter Notebook`|  `conda install -c conda-forge notebook`|
|`numpy`|  `conda install -c conda-forge numpy`|
|`matplotlib`|  `conda install -c conda-forge matplotlib`|
|`pandas`|  `conda install -c conda-forge pandas`|
|`scipy`| `conda install -c anaconda scipy `|
|`scikit-learn`|  `conda install -c conda-forge scikit-learn`|
|`seaborn`|  `conda install -c conda-forge seaborn`|
|`tqdm`| `conda install -c conda-forge tqdm`|
|`opencv`| `conda install -c conda-forge opencv`|
|`optuna`| `pip install optuna`|
|`pytorch` (cpu)| `conda install pytorch torchvision torchaudio cpuonly -c pytorch` |
|`pytorch` (gpu)| `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch` |
|`torchtext`| `conda install -c pytorch torchtext`|


5. To open the notebooks, open Ananconda Navigator or run `jupyter notebook` in the terminal (or `Anaconda Prompt` in Windows) while the `deep_learn` environment is activated.
