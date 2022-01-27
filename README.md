# Lips Don't Lie: FAST Deep Learning algorithm to read lips

<h1 align="center">

  <br>
  <img src="https://github.com/rizkiarm/LipNet/blob/master/assets/lipreading.gif?raw=true" height="300">
</h1>
  <p align="center">
    <a href="https://github.com/MitchellBu">Mitchell Butovsky</a> , <a href="https://github.com/TomBekor">Tom Bekor</a> 
  </p>

Final project as a part of Technion's EE 046211 course "Deep Learning"
* Animation by <a href = https://github.com/rizkiarm> @rizkiarm </a>.

  * [Description](#Description)
  * [Running the project](#Running The Project :runner:)
    + [Inference](#Inference :mag_right:)
    + [Training](#Training :weight_lifting:)
  * [Libraries to Install :books:](#Running The Project :books:)


## Description


## Running The Project :runner:

### Inference :mag_right:
In order to predict the transcript from a given [GRID corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/ "GRID corpus") videos, put them in `` examples/videos`` path.
Then, just run ``inference.py``.
It is possible to change the path/make an inference on a single video by changing the last line of `inference.py`.

### Training :weight_lifting:
In order to train the models from scratch:
1. Download the desired videos to train on from the GRID corpus which can be found [here](http://spandh.dcs.shef.ac.uk/gridcorpus/ "here"). Make sure that you download the **high quality videos** and the corresponding word alignments.

2. Put the videos in the project directory according to the following path format: ``videos/[speaker_id]/[video.mpg]``. 

    Put the alignments according to the following path format: ``alignments/[speaker_id]/[alignment.align]``.  

3. Change the ``SPEAKERS`` attribute in the ``config.py`` file to a list containing all the speaker ids to train on. 

4. Run ``preprocess.py``. This might take a while. 

5. Run ``run.py``.

### Libraries to Install :books:

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
