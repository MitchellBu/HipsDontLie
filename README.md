# Lips Don't Lie: FAST Deep Learning model to read lips

<h1 align="center">

  <br>
  <img src="https://github.com/rizkiarm/LipNet/blob/master/assets/lipreading.gif?raw=true" height="300">
</h1>
  <p align="center">
    <a href="https://github.com/MitchellBu">Mitchell Butovsky</a> , <a href="https://github.com/TomBekor">Tom Bekor</a> 
  </p>

Final project as a part of Technion's IEM 097215 "Deep learning for NLP" & EE 046211 "Deep Learning" 🌠.

Implemented in PyTorch :fire:.
* Animation by <a href = https://github.com/rizkiarm> @rizkiarm </a>.

  * [Description](#description-lips)
  * [The Repository](#the-repository-compass)
  * [Running the project](#running-the-project-runner)
    + [Inference](#inference-mag_right)
    + [Training](#training-weight_lifting)
  * [Libraries to Install](#libraries-to-install-books)


## Description :lips:
In this project we combine the [BlazeFace](https://arxiv.org/pdf/1907.05047.pdf "BlazeFace") algorithm and the [transformer architecture](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf "transformer") and acheive near SOTA performance on the [GRID dataset](http://spandh.dcs.shef.ac.uk/gridcorpus/ "GRID dataset") with very fast training and inference.


<h2 align="center">
<img src="https://i.ibb.co/hYH90Zt/Screenshot-from-2022-01-27-16-01-24.png" height=300> <img src="https://i.ibb.co/vVyfxt7/rsz-transformer.png" height=300>
</h2>

## The Repository :compass:
We provide here a short explaination about the structure of this repository:
- ``videos/[speaker_id]`` and ``alignments/[speaker_id]`` contain the raw data from the GRID dataset;
videos and word alignments respectievly.
- ``npy_landmarks`` and ``npy_alignments`` contain the processed videos and alignments. 
The pre-processing is done **automatically** by running ``preprocess.py``. 
The pre-processing mechanisem itself is splitted to the ``Video.py`` which pre-processes the videos and ``Annotation.py`` which pre-processes the alignments. 
- `dataloader.py` contains  data loaders for both training and testing as well as a tokenizer which prepares the data for the transformer. Tokenization is done using ``vocab.txt`` which contains all the possible tokens, as well as ``<pad>``, ``<sos>`` and ``<eos>`` tokens.
-  ``model.py`` contains our architecture, divided to the Transformer and an additional Landmarks Neural Net modules.
- ``run.py`` is the main file of our project. It trains the architecture and then generates predictions on unseen test samples.
-  ``config.py`` containts all the constants and hyper-parameters that are used in the project.
- Finally, ``inference.py`` is used to make predictions using the pre-trained models.
  
## Running The Project :runner:

### Inference :mag_right:
In order to predict the transcript from some given [GRID corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/ "GRID corpus") videos, put them in `` examples/videos`` path.
Then, just run ``inference.py``.
It is possible to change the path/make an inference on a single video by changing the last line of `inference.py`.

**Important: remember to download our pretrained models [here](https://drive.google.com/drive/folders/1-udLFTgkJSemyzciD0PJwOZvXNJ50AZS?usp=sharing "here"), or create them by running ``run.py``**

### Training :weight_lifting:
In order to train the models from scratch:
1. Download the desired videos to train on from the GRID corpus which can be found [here](http://spandh.dcs.shef.ac.uk/gridcorpus/ "here"). Make sure that you download the **high quality videos** and the corresponding word alignments.

2. Put the videos in the project directory according to the following path format: ``videos/[speaker_id]/[video.mpg]``. 

    Put the alignments according to the following path format: ``alignments/[speaker_id]/[alignment.align]``.  

3. Change the ``SPEAKERS`` attribute in the ``config.py`` file to a list containing all the speaker ids to train on. 

4. Run ``preprocess.py``. This might take a while. 

5. Run ``run.py``.

## Libraries to Install :books:

**Before trying to run anything please make sure to install all the packages below.**
|Library         | Command to Run | Minimal Version |
|----------------|----------------|-----------------|
|`NumPy`|  `pip install numpy`| `1.19.5`|
|`matplotlib`|  `pip install matplotlib`| `3.3.4`|
|`PyTorch`|  `pip install torch`| `1.1.10`|
|`Open CV`| `pip install opencv-python `| `4.5.4`|
|`DLib`| `pip install dlib` | `19.22.1`|
|`scikit-learn`|  `pip install scikit-learn`| `0.24.2`|
|`tqdm`| `pip install tqdm`| `4.62.3`|
