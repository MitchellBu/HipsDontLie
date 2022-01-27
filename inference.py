''' Perform inference on a single .mpg video or a folder of .mpg videos '''
import glob
import torch
import datetime
from dataloader import InferenceLoader, Tokenizer
from model import LandmarksNN, Transformer
from utils import get_vocab_list
import config as cfg
import numpy as np
import os

print('Loading models ...')
from Video import Video

landmarksNN_model_path = cfg.LANDMARKSNN_LOAD
transformer_model_path = cfg.TRANSFORMER_LOAD

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the vocabulary
vocab_file = open(cfg.VOCABULARY_FILE, 'r')
idx2word = vocab_file.read().splitlines()
vocab_size = len(idx2word)
word2idx = {word:idx for idx, word in enumerate(idx2word)}
vocab_file.close()
vocab = get_vocab_list(cfg.VOCABULARY_FILE)

# Intialize DataLoader and Tokenizer
tokenizer = Tokenizer(word2idx)

# Load saved models
landmarks_model = LandmarksNN()
landmarks_model.load_state_dict(torch.load(landmarksNN_model_path))
landmarks_model.to(device)

transformer = Transformer(vocab_size)
transformer.load_state_dict(torch.load(transformer_model_path))
transformer.to(device)

landmarks_model.eval()
transformer.eval()

print('Done!')

def _single_infer(video_path):
    t_start = datetime.datetime.now()
    vid = Video(video_path)
    vid.find_landmarks()
    data = vid.mouth_lmrks
    sample = InferenceLoader(data).get()
    with torch.no_grad():
        # Start with <sos> token
        preds = np.array([['<sos>']])
        # Generate preds in an autoregresive manner
        for pred_idx in range(7):
            tokenizer_res = tokenizer.tokenize([sample], preds)
            batch_inputs, batch_targets,\
            batch_in_pad_masks, batch_tgt_pad_masks = tokenizer_res

            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_in_pad_masks = batch_in_pad_masks.to(device)
            batch_tgt_pad_masks = batch_tgt_pad_masks.to(device)

            # LandmarksNN forward pass
            landmarks_out = landmarks_model.forward(batch_inputs)
            landmarks_out = landmarks_out.to(device)

            # Transformer forward pass
            out = transformer.forward(landmarks_out, batch_targets, batch_in_pad_masks, batch_tgt_pad_masks)
            preds_tokens = torch.argmax(out[:,pred_idx,:], dim=1).cpu()
            preds_tokens = np.array(preds_tokens)

            preds_words = np.array(list(map(lambda x: idx2word[x], preds_tokens)))
            preds = np.concatenate([preds, preds_words.reshape(-1,1)], axis=1)
    t_end = datetime.datetime.now()
    fname = video_path.split('/')[-1]
    print(f'Video: {fname} | Prediction: {preds[0,1:-1]} | runtime: {t_end-t_start}')


def infer(video_or_folder_path):
    if os.path.isdir(video_or_folder_path):
        videos_paths = glob.glob(video_or_folder_path + '/*.mpg')
        for video_path in videos_paths:
            _single_infer(video_path)
    else:
        _single_infer(video_or_folder_path)
   

infer('/home/student/Tom/LipsDontLie/examples/videos')