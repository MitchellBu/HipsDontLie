import numpy as np
import cv2 as cv
import torch
import glob, os
import dlib

np.random.seed(0)

class AnnotationReader:

    def __init__(self, path):
        self.path = path
        annotation_file = open(path, 'r')
        lines = annotation_file.read().splitlines()
        labels = []
        for line in lines:
            attributes = line.split(' ')
            word = attributes[-1] # The word is the last atrribute in each line
            labels.append(word)
        self.labels = labels

    def get_labels(self):
        return self.labels

class DataLoader:

    def __init__(self, videos_path, annotations_path, batch_size, shuffle=True):
        self.video_paths = sorted(glob.glob(videos_path + '/*/*'), key=os.path.basename)
        self.annotation_paths = sorted(glob.glob(annotations_path + '/*/*'), key=os.path.basename)
        self.data_size = len(self.video_paths)
        self.batch_size = batch_size
        if shuffle:
            self.load_order = np.random.permutation(self.data_size)
        else:

            self.load_order = np.arange(self.data_size)
        self.internal_idx = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        self.internal_idx += self.batch_size
        if self.internal_idx > self.data_size:
            raise StopIteration
        batch_samples = []
        batch_targets = []
        for j in range(-self.batch_size, 0):
            file_idx = self.load_order[self.internal_idx + j]
            video_path = self.video_paths[file_idx]
            sample = np.load(video_path, allow_pickle=True)
            sample = torch.Tensor(sample)
            batch_samples.append(sample)
            annotation_path = self.annotation_paths[file_idx]
            target = AnnotationReader(annotation_path).get_labels()
            batch_targets.append(target)
        return batch_samples, batch_targets

class Tokenizer:

    def __init__(self, word2idx, batch_inputs, batch_targets, seq_in_size=80, seq_out_size=20):
        self.inputs = batch_inputs
        self.targets = batch_targets
        self.word2idx = word2idx
        self.seq_in_size = seq_in_size
        self.seq_out_size = seq_out_size
        self.pad_idx = self.word2idx['<pad>']
        self.sos_idx = self.word2idx['<sos>']
        self.eos_idx = self.word2idx['<eos>']

    def tokenize(self):
        batch_inputs = []
        batch_in_pad_masks = []
        batch_tgt_pad_masks = []
        batch_targets = []
        for input, target in zip(self.inputs, self.targets):
            # Pad the input sequence
            input_size = input.size(0)
            pad_size = self.seq_in_size - input_size
            padding = torch.zeros(pad_size, input.size(1))
            input = torch.cat((input, padding), dim=0)
            input = torch.unsqueeze(input, dim=0)
            batch_inputs.append(input)

            # Create the input padding mask
            input_pad_mask = torch.ones(self.seq_in_size, dtype=torch.bool)
            input_pad_mask[:input_size] = 0
            input_pad_mask = torch.unsqueeze(input_pad_mask, dim=0)
            batch_in_pad_masks.append(input_pad_mask)

            # Add <sos> and <eos> tokens to target
            new_target = []
            for word in target:
                word_idx = self.word2idx[word]
                new_target.append(word_idx)
            new_target = [self.sos_idx] + new_target + [self.eos_idx]

            # Create the target mask
            tot_target_size = len(new_target)
            tgt_pad_mask = torch.ones(self.seq_out_size, dtype=torch.bool)
            tgt_pad_mask[:tot_target_size] = 0
            tgt_pad_mask = torch.unsqueeze(tgt_pad_mask, dim=0)
            batch_tgt_pad_masks.append(tgt_pad_mask)

            # Pad the target with <pad>
            pad_size = self.seq_out_size - len(new_target)
            new_target += [self.pad_idx]*pad_size
            new_target = torch.LongTensor(new_target)
            new_target = torch.unsqueeze(new_target, dim=0)
            batch_targets.append(new_target)

        batch_inputs = torch.cat(batch_inputs)
        batch_in_pad_masks = torch.cat(batch_in_pad_masks)
        batch_tgt_pad_masks = torch.cat(batch_tgt_pad_masks)
        batch_targets = torch.cat(batch_targets)
        return batch_inputs, batch_targets, batch_in_pad_masks, batch_tgt_pad_masks
