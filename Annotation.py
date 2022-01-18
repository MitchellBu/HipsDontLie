import os
import glob
from tqdm import tqdm
import numpy as np


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
        return np.array(self.labels)


class AnnotationsCompressor:
    def __init__(self, original_path, compressed_path):
        self.original_path = original_path
        self.compressed_path = compressed_path
        self.annotations_paths = glob.glob(self.original_path + '/*/*')

        for annotation_path in tqdm(self.annotations_paths):
            ann = AnnotationReader(annotation_path).get_labels()
            new_path = '/'.join(annotation_path.split('/')[-2:])
            file_path = self.compressed_path + '/' + new_path[:-4] + '.npy'
            dir_path = '/'.join(file_path.split('/')[:-1])
            os.makedirs(dir_path, exist_ok=True)
            file = open(file_path, 'w+') # create file
            file.close()
            file = open(file_path, 'wb')
            np.save(file, ann, allow_pickle=True)
            file.close()