import os
import glob
from tqdm import tqdm
import numpy as np


class AnnotationReader:
    '''
    Reads annotations from .align.txt file.
    '''
    def __init__(self, path):
        '''
        Args:
        path: str, path to .align.txt file.
        '''
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
        '''
        Returns annotations as np.array.
        '''
        return np.array(self.labels)


class AnnotationsCompressor:
    '''
    Preprocesses and compresses annotations.
    '''
    def __init__(self, original_path, compressed_path):
        '''
        Args:
        original_path: str, annotations directory.
        compressed_path: str, output directory. 
        '''
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
