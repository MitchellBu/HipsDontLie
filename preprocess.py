''' Preprocess the videos and the raw annotatations prior training ''' 
import Video
import Annotation
from utils import create_train_test_val
import config as cfg

speakers = cfg.SPEAKERS

# Preprocess and compress videos and alignments.
Video.LandmarksCompressor(cfg.VIDEOS_DIR, cfg.LANDMARKS_DIR)
# Annotation.AnnotationsCompressor(cfg.ORIGINAL_ALIGNMENTS_DIR, cfg.ANNOTATIONS_DIR)

# For each speaker split landmarks and alignments to train validation and test directories.
print('speakers:',speakers)
for speaker in speakers:
    print(speaker)
    create_train_test_val(f'./npy_landmarks/{speaker}', f'./npy_alignments/{speaker}')
