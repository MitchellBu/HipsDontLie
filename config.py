'''
A configuration file which contains constants
and hyperparameters used in our project
'''

# Paths
VOCABULARY_FILE = 'vocab.txt'
VIDEOS_DIR = './videos'
ORIGINAL_ALIGNMENTS_DIR = './alignments'
ANNOTATIONS_DIR = './npy_alignments'
LANDMARKS_DIR = './npy_landmarks'

# Preprocessing
PRETRAINED_SHAPE_DETECTOR_PATH = './facial-landmarks-models/shape_predictor_68_face_landmarks.dat'

# DataLoader
SPEAKERS = list(range(6,16))
WINDOW_SIZE = 5
MIN_FRAMES = 70

# Tokenizer
SEQUENCE_IN_MAX_LEN = 80
SEQUENCE_OUT_MAX_LEN = 16

# Landmarks NN hyperparameters
INPUT_DIM = 20*2*WINDOW_SIZE
HIDDEN_DIM = 256
LMARKS_DROPOUT=0.3

# Transformer hyperparametes
TRANSFORMER_D_MODEL = 512
TRANSFORMER_N_HEADS = 4
TRANSFORMER_ENCODER_LAYERS = 2
TRANSFORMER_DECODER_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 2048
TRANSFORMER_DROPOUT = 0.1

# Training
BATCH_SIZE = 64
EPOCHS = 60
VERBOSE_ITERS=100
MAX_RANDOM_FLIP_PROB = 0.3

# trained models
TRANSFORMER_BEST = 'final_models/trained_models_90/transformer'
LANDMARKSNN_BEST = 'final_models/trained_models_90/landmarks'
TRANSFORMER_SAVE = 'trained_models/transformer'
LANDMARKSNN_SAVE = 'trained_models/landmarks'
TRANSFORMER_LOAD = TRANSFORMER_SAVE
LANDMARKSNN_LOAD = LANDMARKSNN_SAVE