import numpy as np
import glob, os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import config as cfg

def get_vocab_list(vocab_path):
    '''
    returns np array with the vocabulary words.
    Args:
    vocab_path: str, path to vocabulary.txt
    '''
    with open(vocab_path, 'r') as vocab_file:
        lines = vocab_file.readlines()
        lines = [line.strip() for line in lines]
    return np.array(lines)

def sliding_windows(array_size: int, win_size: int) -> np.array:
    ''' Find the indecies correspond to moving windows of size win_size
    Args:
    array_size: int, the size of the array to create windows from,
    win_size: int, the window size,
    Output: np.array of the indecies '''
    indices = []
    for i in range(array_size-win_size+1):
        indices.append(list(range(i, i+win_size)))
    return np.array(indices)

def calc_batch_accuracy(out, batch_targets, vocab_path='vocab.txt', verbose=False):
    '''
    Calculates accuracy per word between predictions and ground truth.
    Args:
    out: torch.tensor, model output.
    batch_targets: torch.tensor, ground truth.
    vocab_path: str, path to vocabulary.
    verbose: bool, if True, print: original sentence, predicted sentence, and accuracy.
    '''
    vocab = get_vocab_list(vocab_path)
    acc_sum = 0
    batch_size = out.shape[0]
    for seq in range(batch_size):
        pred_indices = torch.argmax(out[seq], axis=1).cpu().numpy()
        predictions = vocab[pred_indices]
        original_sentence = vocab[batch_targets.cpu().numpy()[seq]]
        eos_idx = np.where(original_sentence == '<eos>')[0][0] # first <eos> sentence
        original_sentence = original_sentence[:eos_idx+1]
        predictions = predictions[:eos_idx+1]
        if len(predictions) <= 1:
            acc = 0
        else:
            acc = accuracy_score(original_sentence[:-1], predictions[:-1]) # don't calc accuracy on <sos> and <eos>
        acc_sum += acc
        if verbose:
            print(f'Seq #{seq}')
            print(f'Original sentence: {" ".join(original_sentence)}')
            print(f'Predicted sentence: {" ".join(predictions)}')
            print(f'Accuracy: {acc}')
    batch_acc = acc_sum/batch_size
    return batch_acc

def predicted_sentence(out, vocab_path='vocab.txt'):
    '''
    Return the predicted sentence, using probabilities from out.
    Args:
    out: torch.tensor, model output.
    vocab_path: str, path to vocabulary.
    '''
    vocab = get_vocab_list(vocab_path)
    pred_indices = torch.argmax(out[0], axis=1).cpu().numpy()
    predictions = vocab[pred_indices]
    predictions = list(predictions)
    predictions = predictions[:predictions.index('<eos>')+1]
    return predictions
    

def plot_metric(values, label, save=None):
    '''
    Plot metric with legend. Save if necessary.
    Args:
    values: array-like, metric values.
    label: str, metric name, will be presented on legend.
    save: str, path to save the plot.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(values, '-', label=label)
    ax.tick_params(axis='x', colors='w')
    ax.tick_params(axis='y', colors='w')
    plt.legend()
    plt.show()
    if save:
        plt.imsave(save)

def float_array_form_file(file_path):
    '''
    Load array of floats from path.
    Args:
    file_path: str, the path of the array floats.
    '''
    with open(file_path, 'r') as file:
        lines = file.readlines()
        array = lines[0][1:-1]
        array = array.split(',')
        array = [float(v.strip()) for v in array]
    return array

def write_metric(values, output_path):
    '''
    Writes metric values to file.
    Args:
    values: array-like, metric values to be written.
    output_path: str, new file path.
    '''
    with open(output_path, 'w+') as output_file:
        output_file.write(str(values))

def create_train_test_val(speaker_landmarks_path, speaker_annotations_path, train=0.8, test=0.1):
    '''
    After npy directory is created, split to train validation and test subdirectories.
    Args:
    speaker_landmarks_path: str, path to speaker landmarks npys.
    peaker_annotations_path: str, path to speaker anotations npys.
    train=0.8: float, train precentage.
    test=0.1: float, test precentage.
    '''
    # Read paths
    landmarks = sorted(glob.glob(speaker_landmarks_path + '/*.npy'), key=os.path.basename)
    annotations = [speaker_annotations_path + '/' + landmark.split('/')[-1][:-4]+'.a.npy' for landmark in landmarks]

    # Create splits
    data_size = len(landmarks)
    save_order = np.random.permutation(data_size)
    cut_idx = int(train*data_size)
    splits = {}
    splits['train'] = save_order[:cut_idx]
    splits['test'] = save_order[cut_idx:cut_idx+int(test*data_size)]
    cut_idx += int(test*data_size)
    splits['validation'] = save_order[cut_idx:]

    # Create directories
    landmarks_subdirs = [speaker_landmarks_path + '/' + suffix for suffix in {'train', 'test', 'validation'}]
    annotations_subdirs = [speaker_annotations_path + '/' + suffix for suffix in {'train', 'test', 'validation'}]
    for subdir in landmarks_subdirs:
        os.system('mkdir ' + subdir)
    for subdir in annotations_subdirs:
        os.system('mkdir ' + subdir)

    # Move files
    for target_subdir in {'train', 'test', 'validation'}:
        for idx in splits[target_subdir]:
            landmark_file = landmarks[idx]
            os.system('mv ' + landmark_file + ' ' + speaker_landmarks_path + '/' + target_subdir)
            annotation_file = annotations[idx]
            os.system('mv ' + annotation_file + ' ' + speaker_annotations_path + '/' + target_subdir)

    # Delete the remaining .npz files
    os.system(f'rm {speaker_annotations_path}/*.npz ')

# speakers = list(range(29,35))
# print('speakers:',speakers)
# for speaker in speakers:
#     create_train_test_val(f'./npy_landmarks/{speaker}', f'./npy_alignments/{speaker}')