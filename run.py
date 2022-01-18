from base64 import encode
import torch
from torch import nn, optim
from dataloader import DataLoader, Tokenizer
from model import Encoder, Transformer
import datetime
import numpy as np
from sklearn.metrics import accuracy_score

VOCABULARY_FILE = 'vocab.txt'
VIDEOS_DIR = 'npy_videos'
ANNOTATIONS_DIR = 'npy_alignment'
vocab_file = open(VOCABULARY_FILE, 'r')
idx2word = vocab_file.read().splitlines()
word2idx = {word:idx for idx, word in enumerate(idx2word)}
vocab_file.close()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

encoder = Encoder()
encoder = encoder.to(device)
transformer = Transformer(len(idx2word))
transformer = transformer.to(device)

tokenizer = Tokenizer(word2idx)

encoder_optimizer = optim.Adam(transformer.parameters())
transformer_optimizer = optim.Adam(transformer.parameters())
loss_fn=nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])

loader = DataLoader(videos_path=VIDEOS_DIR, annotations_path=ANNOTATIONS_DIR, batch_size=16, shuffle=True)

def get_vocab_list(vocab_path):
    with open(vocab_path, 'r') as vocab_file:
        lines = vocab_file.readlines()
        lines = [line.strip() for line in lines]
    return np.array(lines)

def calc_batch_accuracy(out, batch_targets, vocab_path='vocab.txt', verbose=False):
    vocab = get_vocab_list(vocab_path)
    acc_sum = 0
    batch_size = out.shape[0]
    for seq in range(batch_size):
        pred_indices = torch.argmax(out[seq], axis=1).cpu().numpy()
        predictions = vocab[pred_indices]
        original_sentence = vocab[batch_targets.cpu().numpy()[seq]]
        acc = accuracy_score(original_sentence, predictions)
        acc_sum += acc
        if verbose:
            print(f'Seq #{seq}')
            print(f'Original sentence: {" ".join(original_sentence)}')
            print(f'Predicted sentence: {" ".join(predictions)}')
            print(f'Accuracy: {acc}')
    batch_acc = acc_sum/batch_size
    return batch_acc

def plot_accuracy(accs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(accs, '-', label='accuracy')
    ax.tick_params(axis='x', colors='w')
    ax.tick_params(axis='y', colors='w')
    plt.legend()
    plt.show()

epochs = 1
losses = []
accs = []

for epoch in range(epochs):
    print(f'<--------------------- Epoch: {epoch + 1} --------------------->')
    tot_loss = 0
    count = 0
    for samples, labels in loader:
        t1 = datetime.datetime.now()
        tokenizer_res = tokenizer.tokenize(samples, labels)
        batch_inputs, batch_targets,\
        batch_in_pad_masks, batch_tgt_pad_masks = tokenizer_res

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        batch_in_pad_masks = batch_in_pad_masks.to(device)
        batch_tgt_pad_masks = batch_tgt_pad_masks.to(device)
        
        encoder_out = encoder.forward(batch_inputs)
        encoder_out = encoder_out.to(device)
        out = transformer.forward(encoder_out, batch_targets, batch_in_pad_masks, batch_tgt_pad_masks)
        
        loss = loss_fn(out.view(-1,len(idx2word)), batch_targets.view(-1))
        encoder_optimizer.zero_grad()
        transformer_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        transformer_optimizer.step()

        # Metrics:
        batch_acc = calc_batch_accuracy(out, batch_targets)
        accs.append(batch_acc)

        tot_loss += loss.item()
        count += 1
        t2 = datetime.datetime.now()
        print(f'Iteration time: {t2-t1} | loss: {loss.item():.3f} | acc: {round(batch_acc*100,2)}%')
        losses.append(tot_loss/count)
        
    print(f'----- Epoch Metrics -----')
    print(f'Average epoch loss: {tot_loss/count:.3f}')
    print(f'Average epoch acc: {sum(accs)/len(accs)}')
    print(f'')

output_file = open('output.txt', 'w+')
output_file.write(str(losses))
output_file.close()
