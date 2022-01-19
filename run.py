import torch
from torch import nn, optim
import datetime
from dataloader import DataLoader, Tokenizer
from model import Encoder, Transformer
from utils import calc_batch_accuracy

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

encoder_optimizer = optim.Adam(encoder.parameters())
transformer_optimizer = optim.SGD(transformer.parameters(), lr=1e-2)
loss_fn=nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])

loader = DataLoader(videos_path=VIDEOS_DIR, annotations_path=ANNOTATIONS_DIR, initial_batch_size=1, shuffle=True)

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
        if count % 200 == 0:
            calc_batch_accuracy(out, batch_targets, verbose=True)
            if count < 1000:
                print('\n!!!!!!!!!!!!!! Increasing difficulty !!!!!!!!!!!!!!')
                loader.increase_difficulty()
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