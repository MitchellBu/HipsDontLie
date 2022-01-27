import torch
from torch import nn, optim
import numpy as np
import datetime
from dataloader import DataLoader, Tokenizer
from model import LandmarksNN, Transformer
from utils import calc_batch_accuracy, write_metric, get_vocab_list
import config as cfg
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the vocabulary
vocab_file = open(cfg.VOCABULARY_FILE, 'r')
idx2word = vocab_file.read().splitlines()
vocab_size = len(idx2word)
word2idx = {word:idx for idx, word in enumerate(idx2word)}
vocab_file.close()
vocab = get_vocab_list(cfg.VOCABULARY_FILE)

# Intialize DataLoaders and Tokenizer
train_loader = DataLoader(landmarks_path=cfg.LANDMARKS_DIR, annotations_path=cfg.ANNOTATIONS_DIR,
 vocab=vocab, mode='train', batch_size=cfg.BATCH_SIZE, shuffle=True)

val_loader = DataLoader(landmarks_path=cfg.LANDMARKS_DIR, annotations_path=cfg.ANNOTATIONS_DIR,
 vocab=vocab, mode='validation', batch_size=100, shuffle=True)

test_loader = DataLoader(landmarks_path=cfg.LANDMARKS_DIR, annotations_path=cfg.ANNOTATIONS_DIR,
 vocab=vocab, mode='test', batch_size=100, shuffle=True)

tokenizer = Tokenizer(word2idx)

landmarks_model = LandmarksNN()
landmarks_model.to(device)
transformer = Transformer(vocab_size)
transformer = transformer.to(device)

# Initialize the loss function, optimizers and schedulers
loss_fn=nn.CrossEntropyLoss(ignore_index=word2idx['<pad>']) # Don't compute the loss if GT=<pad>
landmarks_optimizer = optim.Adam(landmarks_model.parameters())
transformer_optimizer = optim.SGD(transformer.parameters(), lr=1e-1)
landmarks_scheduler = ReduceLROnPlateau(landmarks_optimizer, mode='min', factor=0.1, patience=2, verbose=True)
transformer_scheduler = ReduceLROnPlateau(transformer_optimizer, mode='min', factor=0.1, patience=2, verbose=True)

losses = []
all_accs = []
all_loss = []
val_accs = []

flip_prob = 0
training_start = datetime.datetime.now()

for epoch in range(cfg.EPOCHS):
    print(f'<------------------- [ Epoch: {epoch + 1} ] ------------------->')
    # Training phase
    landmarks_model.train()
    transformer.train()
    epoch_accs = []
    tot_loss = 0
    count = 0
    flip_prob += cfg.MAX_RANDOM_FLIP_PROB/cfg.EPOCHS
    for samples, labels in train_loader:
        t1 = datetime.datetime.now()
        # Tokenize the samples and targets
        tokenizer_res = tokenizer.tokenize(samples, labels)
        batch_inputs, batch_targets,\
        batch_in_pad_masks, batch_tgt_pad_masks = tokenizer_res

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        batch_in_pad_masks = batch_in_pad_masks.to(device)
        batch_tgt_pad_masks = batch_tgt_pad_masks.to(device)
        
        # LandmarksNN forward pass
        landmarks_out = landmarks_model.forward(batch_inputs)
        landmarks_out = landmarks_out.to(device)

        # Add <sos> token and remove the last token from each target sequence
        # as an input to the decoder
        sos_idx = word2idx['<sos>']
        sos_vector = torch.Tensor([sos_idx]*batch_targets.size(0)).view(-1,1)
        input_targets = batch_targets[:,:-1].cpu().numpy()
        
        # Flip some of the transformer input target labels to improve generalization
        randomized_targets=np.random.choice(a=np.arange(3,len(vocab)), size=input_targets.shape)
        flip_matrix = np.random.binomial(size=input_targets.shape, n=1, p=flip_prob)
        input_targets[np.where(flip_matrix)] = randomized_targets[np.where(flip_matrix)]
        input_targets = torch.from_numpy(input_targets)
        sos_batch_targets = torch.cat([sos_vector, input_targets], axis=1).type(torch.LongTensor).to(device)

        # Transformer forward pass
        out = transformer.forward(landmarks_out, sos_batch_targets, batch_in_pad_masks, batch_tgt_pad_masks)

        # Backpropagation
        loss = loss_fn(out.view(-1,vocab_size), batch_targets.view(-1))
        landmarks_optimizer.zero_grad()
        transformer_optimizer.zero_grad()
        loss.backward()
        landmarks_optimizer.step()
        transformer_optimizer.step()

        # Metrics
        batch_acc = calc_batch_accuracy(out, batch_targets)
        all_accs.append(batch_acc)
        epoch_accs.append(batch_acc)

        tot_loss += loss.item()
        count += 1
        if count % cfg.VERBOSE_ITERS == 0:
            calc_batch_accuracy(out, batch_targets, verbose=True)
        t2 = datetime.datetime.now()
        iter_time = str(t2-t1).split(':')[-1][:-3]
        print(f'Iteration time: {iter_time} | loss: {loss.item():.3f} | acc: {round(batch_acc*100,2):.2f}%')
        losses.append(tot_loss/count)
        all_loss.append(loss.item())

        # Scheduler
        iters_to_scheduler = 50
        if len(all_loss) % iters_to_scheduler == 0:
            scheduler_loss = sum(all_loss[-iters_to_scheduler:])/iters_to_scheduler
            print(f'scheduler_loss: {scheduler_loss:.3f}')
            landmarks_scheduler.step(scheduler_loss)
            transformer_scheduler.step(scheduler_loss)
        
    print(f'-----------------------------------------------------')
    print(f'                     [ Metrics ]                     ')
    print(f'Average epoch loss: {tot_loss/count:.3f}')
    print(f'Average epoch acc: {sum(epoch_accs)/len(epoch_accs)*100:.2f}%')

    # Evaluation phase
    landmarks_model.eval()
    transformer.eval()
    with torch.no_grad():
        epoch_val_accs = []
        for samples, targets in val_loader:
            # Start with <sos> token
            preds = np.array([['<sos>']]*len(samples))
            # Generate preds in an autoregresive manner
            for pred_idx in range(7):
                tokenizer_res = tokenizer.tokenize(samples, preds)
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

            # Accuracy evaluation
            preds = preds[:,1:-1]
            acc = accuracy_score(np.array(targets).flatten(), preds.flatten())
            epoch_val_accs.append(acc)
        print(f'Validation accuracy: {np.mean(epoch_val_accs)*100:.2f}%\n')

training_end = datetime.datetime.now()
print(f'\nTotal training time: {str(training_end-training_start)[:-3]}')

# Write the evaluated metrics to fiales
write_metric(losses, 'metrics/tain_losses.txt')
write_metric(all_accs, 'metrics/tain_accs.txt')
write_metric(val_accs, 'metrics/val_accs.txt')

# Save the trained models
torch.save(transformer.state_dict(), cfg.TRANSFORMER_SAVE)
torch.save(landmarks_model.state_dict(), cfg.LANDMARKSNN_SAVE)