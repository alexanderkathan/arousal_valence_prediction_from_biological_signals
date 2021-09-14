import torch
from datetime import datetime
from dateutil import tz
from torch.nn import CrossEntropyLoss
from utils import Logger, seed_worker
from dataset import MuSeDataset
from data_parser import load_data
from datetime import datetime
from dateutil import tz
from train import train_model
from eval import evaluate
from model import Model
from loss import CCCLoss
import config


def main():
    print('Loading data ...')
    data = load_data(normalize=False)
    data_loader = {}
    for partition in data.keys():  # one DataLoader for each partition
        set = MuSeDataset(data, partition)
        batch_size = 1024 if partition == 'train' else 1
        shuffle = True if partition == 'train' else False  # shuffle only for train partition
        data_loader[partition] = torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=shuffle, num_workers=4,
                                                             worker_init_fn=seed_worker)

    criterion = CCCLoss()
    score_str = 'CCC'

    if eval_model is None:  # Train and validate for each seed
        val_losses, val_scores, best_model_files, test_scores = [], [], [], []

        model = Model()

        print('=' * 50)
        print('Training model...')

        val_loss, val_score, best_model_file = train_model(model=model, model_path=config.MODEL_FOLDER, data_loader=data_loader, epochs=epochs, lr=lr, use_gpu=use_gpu, criterion=criterion)
        # run evaluation only if test labels are available
        test_loss, test_score = evaluate(model, data_loader['test'], criterion, use_gpu)
        test_scores.append(test_score)
        print(f'[Test CCC]:  {test_score:7.4f}')
        val_losses.append(val_loss)
        val_scores.append(val_score)
        best_model_files.append(best_model_file)

        best_idx = val_scores.index(max(val_scores))  # find best performing seed
        model_file = best_model_files[best_idx]  # best model of all of the seeds

    else:  # Evaluate existing model (No training)
        model_file = eval_model
        model = torch.load(model_file)
        _, valid_score = evaluate(model, data_loader['devel'], criterion)
        print(f'Evaluating {model_file}:')
        print(f'[Val {score_str}]: {valid_score:7.4f}')
        test_score = evaluate(model, data_loader['test'], criterion, use_gpu)
        print(f'[Test {score_str}]: {test_score:7.4f}')


if __name__ == '__main__':
    use_gpu = False
    lr = 0.001
    epochs = 2
    eval_model = None
    main()
    #print(set.__getitem__(1))

