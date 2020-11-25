import hydra

import numpy as np

from pathlib import Path

from torch.nn import BCELoss, MSELoss
from torch.utils.data import DataLoader

from qhoptim.pyt import QHAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.LT_models import LTBinaryClassifier, LTRegressor
from src.monitors import MonitorTree
from src.optimization import train_stochastic, evaluate
from src.datasets import Dataset, TorchDataset
from src.utils import deterministic

import time

clf_datasets = ["CLICK", "HIGGS", "MUSHROOMS"]

@hydra.main(config_path='config/tuned-click.yaml')
def main(cfg):

    cwd = Path(hydra.utils.get_original_cwd())

    if cfg.DATA_NAME in clf_datasets:
        data = Dataset(cfg.DATA_NAME, data_path=cwd / "DATA", normalize=True, seed=459107)
        print('classes', np.unique(data.y_test))

    else:
        data = Dataset(cfg.DATA_NAME, data_path=cwd / "DATA", normalize=True, normalize_target=True)
        in_features = data.X_train.shape[1]
        out_features = 1
        print("target mean = %.5f, std = %.5f" % (data.mean_y, data.std_y))

    state = {
        'batch-size': cfg.training.BATCH_SIZE,
        'learning-rate': cfg.training.LR,
        'dataset': cfg.DATA_NAME,
        'reg': cfg.training.REG,
    }  

    pruning = cfg.training.REG > 0

    test_losses, train_times, test_times = [], [], []
    for SEED in cfg.training.SEEDS:
        deterministic(SEED)

        save_dir = cwd / "results/{cfg.DATA_NAME}/seed={cfg.training.SEED}/"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("results will be saved in:", save_dir.resolve())

        trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=cfg.training.BATCH_SIZE, num_workers=cfg.training.WORKERS, shuffle=True)
        valloader = DataLoader(TorchDataset(data.X_valid, data.y_valid),  num_workers=cfg.training.WORKERS, batch_size=cfg.training.BATCH_SIZE*2, shuffle=False)
        testloader = DataLoader(TorchDataset(data.X_test, data.y_test),  num_workers=cfg.training.WORKERS, batch_size=cfg.training.BATCH_SIZE*2, shuffle=False)

        if cfg.DATA_NAME in clf_datasets:

            model_cls = LTBinaryClassifier
            model = LTBinaryClassifier(cfg.model.TREE_DEPTH, data.X_train.shape[1], pruned=pruning, linear=cfg.model.LINEAR, layers=cfg.model.MLP_LAYERS, dropout=cfg.model.DROPOUT)

            state['loss-function'] = 'BCE'
            # init loss
            criterion = BCELoss(reduction="sum")
            # evaluation criterion => error rate
            eval_criterion = lambda x, y: (x != y).sum()

        else:
            
            model_cls = LTRegressor        
            model = LTRegressor(cfg.model.TREE_DEPTH, in_features, out_features, pruned=pruning, linear=False, layers=cfg.model.MLP_LAYERS, dropout=cfg.model.DROPOUT)

            # init loss
            criterion = MSELoss(reduction="sum")
            eval_criterion = lambda x, y: criterion(x, y) * data.std_y**2
            state['loss-function'] = 'MSE'

        # init optimizer
        optimizer = QHAdam(model.parameters(), lr=cfg.training.LR, nus=(0.7, 1.0), betas=(0.995, 0.998))

        # init learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)
        
        # init train-eval monitoring 
        monitor = MonitorTree(pruning, save_dir)

        best_val_loss = float("inf")
        best_e = -1
        no_improv = 0
        t0 = time.time()
        for e in range(cfg.training.EPOCHS):
            train_stochastic(trainloader, model, optimizer, criterion, epoch=e, reg=cfg.training.REG, monitor=monitor)

            val_loss = evaluate(valloader, model, {'val': eval_criterion}, epoch=e, monitor=monitor)
            print("Epoch %i: validation loss = %f\n" % (e, val_loss["val"]))
            no_improv += 1

            if val_loss["val"] <= best_val_loss:
                best_val_loss = val_loss["val"]
                best_e = e
                no_improv = 0
                model_cls.save_model(model, optimizer, state, save_dir, epoch=e, val_loss=best_val_loss)

            if no_improv == cfg.training.EPOCHS // 10:
                break

            # reduce learning rate if needed
            lr_scheduler.step(val_loss['val'])
            monitor.write(model, e, train={"lr": optimizer.param_groups[0]['lr']})

        t1 = time.time()
        monitor.close()
        print("best validation error rate (epoch {}): {}\n".format(best_e, best_val_loss))

        model = model_cls.load_model(save_dir)
        t2 = time.time()
        test_loss = evaluate(testloader, model, {'val': eval_criterion})
        print("test error rate (model of epoch {}): {}\n".format(best_e, test_loss['val']))
        t3 = time.time()
        test_losses.append(test_loss['val'])
        train_times.append(t1 - t0)
        test_times.append(t3 - t2)

    print(np.mean(test_losses), np.std(test_losses))
    np.save(save_dir / '../test-losses.npy', test_losses)
    print("Avg train time", np.mean(train_times))
    print("Avg test time", np.mean(test_times))

if __name__ == "__main__":
    main()
