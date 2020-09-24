import os, sys
import time
import numpy as np

import node
import torch, torch.nn as nn
import torch.nn.functional as F

from qhoptim.pyt import QHAdam
from tqdm import tqdm

DATA_NAME = "YAHOO"
BATCH_SIZE = 1024
EPOCHS = 100

device = torch.device("cpu")

data = node.Dataset(DATA_NAME, random_state=1337, quantile_transform=True, quantile_noise=1e-3)
in_features = data.X_train.shape[1]

mu, std = data.y_train.mean(), data.y_train.std()
normalize = lambda x: ((x - mu) / std).astype(np.float32)
data.y_train, data.y_valid, data.y_test = map(normalize, [data.y_train, data.y_valid, data.y_test])

print("mean = %.5f, std = %.5f" % (mu, std))

model = nn.Sequential(
    node.DenseBlock(in_features, 128, num_layers=8, tree_dim=3, depth=6, flatten_output=False,
                   choice_function=node.entmax15, bin_function=node.entmoid15),
    node.Lambda(lambda x: x[..., 0].mean(dim=-1)),  # average first channels of every tree
    
)

t0 = time.time()
with torch.no_grad():
    res = model(torch.as_tensor(data.X_train[:5000]))
    # trigger data-aware init

optimizer_params = { 'nus':(0.7, 1.0), 'betas':(0.95, 0.998) }

trainer = node.Trainer(
    model=model, 
    loss_function=F.mse_loss,
    experiment_name=f"node-{DATA_NAME}",
    warm_start=False,
    Optimizer=QHAdam,
    optimizer_params=optimizer_params,
    verbose=False,
    n_last_checkpoints=5
)

loss_history, mse_history = [], []
best_mse = float('inf')
best_step_mse = 0
report_frequency = 100
early_stopping_rounds = 5000

for batch in tqdm(node.iterate_minibatches(data.X_train, data.y_train, batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS)):
    metrics = trainer.train_on_batch(*batch, device=device)
    
    loss_history.append(metrics['loss'])

    if trainer.step % report_frequency == 0:
        trainer.save_checkpoint()
        trainer.average_checkpoints(out_tag='avg')
        trainer.load_checkpoint(tag='avg')
        mse = trainer.evaluate_mse(
            data.X_valid, data.y_valid, device=device, batch_size=BATCH_SIZE*2)

        if mse < best_mse:
            best_mse = mse
            best_step_mse = trainer.step
            trainer.save_checkpoint(tag='best_mse')
        mse_history.append(mse)
        
        trainer.load_checkpoint()  # last
        trainer.remove_old_temp_checkpoints()

        print("Loss %.5f" % (metrics['loss']))
        print("Val MSE: %0.5f" % (mse))

        if trainer.step > best_step_mse + early_stopping_rounds:          
            break

        print("Best step: ", best_step_mse)
        print("Best Val MSE: %0.5f" % (best_mse))

t1 = time.time()

print("Best step: ", best_step_mse)
print("Best Val MSE: %0.5f" % (best_mse * std ** 2))
print(f"Training time: {t1 - t0}s")

t2 = time.time()

trainer.load_checkpoint(tag='best_mse')
mse = trainer.evaluate_mse(data.X_test, data.y_test, device=device, batch_size=BATCH_SIZE*2)

t3 = time.time()
print('Best step: ', trainer.step)
print("Test MSE: %0.5f" % (mse * std ** 2))
print(f"Inference time: {t3 - t2}s")
