""" Code adapted from https://github.com/Spijkervet/SimCLR/blob/148d5987c90c70003d6611c5f22d8346d4649dbe/modules/nt_xent.py"""

import torch
import torch.nn as nn
import torch.distributed as dist

class GatherLayer(torch.autograd.Function):
    '''Gather tensors from all process, supporting backward propagation.
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
            for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

class NT_Xent(nn.Module):

    def __init__(self, temperature, world_size=1):

        super(NT_Xent, self).__init__()

        self.temperature = temperature
        self.world_size = world_size # number of parellel processes

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.sim_func = nn.CosineSimilarity(dim=2)

    def negative_mask(self, batch_size, world_size):

        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        return mask

    def forward(self, z_i, z_j, batch_size):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.sim_func(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        mask = self.negative_mask(batch_size, self.world_size)
        negative_samples = sim[mask].reshape(N, -1) # matrix of size (N, N - 2)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        
        return loss / N
