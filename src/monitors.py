import torch

from tensorboardX import SummaryWriter

class MonitorTree():

    def __init__(self, pruning, logdir=None):

        super(MonitorTree, self).__init__()

        self.writer = SummaryWriter(logdir)
        self.pruning = pruning

    def write(self, model, it, **metrics):

        if self.pruning:

            self.writer.add_scalars('variables/eta_group', 
                {"linf": torch.norm(model.sparseMAP.eta, p=float('inf')),
                 "l1": torch.norm(model.sparseMAP.eta, p=1), 
                 "l0": torch.norm(model.sparseMAP.eta, p=0),
                 # "eta": model.sparseMAP.eta,
                 }, it)

            self.writer.add_scalars('variables/d_group', 
                {"linf": torch.norm(model.sparseMAP.d, p=float('inf')),
                 "l1": torch.norm(model.sparseMAP.d, p=1), 
                 "l0": torch.norm(model.sparseMAP.d, p=0),
                 # "d": model.sparseMAP.d,
                 }, it)

        for key, item in metrics.items():
            self.writer.add_scalars(key, item, it)
        # self.writer.add_graph(model, x)

    def close(self, logfile="./monitor_scalars.json"):
        self.writer.export_scalars_to_json(logfile)
        self.writer.close()

