import torch

from tensorboardX import SummaryWriter

class MonitorTree():

    def __init__(self, pruning, logdir=None):

        super(MonitorTree, self).__init__()

        self.writer = SummaryWriter(logdir)
        self.pruning = pruning

    def write(self, model, it, report_tree=False, **metrics):

        if report_tree and self.pruning:

            self.writer.add_scalars('variables/d_group', 
                { 
                 "l0": torch.norm(model.latent_tree.d, p=0),
                 "l2": torch.norm(model.latent_tree.d, p=2),
                 # "d": model.latent_tree.d,
                 }, it)

        for key, item in metrics.items():
            self.writer.add_scalars(key, item, it)
        # self.writer.add_graph(model, x)

    def close(self, logfile="./monitor_scalars.json"):
        self.writer.export_scalars_to_json(logfile)
        self.writer.close()

