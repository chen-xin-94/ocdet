from enum import Enum
import torch
from collections import defaultdict


# https://github.com/pytorch/examples/blob/2c57b0011a096aef83da3b5265a14db2f80cb124/imagenet/main.py#L279-L282
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    # def all_reduce(self):

    #     import torch.distributed as dist

    #     if torch.cuda.is_available():
    #         device = torch.device("cuda")
    #     elif torch.backends.mps.is_available():
    #         device = torch.device("mps")
    #     else:
    #         device = torch.device("cpu")
    #     total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
    #     dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
    #     self.sum, self.count = total.tolist()
    #     self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        # entries = [" *"]
        entries = [self.prefix]
        entries += [meter.summary() for meter in self.meters]
        # print(" ".join(entries))
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class dotdict(dict):
    """dot.key access to dictionary attributes with default None for missing keys"""

    def __getattr__(self, key):
        return self.get(key, None)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def make_divisible(v, divisor, min_value=None):
    """Ensures that all layers have a channel number that is divisible by divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
