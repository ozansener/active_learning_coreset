class ActiveSampler(object):
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last, cur_set):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.cur_set = list(cur_set)

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(self.cur_set[idx])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.cur_set) // self.batch_size
        else:
            return (len(self.cur_set) + self.batch_size - 1) // self.batch_size
