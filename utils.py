class ExponentialMovingAverage(object):
    """Keyed tracker that maintains an exponential moving average for each key.

    Args:
      keys(list of str): keys to track.
      alpha(float): exponential smoothing factor (higher = smoother).
    """

    def __init__(self, keys, alpha=0.999):
        self._is_first_update = {k: True for k in keys}
        self._alpha = alpha
        self._values = {k: 0 for k in keys}

    def __getitem__(self, key):
        return self._values[key]

    def update(self, key, value):
        if self._is_first_update[key]:
            self._values[key] = value
            self._is_first_update[key] = False
        else:
            self._values[key] = self._values[key] * \
                self._alpha + value*(1.0-self._alpha)


class Averager(object):
    """Keeps track of running averages, for each key."""

    def __init__(self, keys):
        self.values = {k: 0.0 for k in keys}
        self.counts = {k: 0 for k in keys}

    def __getitem__(self, key):
        if self.counts[key] == 0:
            return 0.0
        return self.values[key] * 1.0/self.counts[key]

    def reset(self):
        for k in self.values.keys():
            self.values[k] = 0.0
            self.counts[k] = 0

    def update(self, key, value, count=1):
        self.values[key] += value*count
        self.counts[key] += count


# class Timer(object):
#     """A simple named timer context.
#
#     Usage:
#       with Timer("header_name"):
#         do_sth()
#     """
#
#     def __init__(self, header=""):
#         self.header = header
#         self.time = 0
#
#     def __enter__(self):
#         self.time = time.time()
#
#     def __exit__(self, tpye, value, traceback):
#         elapsed = (time.time()-self.time)*1000
#         print("{}, {:.1f}ms".format(self.header, elapsed))
