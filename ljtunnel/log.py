"""ljtunnel.log -- timestamped console log that can also be written to a file (a run's log)."""
import os
import time

T0 = time.time()
_SINKS = []


def log(*a):
    msg = f"[{time.time() - T0:7.0f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    for f in _SINKS:
        f.write(msg + "\n")
        f.flush()


class tee:
    """with tee('data/logs/run.log'): ...   everything logged inside also goes to the file."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.f = open(self.path, 'a')
        _SINKS.append(self.f)
        return self

    def __exit__(self, *exc):
        _SINKS.remove(self.f)
        self.f.close()
