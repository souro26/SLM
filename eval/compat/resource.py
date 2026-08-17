"""
eval/compat/resource.py

Dummy resource module for Windows compatibility with Unix-only evaluation libraries.
"""

import signal

RLIMIT_AS = 9
RLIMIT_DATA = 2
RLIMIT_NOFILE = 7
RLIMIT_CPU = 0
RLIMIT_STACK = 3

if not hasattr(signal, "setitimer"):
    signal.setitimer = lambda *args: None
    signal.ITIMER_REAL = 0


def getrlimit(resource, *args, **kwargs):
    return (-1, -1)


def setrlimit(resource, limits, *args, **kwargs):
    pass
