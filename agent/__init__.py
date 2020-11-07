import sys

from .mnist import MNISTAgent

def get_agent_cls(name):
    return getattr(sys.modules[__name__], name)
