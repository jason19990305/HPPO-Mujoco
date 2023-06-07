import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, args):
        self.clear()
    def clear(self):
        self.s = []
        self.a = []
        self.r = []
        self.s_ = []
        self.a_ = []
        self.dw = []
        self.done = []
        self.weight = []
        self.discount = []

        self.count = 0
        
    def store(self, s, a, r, s_,done,discount):
        self.s.append(s)
        self.a.append(a)
        self.r.append([r])
        self.s_.append(s_)
        self.discount.append([discount])
        self.done.append([done])

        self.count += 1

    def unpack(self):
        return self.s, self.a, self.r, self.s_,self.done , self.weight , self.discount
    
