import numpy as np

''' class general for every simulation'''
class Simulation:
    def __init__(self, h_step,  time, time_max, frameforsec, f_int, g, output):
        self.h_step     = h_step
        self.time          = time
        self.time_max   = time_max 
        self.frameforsec= frameforsec
        self.f_int      = f_int
        self.g          = g
        self.output     = output

    def increment_time(self):
        self.time += self.h_step
