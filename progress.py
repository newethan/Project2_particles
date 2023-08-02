import sys
import math

# progress bar
class Bar:
    # initialize progress bar
    def __init__(self, num_signs_to_comp:int=50, marker:str='='):
        if not 100 % num_signs_to_comp == 0:
            raise Exception('Number of signs in a complete bar must divide 100 evenly.')
        self.num_signs_to_comp = num_signs_to_comp # number of signs in progress bar (default 50, one per 2%)
        self.marker = marker # the sign (default '=')
        self.p = 0 # number of signs to print
        self.print() # print bar to initialize
    
    # print the bar
    def print(self):
        # overwrite previous print
        sys.stdout.write('\r')
        # the bar itself
        sys.stdout.write(f"[%-{self.num_signs_to_comp}s] %d%%" % (self.marker*self.p,
                                           (100//self.num_signs_to_comp)*self.p))
        sys.stdout.flush()
    
    # update progress, takes in fraction of progress
    def update(self, frac: float):
        # calculate new number of signs to print
        self.p = int(math.ceil(self.num_signs_to_comp * frac))
        self.print() # print bar
        