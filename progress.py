import sys
import math

class Bar:
    def __init__(self, num_signs_to_comp:int=50, marker:str='='):
        if not 100 % num_signs_to_comp == 0:
            raise Exception('Number of signs in a complete bar must divide 100 evenly.')
        self.num_signs_to_comp = num_signs_to_comp
        self.marker = marker
        self.p = 0
        self.print()
    
    def print(self):
        sys.stdout.write('\r')
        sys.stdout.write(f"[%-{self.num_signs_to_comp}s] %d%%" % (self.marker*self.p,
                                           (100//self.num_signs_to_comp)*self.p))
        sys.stdout.flush()
    
    def update(self, frac: float):
        self.p = int(math.ceil(self.num_signs_to_comp * frac))
        self.print()
        