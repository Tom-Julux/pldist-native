import os
import numpy as np
from numba.pycc import CC

cc = CC('pldist')

output_dir = '{}'.format(os.path.abspath('.'))
cc.output_dir = output_dir

@cc.export('pldist', 'f8(f8[:],f8[:],f8[:])')
def pldist(point, start, end):
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)

    return np.abs(np.linalg.norm(np.cross(end - start, start - point))) /\
        np.linalg.norm(end - start)

if __name__ == "__main__":
    cc.compile()