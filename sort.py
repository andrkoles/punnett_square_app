import numpy as np

def sort_genotype(genotype):
    """
    Sort a genotype giving precendence to capital letters.
    """
    temp = sorted(genotype, key=lambda L: (L.lower(), L))
    return ''.join(temp)

# Vectorize sort_genotype(), for use with NumPy arrays
sort_genotype_vec = np.vectorize(sort_genotype)