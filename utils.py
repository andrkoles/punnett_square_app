import numpy as np

def sort_genotype(genotype):
    """
    Sort a genotype giving precendence to capital letters.
    """
    temp = sorted(genotype, key=lambda L: (L.lower(), L))
    return ''.join(temp)

# Vectorize sort_genotype(), for use with NumPy arrays
sort_genotype_vec = np.vectorize(sort_genotype)

def map_genotypes(genotype):
    """
    Map a genotype to a unique number. 
    """
    is_dominant = [not i.islower() for i in sort_genotype(genotype)]
    loci = [is_dominant[i:i+2] for i in range(0, len(is_dominant), 2)]

    s = 0
    for i, j in zip(loci, range(1, len(loci) + 1)):
        s = s + sum(i) * (3 ** (j - 1))
    return s

map_genotypes_vec = np.vectorize(map_genotypes)

def map_phenotypes(phenotype):
    """
    Map a phenotype to a unique number. 
    """
    # Is dominant, as in map_genotypes function
    is_d= [not i.islower() for i in sort_genotype(phenotype)]

    s = 0
    for i, j in zip(is_d[::2], range(1, len(is_d[::2]) + 1)):
        s = s + i * (2 ** (j - 1))
    # quant = 2 ** (len(phenotype) // 2)
    return s

map_phenotypes_vec = np.vectorize(map_phenotypes)