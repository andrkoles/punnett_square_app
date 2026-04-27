import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from sort import sort_genotype, sort_genotype_vec

class PunnettSquare:
    def __init__(self, parent_a='Aa', parent_b='Aa'):
        self.parent_a = sort_genotype(parent_a)
        self.parent_b = sort_genotype(parent_b)


    def __repr__(self):
        return f'PunnettSquare({self.parent_a!r}, {self.parent_b!r})'
    
    def _n_traits(self):
        """
        Number of traits in the cross.
        """
        return len(self.parent_a) // 2
    
    def _size(self):
        """
        Number of Punnett square rows (or columns).
        """
        return 2 ** self._n_traits()
    

    def parent_a_gam(self):
        """
        Gametes from first parent.
        """
        length = len(self.parent_a)
        parts = [self.parent_a[i:i+2] for i in range(0, length, 2)]
        parts_joined = [''.join(i) for i in product(*parts)]
        return np.array(parts_joined)
    
    def parent_b_gam(self):
        """
        Gametes from second parent.
        """
        length = len(self.parent_b)
        parts = [self.parent_b[i:i+2] for i in range(0, length, 2)]
        parts_joined = [''.join(i) for i in product(*parts)]
        return np.array(parts_joined)
    
    def genotypes(self):
        """
        Calculates genotypes of the cross between the two parents. Takes
        advantage of NumPy broadcasting rules.
        """
        array = self.parent_a_gam().reshape(1, -1) + self.parent_b_gam().reshape(-1, 1)
        return sort_genotype_vec(array)
    
    def gen_encoded(self):
        """
        Genotypes encoded. Assigns a number to each unique genotype which
        facilitates plotting.
        """
        keys = np.unique(self.genotypes())
        arr = np.arange(len(keys))
        mapping = dict([(k, v) for k, v in zip(keys, arr)])
        encoded = np.array([mapping[item] for item in self.genotypes().flatten()])
        return encoded.reshape(self._size(), self._size())
    
    def freq_table(self):
        """
        Frequency table for the genotypes in the cross.
        """
        unique, counts = np.unique(self.genotypes(), return_counts=True)
        # Number of squares, or number of genotypes in the cross
        n_squares = self._size() ** 2
        d = {
            "Genotype": unique, 
            "Frequency": counts,
            "Percent": (counts / n_squares) * 100
        }
        freqs = pd.DataFrame(data=d, index=None)
        return freqs.sort_values(by=['Frequency'], ascending=False)
    
    def plot_square(self):
        """
        Plots the Punnett Square using Matplotlib.
        """
        fig, ax = plt.subplots()
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        fig.dpi = 120
        ax.imshow(self.gen_encoded(), cmap='rainbow')
        ax.set_xticks(ticks=(range(self._size())), labels=self.parent_a_gam())
        ax.set_yticks(ticks=range(self._size()), labels=self.parent_b_gam())
        for i in range(self._size()):
            for j in range(self._size()):
                plt.text(i, j, self.genotypes()[j, i], fontsize=50 / self._size(),
                         ha='center', color='black')
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(self.genotypes().shape[1]+1)-.5, minor=True)
        ax.tick_params(axis='both', which='major', labelsize=50 / self._size())
        ax.set_yticks(np.arange(self.genotypes().shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        plt.tight_layout()
        ax.tick_params(which="minor", bottom=False, left=False)
        # plt.show()

    def _xy_gap(self):
        """
        Controls xgap and ygap options in Plotly figure.
        """
        match self._size():
            case 2:
                return 2
            case 4:
                return 1.5
            case 8:
                return 0.9
            case 16:
                return 0.6
            case 32:
                return 0.4
            case _:
                return 0.2

    def plotly_square(self, width=400, fontsize=15):
        """
        Plots the Punnett Square using Plotly.
        """
        fig = go.Figure(data=go.Heatmap(
            z=np.flipud(self.gen_encoded()),
            text=np.flipud(self.genotypes()),
            texttemplate="%{text}",
            xgap=self._xy_gap(),
            ygap=self._xy_gap(),
            colorscale='portland',
         ))
        fig.update_layout(
            xaxis = dict(
                tickmode='array',
                tickvals=list(range(self._size())),
                ticktext=self.parent_a_gam(),
                side='top',
            ),
            yaxis = dict(
                tickmode='array',
                tickvals=list(range(self._size())),
                ticktext=np.flipud(self.parent_b_gam()),
            ),
            font=dict(
                size=fontsize
            ),
            autosize=False,
            width=width,
            height=width,
        )
        fig.update_traces(
            showscale=False
        )
        return fig