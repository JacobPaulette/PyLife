import numpy as np
from scipy import ndimage
import re

class Life:
    """The controller class for PyLife.
    
    Parameters:

    rule_string : Determines rules for game, e.g. "B3/S23" "B36/S23", see 
        http://www.conwaylife.com/wiki/Rules#Rules for more info.
    matrix : Numpy 2d array containing 1 or 0, representing live and dead
        cells. 
    wrapping : Boolean, True if matrix wraps, False if not.

    This object runs the game by updating the matrix to next generation
    with the update_matrix method and returning the matrix with the
    view_matrix method.
    """

    def __init__(self, matrices, wrapping):
        """
        self.matrix_A = Matrix(matrix_A, wrapping)
        self.matrix_B = Matrix(matrix_B, wrapping)
        self.matrix_C = Matrix(matrix_C, wrapping)
        """
        self.matrices = [Matrix(i, wrapping) for i in matrices]
        self.generation = 0
        self.size = len(self.matrices)

    def view_matrix(self, i):
        """Return matrix as a numpy 2d array."""
        return self.matrices[i%(len(self.matrices))].view_matrix()

    def get_generation(self):
        """Return current generation of the game."""
        return self.generation

    def get_size(self):
        return self.size

    def update_matrix(self):
        """Cycle to next generation of the game.

        Generates a matrix of neighbors for each element.
        Calculates new array from current matrix and neighbor
        matrix.  Sets the new array to the matrix object and 
        increments self.generation.
        """
        """
        neighbors_A = self.matrix_A.find_neighbors()
        neighbors_B = self.matrix_B.find_neighbors()
        neighbors_C = self.matrix_C.find_neighbors()
        """
        n_list = [i.find_neighbors() for i in self.matrices]
        sz = len(self.matrices)
        slicelen = int((sz-1)/2)
        for i in range(sz):
            bs = (i+1)%sz
            be = ((i+slicelen)%sz) + 1
            ss = (i+slicelen+1)%sz
            se = ((i+(2*slicelen))%sz) + 1
            self.matrices[i].epoch(n_list[ss:se], n_list[bs:be])
        self.generation += 1



class Matrix:
    """Stores current matrix. Detects if matrix is malformed. Returns matrix
    with view_matrix method. Updates matrix with set_matrix method. Returns 
    a neighbor matrix with the find_neighbors method.
    """

    FILTER = np.array([
            [1,1,1],
            [1,0,1],
            [1,1,1]]) #Filter constant for convolution in find_neighbors

    def __init__(self, matrix, wrapping):
        self.matrix = matrix
        self.wrapping = wrapping
        self.rows = self._check_rows()
        self.rows = len(matrix)
        self.mode = self._check_mode(wrapping)

    @staticmethod
    def _check_mode(wrapping):
        """Return 'wrap' if True, 'constant' otherwise.
    
        Used for ndimage.convolve method.
        """
        if wrapping:
            return 'wrap'
        else:
            return 'constant'
            
    def _check_rows(self):
        """Check if each row in matrix is of equal length."""
        matx = self.matrix
        samebool = all(len(i) == len(matx[0]) for i in matx)
        if samebool:
            return len(matx[0])
        else:
            print("ERROR: INVALID MATRIX")
            sys.exit(0)

    def set_matrix(self, new_matrix):
        """Set new_matrix as self.matrix."""
        self.matrix = new_matrix

    def view_matrix(self):
        """Return current matrix"""
        return self.matrix

    def find_neighbors(self):
        """Return a neighbor matrix of equal dimensions to self.matrix, 
        where each element of the neighbor matrix is equal to the sum of 
        the 8 adjacent cells around the equivalent self.matrix element.
        
        Lookup ndimage.convolve for details.
        """
        return ndimage.convolve(self.matrix, Matrix.FILTER, mode=self.mode)


    def epoch(self, n_survivals, n_births):
        """
        og_shape = n_matrix.shape # find dimensions of neighbor_matrix
        length = og_shape[0] * og_shape[1] # multiply dimensions
        flat_nmatx = n_matrix.reshape(1,length)[0] # flatten n_matrix
        flat_matx = matrix.reshape(1, length)[0] # flatten matrix
        survive_mask = np.in1d(flat_nmatx, self.rules[1]) # see np.in1d
        birth_mask = np.in1d(flat_nmatx, self.rules[0]) 
        f = (flat_matx & survive_mask) | birth_mask # idk why this works
        return f.reshape(og_shape) # og_shape == original dimensions
        """
        n_matrix = self.find_neighbors()
        survival_mask = np.zeros(self.matrix.shape)
        for i in n_survivals:
            survival_mask = np.logical_or(survival_mask, (n_matrix == i))
        birth_mask = np.zeros(self.matrix.shape)
        for i in n_births:
            birth_mask = np.logical_or(birth_mask, (n_matrix == i))
        birth_mask = birth_mask.astype(int)
        survival_mask = survival_mask.astype(int)
        self.matrix = (self.matrix & survival_mask) | birth_mask


