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

    def __init__(self, matrix, rule_string, wrapping):
        self.rules = Rules(rule_string)
        self.matrix = Matrix(matrix, wrapping)
        self.generation = 0

    def view_matrix(self):
        """Return matrix as a numpy 2d array."""
        return self.matrix.view_matrix()

    def get_generation(self):
        """Return current generation of the game."""
        return self.generation

    def update_matrix(self):
        """Cycle to next generation of the game.

        Generates a matrix of neighbors for each element.
        Calculates new array from current matrix and neighbor
        matrix.  Sets the new array to the matrix object and 
        increments self.generation.
        """
        neighbors = self.matrix.find_neighbors()
        matrix = self.view_matrix()
        new_matrix = self.rules.epoch(matrix, neighbors)
        self.matrix.set_matrix(new_matrix)
        self.generation += 1


class Rules:
    """Parses rule_string into and calculates the next
    generation of matrix from the current matrix and the neighbor matrix.

    Parameters:

    rule_string : Determines rules for game, "B3/S23", "B36/S23".
        Note: rule_string is insensitive to case.
    """

    def __init__(self, rule_string):
        self.rules = self._parse(rule_string)

    def _parse(self, rule_string):
        """Converts rulestring in the form "B(somedigits)/S(somedigits)"
        to a dictionary such as:

        rule_string = "B3/S23" 
        dictionary = {1 : [2,3], 0 : [3]}

        1 in the dictionary represents that the current element is alive,
        and it requires 2 or 3 neighbors to stay alive. 0 represents
        that the current element is dead and needs 3 neighbors to live.
        These are called the survival, or s conditions and the birth, or
        b conditions respectively.
        """
        rules = rule_string.lower()
        if 'b' in rules  and 's' in rules:
            birth = re.findall(r'[b]\d+', rules)[0].replace('b', '')
            survive = re.findall(r'[s]\d+', rules)[0].replace('s','')
            blist = [int(i) for i in birth]
            slist = [int(i) for i in survive]
            return { 1 : slist, 0 : blist} 
        else: # if rule_string contains neither b or s, e.g. "3/23"
            bd = re.findall(r'\d+', rules) 
            blist = [int(i) for i in bd[0]]
            slist = [int(i) for i in bd[1]]
            return { 1 : slist, 0 : blist}
        
    def epoch(self, matrix, n_matrix):
        """Calculates new matrix.

        Parameters:

        matrix : 2d array of binary cells
        n_matrix : 2darray representing neighbors for each cell in matrix


        To maximize performance I used numpy/scipy methods for intensive 
        calculations. I could not find a method that transforms an array
        based on a compound bool expression similar to the below example.

        new_arr = np.zeros(1, length)
        for i in range(len(new_arr)):
            if n_matrix[i] in self.rules[matrix[i]]:
                new_arr[i] = 1
        return  new_arr.reshape(og_shape)

        However, using np.in1d you can create a boolean mask of the
        neighbor_matrix with both the survival list conditions, and the
        birth list conditions.  I hypothesized if you perform some 
        logic operation between the matrix, the survive_mask, and
        birth_mask, it will produce the desired output. Many 
        configurations I thought would work failed. But by brute force
        I determined the only correct configuration. There is probably a 
        more elegant solution, but this works, and it is fast.

        (P.S. changing the logic operations and order of operands in the f
        variable can sometimes produce neat effects.)
        """
                
        og_shape = n_matrix.shape # find dimensions of neighbor_matrix
        length = og_shape[0] * og_shape[1] # multiply dimensions
        flat_nmatx = n_matrix.reshape(1,length)[0] # flatten n_matrix
        flat_matx = matrix.reshape(1, length)[0] # flatten matrix
        survive_mask = np.in1d(flat_nmatx, self.rules[1]) # see np.in1d
        birth_mask = np.in1d(flat_nmatx, self.rules[0]) 
        f = (flat_matx & survive_mask) | birth_mask # idk why this works
        return f.reshape(og_shape) # og_shape == original dimensions


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


