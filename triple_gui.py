import pygame
import numpy as np
import sys


def color_matrix(array):
    """Replace binary values with RGB black and white."""
    size = array.shape + (3,) 
    color = np.full(size, [255,255,255], dtype=np.int)
    color[array == 1] = [0,0,0]
    return color


def scale(array, n):
    """Scale the array for blitting by factor n."""
    return np.kron(array, np.ones((n,n), dtype=np.int))


def mirror(array):
    """Invert the array for blitting."""
    return np.rot90(np.flipud(array), 3)


def transform(array, pixel_size):
    return mirror(color_matrix(scale(array, pixel_size)))
    

def main(life, pixel_size=3, wait = 50, gen = -1, speed=1, frame=None):
    """Run life simulation."""
    pygame.init()
    pygame.display.set_caption("PyLife")
    mat = mirror(scale(life.view_matrix(0), pixel_size))# to find dimensions
    screen = pygame.display.set_mode(mat.shape)
    switch = 5
    count_total = 0
    count = 0
    print(0)
    while 1: # mainloop
        for e in pygame.event.get(): # quit program if window closed
            if e.type == pygame.QUIT: sys.exit() 
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                sys.exit()
        if gen == 0: # end mainloop if there are no generations left
            break    
        else:
            gen -= 1
        """
        super_imposed = np.zeros(life.view_matrix(0).shape)
        for i in range(life.get_size()):
            super_imposed = np.logical_or(super_imposed, life.view_matrix(i))
        """

        if frame is None:
            super_imposed = life.view_matrix(count % life.get_size())
            count_total += 1
            if (count_total % switch == 0):
                count += 1
                print(count % life.get_size())
        else:
            super_imposed = life.view_matrix(frame)
        mat = transform(super_imposed, pixel_size) 
        pygame.surfarray.blit_array(screen, mat)
        for i in range(speed):
            life.update_matrix() 
        pygame.display.flip()
        pygame.time.wait(wait)


##############Development#################################

def dirty(life, pixel_size=3, wait = 50, gen = -1):
    """Run life simulation."""
    pygame.init()
    pygame.display.set_caption("PyLife")
    mat = mirror(scale(life.view_matrix(), pixel_size))# to find dimensions
    screen = pygame.display.set_mode(mat.shape)
    prev = None

    while 1: # mainloop
        for e in pygame.event.get(): # quit program if window closed
            if e.type == pygame.QUIT: sys.exit() 
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                sys.exit()
        if gen == 0: # end mainloop if there are no generations left
            break    
        else:
            gen -= 1

        mat = life.view_matrix()
        rekt = dict()

        if life.get_generation() == 0:
            screen.fill([255,255,255])
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if mat[i][j] == 1:
                        r = pygame.Rect(j, i, j+pixel_size, i+pixel_size)
                        screen.fill([0,0,0], r)
                        rekt[(i,j)] = r 
                        prev = mat
        else:
            change = prev ^ mat
            coords = np.nonzero(change)
            for i in range(len(coords[0])):
                m = coords[0][i]
                n = coords[1][i]
                if prev[m][n] == 1:
                    screen.fill([255,255,255], rekt[(m,n)])
                else:
                    r = pygame.Rect(j,i, pixel_size, pixel_size)
                    screen.fill([0,0,0], r)
                    rekt[(m,n)] = r 
                    prev = mat 
        pygame.display.flip()
        life.update_matrix()
        input()
