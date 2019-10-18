import rle
import triple_gui
from interdependent import Life

import numpy as np
from scipy import ndimage

import re
import argparse
import time
import sys



def assign_args():
    """Create optional command line arguments."""
    h = [
    "Generations of life.",
    "Time in ms between frames.",
    "Cell size, recomended (2-5).",
    "Call flag to let matrix wrap around itself.",
    "Frames of dead cells around pattern, recommend at least 5.",
    "File/directory of for rle pattern file.",
    "Make random life matrix of (n,n) dimensions.",
    "Override default rule 'b3/s23'.",
    "Turns off GUI, for benchmarking Life class.",
    "Number of interdependent matrices, min 3."]
    
    global generations, wait, pixel_size, wrapping, frame, rule
    parser = argparse.ArgumentParser()
    parser.add_argument("-g" ,"--gen", type=int, help=h[0])
    parser.add_argument("-w", "--wait", type=int, help= h[1])
    parser.add_argument("-p", "--pix", type=int, help=h[2])
    parser.add_argument("-wr", "--wrap", action="store_true", help=h[3])
    parser.add_argument("-fr", "--frame", type=int, help=h[4])
    parser.add_argument("-f", "--file", type=str, help=h[5])
    parser.add_argument("-r", "--rand", type=int, help=h[6])
    parser.add_argument("-ru", "--rule", type=str, help=h[7])
    parser.add_argument("--nogui", action="store_true", help=h[8])
    parser.add_argument("-n", "--num", type=int, help=h[9])
    args = parser.parse_args()
    if type(args.gen) is int:
        generations = args.gen
    if type(args.wait) is int:
        wait = args.wait
    if type(args.pix) is int:
        pixel_size = args.pix
    if type(args.wrap) is bool:
        wrapping = args.wrap
    if type(args.frame) is int:
        frame = args.frame
    if type(args.rule) is str:
        rule = args.rule

    return args
        

def random_life(n, wrapping,num):
    """Return a Life object with random nxn matrix"""
    mat = [np.random.randint(2, size=(n,n)) for i in range(num)]
    
    return Life(mat, wrapping)


def main():
    """Main runtime function."""
    file_data = None 
    args = assign_args()
    
    """
    if type(args.file) is str: # Fetches rle file if specified.
        try:
            with open(args.file, 'r') as f:
                file_data = f.read()
        except:
            print("ERROR: File Does Not Exist")
            sys.exit(0)

    if file_data == None: # If file not specified, parses rle.string
        data = rle.parse_rle(rle.string, frame, frame)
    else: # Parses file specified for rle pattern data.
        try:
            data = rle.parse_rle(file_data, frame, frame)
        except:
            print("ERROR: Corrupt File Data")
            sys.exit(0)
    """
    wait = (50 if args.wait is None else args.wait) # wait time between generation in milliseconds. must be > 0
    pixel_size = (5 if args.pix is None else args.pix) # wait time between generation in milliseconds. must be > 0
    frame = 10 # Frame of dead cells around rle_pattern. should be > 5
    wrapping = (False if args.wrap is None else True) # wait time between generation in milliseconds. must be > 0
    num = (3 if args.num is None else args.num)
    if (num %2 ==0):
        return
    elif (num < 3):
        return
    generations = -1 # Number of generations before program dies.
    rule = None
    #



    if type(args.rand) is int:
        life = random_life(args.rand, wrapping, num)
    else:
        return
    """
    else:
        if rule is None:
            life = Life(data['matrix'], data['rulestring'], wrapping)
        else:
            life = Life(data['matrix'], rule, wrapping)"""

    a = time.time()
    triple_gui.main(life, pixel_size=pixel_size, wait=wait, gen=generations, speed=1)
    b = time.time()

    print("runtime: " + str(b-a) + " seconds")


if __name__ == '__main__':
    main()
