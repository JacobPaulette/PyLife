PyLife - a Game of Life implementation
======================================

by Jacob Paulette
-----------------


Implemented in Python 3.
Compatible with Python 2.
Reads rle pattern files.

External Dependencies:
---------------------

scipy

numpy

pygame 1.9.1 for Python 2, pygame 1.9.2 for Python 3


Usage:
------


optional arguments:
  -h, --help            show this help message and exit
  -g GEN, --gen GEN     Generations of life.
  -w WAIT, --wait WAIT  Time in ms between frames.
  -p PIX, --pix PIX     Cell size, recomended (2-5).
  -wr, --wrap           Call flag to let matrix wrap around itself.
  -fr FRAME, --frame FRAME
                        Frames of dead cells around pattern, recommend at
                        least 5.
  -f FILE, --file FILE  File/directory of for rle pattern file.
  -r RAND, --rand RAND  Make random life matrix of (n,n) dimensions.
  --nogui               Turns off GUI, for benchmarking Life class.


In terminal:

>python main.py    # will use default pattern rle.string, in rle.py

>python main.py -f file.txt -p 2 -wr # etc.
