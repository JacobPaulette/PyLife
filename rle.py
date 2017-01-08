"""An rle reader.

Use:

rle_string = rle.string # See rle.string
data = parse_rle(rle_string, xborder = 10, yborder = 10)
data['matrix'] : 2d numpy array matrix.
data['rulestring'] : rule_string in the form "B(somedigits)/S(somedigits)"
data['dimensions'] : {'y' : integer, 'x' : integer}
"""
import re
import numpy as np


def _find_xy(string):
	nospace = string.replace(" ", "")
	xstring = re.findall('x=\d+', nospace)[0]
	x = int(re.findall('\d+', xstring)[0])
	ystring = re.findall('y=\d+', nospace)[0]
	y = int(re.findall('\d+', ystring)[0])
	return y, x 


def _find_rule(string):
	nospace = string.replace(" ", '')
	rulestring = re.findall('[bB]\d+/[sS]\d+', nospace)[0].lower()
	return rulestring


def _find_pattern(string):
    nospace = string.replace(" ", "")
    nospace = nospace.replace(re.findall('x=\d+', nospace)[0], '')
    nospace = nospace.replace(re.findall('y=\d+', nospace)[0], '')
    norule = nospace.replace(re.findall('b\d+/s\d+', nospace, re.IGNORECASE)[0], '')
    nonew = norule.replace("\n", "")
    patternstring = re.findall('[bo$\d+]+!', nonew)[0]
    return patternstring


def _classify(rle_list):
    out_list = list()
    sample = ['b', 'o', '$', '!']
    spec_index = 0
    for i in range(len(rle_list)):
        if rle_list[i] in sample:
            b = 1
            while (rle_list[i-b:i].isdigit()):
                b += 1
            b -= 1
            if not rle_list[i-1:i].isdigit():
                out_list.append((1, rle_list[i]))
            else:
                out_list.append((int(rle_list[i-b:i]), rle_list[i]))
                
    return out_list


def _make_matrix(string):
    dims = _find_xy(string)
    pattern_list = _classify(_find_pattern(string))
    matrix = np.zeros(dims, dtype=np.int)
    xcount = 0
    ycount = 0
    for i in pattern_list:
        if i[1] == 'o':
            for i in range(i[0]):
                matrix[ycount][xcount] = 1
                xcount += 1
        elif i[1] == 'b':
            for i in range(i[0]):
                xcount += 1
        elif i[1] == '$':
            for i in range(i[0]):
                ycount += 1
                xcount = 0
        elif i[1] == '!':
            break

    return matrix


def _frame(arr, xborder=0, yborder=0):
    new_arr = list(arr)
    xframe = [0] * xborder
    for i in range(len(arr)):
        new_arr[i] = xframe + arr[i] + xframe
    yframe = [0] * (len(new_arr[0]))
    for i in range(yborder):
        new_arr.insert(0, yframe)
        new_arr.append(yframe)
    return new_arr
				

def parse_rle(string, xborder = 5, yborder = 5):
    matrix = _make_matrix(string).tolist()
    border_matrix = np.array(_frame(matrix, xborder, yborder))
    dims = (len(border_matrix), len(border_matrix[0]))
    dimsdict = { 'y' : dims[0], 'x' : dims[1] }
    rule = _find_rule(string)
    return { "matrix" : border_matrix, "dimensions" : dimsdict, "rulestring" : rule}


        
        
string = """
#N twogun
#O V. Everett Boyer and Doug Petrie
#C The smallest known period-60 gun; it uses two copies of the Gosper 
#C glider gun.
x = 39, y = 27, rule = b3/s23
27bo11b$25bobo11b$15b2o6b2o12b2o$14bo3bo4b2o12b2o$3b2o8bo5bo3b2o14b$3b
2o8bo3bob2o4bobo11b$13bo5bo7bo11b$14bo3bo20b$15b2o22b$26bo12b$27b2o10b
$26b2o11b4$21b2o16b$9bobo10b2o15b$9bo2bo8bo17b$2o10b2o11b2o12b$2o8bo3b
2o8bobo12b$5b2o5b2o9bo6b2o7b$4bo4bo2bo10bo2bo2bo2bo6b$9bobo11bo6b3o6b$
24bobo5b3o4b$25b2o6bobo3b$35bo3b$35b2o!
"""
