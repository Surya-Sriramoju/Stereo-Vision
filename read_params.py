import numpy as np
def get_K(path):
    K = []
    with open(path+'/calib.txt') as f:
        lines = f.readlines()
        k_string = lines[0].split(';')
        for i in k_string:
            if ('cam0' in i) or ('cam1' in i):
                x = i.split(' ')
                x1,x2,x3 = x
                x1 = float(x1[6:])
                x2 = float(x2)
                x3 = float(x3)
                K.append([x1,x2,x3])
            else:
                if "]\n" in i:
                    x = i.split()
                    x1,x2,x3 = x
                    x1 = float(x1)
                    x2 = float(x2)
                    x3 = float(x3[:-1])
                    K.append([x1,x2,x3])
                else:
                    x = i.split(' ')
                    _,x1,x2,x3 = x
                    x1 = float(x1)
                    x2 = float(x2)
                    x3 = float(x3)
                    K.append([x1,x2,x3])
    K = np.array(K).reshape(3,-1)
    return K

