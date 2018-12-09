
import numpy as np
# filename = "/Users/chingandywu/chinganwu/KTH/Y2P1/Project Course in Data Science/Data_science_dataset/Dataset_test.txt"
# file = open(filename, mode='r')
# result = []
# number_of_lines = 5
# # data = np.loadtxt(filename, delimiter=',', dtype='str')
# for line in file:
#
# text = file.readline()
# file.close()

def separate_classes(filepath):
    """ separate the data into two different classes"""
    dict = {}
    cage_r = []
    cage_ir = []
    objects = []

    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
#         while line and cnt <= 10:
        while line:
            data =line.split(',')
#             print("Line {}:\n {}\n".format(cnt, data))
#             print(data[-3])
            if data[-3] == '1':
                cage_ir.append(data)
            elif data[-3] == '0':
                cage_r.append(data)
            else:
                print("this data point is of no use.")
            line = fp.readline()
            cnt += 1
            if cnt % 100 == 0:
                print("Processing:", cnt)
#     print(cage_ir)
    return cage_r, cage_ir

def check_objects(subset):
    objects = []
    for data in subset:
        object = data[-7]
        if object in objects:
            pass
        else:
            objects.append(object)

    return objects


def print_out_part(filepath):
    """ print out part of the data"""
    with open(filepath) as fp:
       line = fp.readline()
       cnt = 1
       while line and cnt <= 10:
           print("Line {}:\n {}\n".format(cnt, line.strip()))
           line = fp.readline()
           cnt += 1
    return


def file_len(filename):
    """ count the number of lines in the file """

    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# print_out_part(file_1)
# print("The number of lines: ", file_len(file_1))
# print("The number of lines: ", file_len(file_2))
# print("The number of lines: ", file_len(file_3))

def main():
    file_1 = "/Users/chingandywu/GRASP/data_gen/dataset/dataset_100_200.txt"
    # file_2 = '/Users/chingandywu/GRASP/data_gen/dataset_400_500.txt'
    # file_3 = '/Users/chingandywu/GRASP/data_gen/dataset/dataset_100_200.txt'

    print("The number of lines: ", file_len(file_1))
    cage_r, cage_ir = separate_classes(file_1)
    print("Objects in cage-irrelevant: ", len(check_objects(cage_ir)))
    print("Objects in cage-rrelevant: ", len(check_objects(cage_r)))

if __name__ == '__main__':
    main()

"""
***Dataset structure***
1 configuration per line
following structure:
Object-name, #object-circles,x1,y1,r1,...,xn,yn,rn,configuration-name,#gripper-circle,xg1,yg1,rg1,...,xg4,yg4,rg4,-,Object-name,configuration-name,label_quality,debug_1,label_subset,debug_2,debug_3

The label subset is the label you should use(1==cage-irrelevant,0=cage-relevant)
"""
