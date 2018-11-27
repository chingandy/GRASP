
import random
import numpy as np
from itertools import cycle
import sys

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

def build_dict(cage_r, cage_ir, objects):

    # cage_r, cage_ir = separate_classes(filetpath)
    # objects = check_objects(cage_r)

    dict = {}
    dict_ir = {}

    for data in cage_r:
        try:
            key = objects.index(data[-7])
            if key in dict:
                dict[key].append(data)
            else:
                dict[key] = [data]
        except ValueError:
            print("the objects between two classes are different. One object from cage relevent class will be excluded.")

    for data in cage_ir:
        try:
            key = objects.index(data[-7])
            if key in dict_ir:
                dict_ir[key].append(data)
            else:
                dict_ir[key] = [data]
        except ValueError:
            print("the objects between two classes are different. One object from cage irrelevant class will be excluded.")
            pass

    return dict, dict_ir


def rebuilt_dataset(filepath):
    """result in NaN loss during training, not good """

    new_path = "/Users/chingandywu/GRASP/rebuilt-dataset/test1.txt"
    cage_r, cage_ir = separate_classes(filepath)
    objects_r = check_objects(cage_r)
    objects_ir = check_objects(cage_ir)
    if objects_r == objects_ir:
        print("the same")
        objects = objects_r
    else:
        print("different")
        quit()
    dict_r, dict_ir = build_dict(cage_r, cage_ir, objects)
    rebuilt_size = len(cage_ir) # we first rebuild the dataset based on the size of cage-irrelevant case
    # class_ind = 1 # 1 denotes cage_ir
    objects_size = len(objects)

    # rebuilt_size = 10
    with open(new_path,"w") as w:
        for itr in range(rebuilt_size):
            object_idx = itr % objects_size # iterate through all the objects (choose a object subset)

            choice = random.choice(dict_ir[object_idx])
            choice = ','.join(choice)
            w.write(choice)
            choice = random.choice(dict_r[object_idx])
            choice = ','.join(choice)
            w.write(choice)


    # object_idx = 2
    # # object = objects[object_idx]
    # # print(object)
    # print(dict_ir.keys())
    # choice = random.choice(dict_ir[object_idx])
    # choice = ','.join(choice)
    # print(choice)
    # print(type(choice))

def rebuilt_dataset_2(filepath, savepath):
    """ oversample the minor class of the dataset a bit """

    save_path = filepath.split("/")[-1]
    if savepath == 0:
        save_path = "/Users/chingandywu/GRASP/rebuilt-dataset/" + "re_" + save_path
    else:
        save_path = savepath + "re_" + save_path

    cage_r, cage_ir = separate_classes(filepath)

    rebuilt_size = len(cage_ir) # we first rebuild the dataset based on the size of cage-irrelevant case

    with open(save_path,"w") as w:
        for e in cage_ir:
            w.write(','.join(e))

        pool = cycle(cage_r)
        count = 0
        size_aug_ind = 72
        for e in pool:
            if count < rebuilt_size/size_aug_ind:
                w.write(','.join(e))
                count += 1
            else:
                quit()


def rebuilt_dataset_split(filepath, savepath):

    save_path = filepath.split("/")[-1]
    if savepath == 0:
        save_path_1 = "/Users/chingandywu/GRASP/rebuilt-dataset/" + "re_" + save_path + "_train"
        save_path_2 = "/Users/chingandywu/GRASP/rebuilt-dataset/" + "re_" + save_path + "_test"

    else:
        save_pat_1 = savepath + "train_re_" + save_path
        save_path_2 = savepath + "test_re_" + save_path

    cage_r, cage_ir = separate_classes(filepath)

    rebuilt_size = len(cage_ir) # we first rebuild the dataset based on the size of cage-irrelevant case
    try:
        with open(save_path_1, 'w') as a, open(save_path_2, 'w') as b:

            for key, item in enumerate(cage_ir):
                if key % 4 == 1:
                    # print("writing in the train dataset")
                    a.write(','.join(item))
                else:
                    b.write(','.join(item))
                    # print("writing in the test dataset")

            pool = cycle(cage_r)
            count = 0
            size_aug_ind = 72
            for key, item in enumerate(pool):
                if count < rebuilt_size/size_aug_ind:
                    if key % 4 == 1:
                        a.write(','.join(item))
                        # print("writing in the train dataset")
                    else:
                        b.write(','.join(item))
                        # print("writing in the test dataset")
                    count += 1
                else:
                    quit()

    except IOError as e:
        print ('Operation failed: %s' % e.strerror)




def built_small_dataset(filepath, savepath):
    """ build a small dataset contains only n objects with even class distribution.
        n: the number of objects
    """
    n = input("Please enter the number of objects: ")
    n = int(n)
    save_path = filepath.split("/")[-1]
    if savepath == 0:
        save_path = "/Users/chingandywu/GRASP/rebuilt-dataset/" + "test_small_" + save_path
    else:
        save_path = savepath + "test_small_" + save_path

    cage_r, cage_ir = separate_classes(filepath)
    objects_r = check_objects(cage_r)
    objects_ir = check_objects(cage_ir)

    print("#"*100)
    print("objects_r:\n", len(objects_r),"\n", objects_r)
    print("objects_ir:\n", len(objects_ir), "\n", objects_ir)
    if len(objects_r) != len(objects_ir):
        print("Warning: the numbers of objects in two classes are different. This may occur a problem.")
    print("#"*100)

    dict_r, dict_ir = build_dict(cage_r, cage_ir, objects_r)
    # print("dict_r keys: \n",dict_r.keys())
    with open(save_path, "w") as w:
        for i in range(n):
            w.write(','.join(dict_r[i][0]))
            w.write(','.join(dict_ir[i][0]))








if __name__ == '__main__':

    print(sys.argv)

    if(len(sys.argv) < 3):
        print('usage: preprcs.py [r/s/rs] <.txt file> <save path>, where <.txt file> is the file you want to preprocess')
        quit()

    # datasetfile= str(sys.argv[1])
    flag = str(sys.argv[1])
    filepath = str(sys.argv[2])
    savepath = 0
    if len(sys.argv) > 3:
        savepath = str(sys.argv[3])
    # filepath = "/Users/chingandywu/GRASP/data_gen/dataset_300_400.txt"
    if flag == "r":
        rebuilt_dataset_2(filepath, savepath)
    elif flag == "s":
        built_small_dataset(filepath, savepath)
    elif flag == "rs":
        rebuilt_dataset_split(filepath, savepath)
    else:
        quit()
