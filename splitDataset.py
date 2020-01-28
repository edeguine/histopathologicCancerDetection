import os, sys
import subprocess

def main():
    csv = open(sys.argv[1]).readlines()[1:]

    bigtrain = 200000
    bigtest = 10000

    train = 50000
    test = 10000

    ctTrain = 0
    ctTest = 0

    root = "/home/ubuntu/histoCancerKaggle/"

    subset = 'big'

    trainCsv = open(root + f'data/{subset}train_labels.csv', 'w')
    testCsv = open(root + f'data/{subset}test_labels.csv', 'w')

    for line in csv:
        filename, label = line.split(',')
        label = int(label)
        
        if ctTrain < train:
            cmd = f"cp {root}data/train/{filename}.tif {root}data/{subset}train/{filename}.tif"
            ctTrain += 1
            trainCsv.write(f"{filename},{label}\n")
        elif ctTrain >= train and ctTest < test:
            cmd = f"cp {root}data/train/{filename}.tif {root}data/{subset}test/{filename}.tif"
            testCsv.write(f"{filename},{label}\n")
            ctTest += 1
        else:
            break
        print(cmd)
        os.system(cmd)

    trainCsv.close()
    testCsv.close()
if __name__ == "__main__":
    main()
