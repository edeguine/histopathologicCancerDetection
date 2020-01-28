import torch
import torch.nn as nn

import loadData
import pipeline
import myCNN

cuda = True

def main():
    model = myCNN.myConvNet()
    if cuda:
        model = model.cuda()
    model.load_state_dict(torch.load('models/model_300.pth'))
    model.eval()

    testset = loadData.loadTestSet('test', meanNorm=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    classWriter = loadData.TestClassification('test')

    ct = 0
    for (image, fname) in test_loader:
        inputs = image.float()
        
        if cuda:
            inputs = inputs.cuda()

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        print(ct, fname, predicted.item())
        ct += 1

        classWriter.write(fname, predicted.item())

if __name__ == "__main__":
    main()
