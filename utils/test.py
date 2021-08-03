import torch
import torch.nn.functional as  F
def label2image(labels):
    pass

if __name__ == '__main__':
    # inputs = torch.rand(2, 8, 224, 224)
    # targets = torch.rand(2, 224, 224)
    # # for input, target in zip(inputs, targets):
    # max_pred = torch.argmax(inputs, dim=1)



    t = torch.tensor([
        [[1,2,3],[4,5,6],[0,8,0]],
        [[0,0,4],[0,0,8],[0,0,0]],
        [[0.3, 4, 5], [6., 1, 3], [10., 1, 3]],
    ])
    # ,[
    #     [[1,2,3],[4,0,6],[0,8,9]],
    #     [[0,0,0],[0,2,8],[0,7,12]],
    #     [[8,0,2],[0,9,1],[0,6,13]]
    # ]])
    m = torch.tensor([
        [[0.2,3,3],[4.,0,6],[0.,8,4]],
        [[0.3,4,5],[6.,1,3],[10.,1,3]],
        [[0.5,3,2],[0.,9,1],[0.,1,3]],
    ])
    # original
    # 1.1027235984802246
    # 1.1287791728973389
    # 1.1416881084442139

    for p, c in zip(t.flatten(), m.flatten()):
        print(p, c)


