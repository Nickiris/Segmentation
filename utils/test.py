import torch

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
    ])
    # ,[
    #     [[1,2,3],[4,0,6],[0,8,9]],
    #     [[0,0,0],[0,2,8],[0,7,12]],
    #     [[8,0,2],[0,9,1],[0,6,13]]
    # ]])
    m = torch.tensor([
        [[1,2,3],[4,0,6],[0,8,9]],
        [[0,0,0],[0,2,0],[0,7,12]],
    ])
    c = m[0]
    print(c)
    print(torch.sum(c[1,:]))