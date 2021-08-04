import torch

# Todo optimization implementation
def label2mask(label, color_map):
    """

    label: torch.long(N, H, W)
    color_map: List[List]
    return: mask torch.ByteTensor(N, C, H, W)
    """
    N, H, W = label.size()
    mask = torch.zeros((N, 3, H, W), dtype=torch.uint8)
    for i in range(N):
        for h in range(H):
            for w in range(W):
                rgb = color_map[label[i, h, w]]
                mask[i, 0, h, w], mask[i, 1, h, w], mask[i, 2, h, w] = rgb[0], rgb[1], rgb[2]

    return mask


