import torch

# Todo optimization implementation
def label2mask(label, color_map):
    """

    label: torch.long(N, H, W)
    color_map: List[List]
    return: mask torch.ByteTensor(N, C, H, W)
    """
    color_map = torch.as_tensor(color_map, dtype=torch.uint8)
    mask = label.long()

    return color_map[mask,:]


