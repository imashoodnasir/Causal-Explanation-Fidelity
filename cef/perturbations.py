import torch
import torch.nn.functional as F

def topk_mask(cam: torch.Tensor, p: float):
    """Return binary mask selecting top-p mass pixels (per image).
    cam: [B,H,W] normalized to [0,1] and sum to arbitrary mass.
    p in [0,1]
    """
    B,H,W = cam.shape
    flat = cam.view(B, -1)
    k = (flat.size(1) * p).long().clamp(min=1)
    # compute thresholds per image
    vals, idx = torch.sort(flat, dim=1, descending=True)
    thresh = vals.gather(1, (k-1).unsqueeze(1)).squeeze(1)  # value at k-th
    mask = (flat >= thresh.unsqueeze(1)).float().view(B,H,W)
    return mask

def delete_op(x: torch.Tensor, mask: torch.Tensor, mode='blur'):
    """Delete salient region indicated by mask=1."""
    if mode == 'mean':
        baseline = x.mean(dim=(2,3), keepdim=True)
        return x*(1-mask.unsqueeze(1)) + baseline*mask.unsqueeze(1)
    # default blur
    k = 11
    pad = k//2
    xpad = F.pad(x, (pad,pad,pad,pad), mode='reflect')
    xblur = F.avg_pool2d(xpad, k, stride=1)
    return x*(1-mask.unsqueeze(1)) + xblur*mask.unsqueeze(1)

def insert_op(x: torch.Tensor, mask: torch.Tensor, x0: torch.Tensor):
    """Insert salient region from x into baseline x0 where mask=1."""
    return x0*(1-mask.unsqueeze(1)) + x*mask.unsqueeze(1)
