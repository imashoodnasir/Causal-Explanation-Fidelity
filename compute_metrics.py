import torch
import torch.nn.functional as F

def pai_iou(saliency: torch.Tensor, mask: torch.Tensor, thresh: float=0.5):
    """Simple perceptual alignment index as IoU between binarized saliency and lesion mask.
    saliency: [B,1,H,W] in [0,1]; mask: [B,1,H,W] in {0,1}
    """
    s = (saliency >= thresh).float()
    inter = (s*mask).sum(dim=(1,2,3))
    union = (s+mask - s*mask).sum(dim=(1,2,3)) + 1e-8
    return (inter/union).mean().item()

def cas_proxy(saliency: torch.Tensor, logits: torch.Tensor):
    """Toy proxy: lower entropy of logits + concentration of saliency (Gini) indicates cognitive alignment.
    Returns scalar in [0,1] (heuristic).
    """
    p = torch.softmax(logits, dim=1)
    ent = -(p * (p+1e-8).log()).sum(dim=1).mean()  # higher = less certain
    # saliency concentration via Gini: 1 - normalized L1 distance from uniform
    B,_,H,W = saliency.shape
    s = saliency.view(B,-1)
    s = s/(s.sum(dim=1, keepdim=True)+1e-8)
    gini = 1.0 - (2*s.sort(dim=1)[0] * torch.arange(1, s.size(1)+1, device=s.device) / s.size(1)).sum(dim=1).mean()/s.size(1)
    # combine (heuristic scaling)
    val = torch.sigmoid(2.0*gini - 0.5*ent)
    return val.item()
