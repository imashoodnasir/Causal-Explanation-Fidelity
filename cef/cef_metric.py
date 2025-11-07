import torch
import torch.nn.functional as F
from .perturbations import topk_mask, delete_op, insert_op

@torch.no_grad()
def compute_cef_for_batch(model, images, cams, targets, steps: int=10, device='cpu'):
    """Compute CEF for a batch of images given CAMs (normalized per-image in [0,1]).
    - model: classifier
    - images: [B,3,H,W]
    - cams: [B,H,W] (normalized)
    - targets: [B] class indices
    Returns: list of CEF values per image.
    """
    model.eval()
    B,_,H,W = images.shape
    probs = torch.softmax(model(images.to(device)), dim=1)
    # choose per-sample target if None: use predicted
    t = targets.to(device)

    # Prepare baseline x0 (per-channel mean image)
    x0 = images.mean(dim=(0,2,3), keepdim=True).repeat(B,1,H,W).to(device)

    p_levels = torch.linspace(0,1,steps+1, device=device)

    auc_del = torch.zeros(B, device=device)
    auc_ins = torch.zeros(B, device=device)

    for i in range(1, len(p_levels)):
        p = p_levels[i].item()
        mask = topk_mask(cams.to(device), p)
        x_del = delete_op(images.to(device), mask)  # [B,3,H,W]
        x_ins = insert_op(images.to(device), mask, x0)

        py_del = torch.softmax(model(x_del), dim=1)[torch.arange(B), t]
        py_ins = torch.softmax(model(x_ins), dim=1)[torch.arange(B), t]

        dp = (p_levels[i] - p_levels[i-1]).item()
        auc_del += py_del * dp
        auc_ins += py_ins * dp

    # Baselines for normalization (random and uniform); oracle ~ cams already closest, so skip explicit oracle to keep simple and stable
    # Random baseline: shuffle cams
    cams_rand = cams.view(B, -1)
    idx = torch.randperm(cams_rand.size(1), device=device)
    cams_rand = cams_rand[:, idx].view(B, H, W)
    auc_del_rand = torch.zeros(B, device=device)
    auc_ins_rand = torch.zeros(B, device=device)
    for i in range(1, len(p_levels)):
        p = p_levels[i].item()
        mask = topk_mask(cams_rand, p)
        x_del = delete_op(images.to(device), mask)
        x_ins = insert_op(images.to(device), mask, x0)
        py_del = torch.softmax(model(x_del), dim=1)[torch.arange(B), t]
        py_ins = torch.softmax(model(x_ins), dim=1)[torch.arange(B), t]
        dp = (p_levels[i] - p_levels[i-1]).item()
        auc_del_rand += py_del * dp
        auc_ins_rand += py_ins * dp

    # Uniform baseline: mask by uniform saliency
    uniform = torch.ones_like(cams, device=device) / (H*W)
    auc_del_uni = torch.zeros(B, device=device)
    auc_ins_uni = torch.zeros(B, device=device)
    for i in range(1, len(p_levels)):
        p = p_levels[i].item()
        mask = topk_mask(uniform, p)
        x_del = delete_op(images.to(device), mask)
        x_ins = insert_op(images.to(device), mask, x0)
        py_del = torch.softmax(model(x_del), dim=1)[torch.arange(B), t]
        py_ins = torch.softmax(model(x_ins), dim=1)[torch.arange(B), t]
        dp = (p_levels[i] - p_levels[i-1]).item()
        auc_del_uni += py_del * dp
        auc_ins_uni += py_ins * dp

    # Normalize AUCs to [0,1] using min-max over baselines
    eps = 1e-6
    a_del_min = torch.minimum(auc_del_rand, auc_del_uni)
    a_del_max = torch.maximum(auc_del_rand, auc_del_uni)
    a_ins_min = torch.minimum(auc_ins_rand, auc_ins_uni)
    a_ins_max = torch.maximum(auc_ins_rand, auc_ins_uni)

    a_del_hat = (auc_del - a_del_min) / (a_del_max - a_del_min + eps)
    a_ins_hat = (auc_ins - a_ins_min) / (a_ins_max - a_ins_min + eps)

    cef = 0.5*(1 - a_del_hat) + 0.5*(a_ins_hat)
    return cef.detach().cpu().tolist()
