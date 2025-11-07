import torch
import torch.nn.functional as F

class GradCAMPlusPlus:
    """Generic Grad-CAM++ for a given target layer (module) and class index."""
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        # grad_out is a tuple with length equal to number of outputs; take first
        self.gradients = grad_out[0].detach()

    def generate(self, x, target_class=None):
        logits = self.model(x)
        if target_class is None:
            target_class = logits.argmax(dim=1)
        one_hot = F.one_hot(target_class, num_classes=logits.shape[1]).float()
        self.model.zero_grad(set_to_none=True)
        (logits * one_hot).sum().backward(retain_graph=True)

        A = self.activations  # [B, C, H, W]
        dYdA = self.gradients # [B, C, H, W]

        # Grad-CAM++ weights
        B, C, H, W = dYdA.shape
        d2 = dYdA**2
        d3 = dYdA**3
        # Avoid division by zero
        eps = 1e-8
        # alpha: as per Grad-CAM++ paper
        num = d2
        den = 2*d2 + (A * d3).sum(dim=(2,3), keepdim=True)
        alpha = num / (den + eps)
        weights = (alpha * F.relu(dYdA)).sum(dim=(2,3), keepdim=True)  # [B, C, 1, 1]

        cam = (weights * A).sum(dim=1)  # [B, H, W]
        cam = F.relu(cam)
        # normalize per image
        cam_min = cam.flatten(1).min(dim=1)[0].view(-1,1,1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1,1,1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

    def close(self):
        self.h1.remove()
        self.h2.remove()
