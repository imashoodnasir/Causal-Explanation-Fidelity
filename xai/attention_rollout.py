import torch
import torch.nn.functional as F

class AttentionRolloutViT:
    """Attention rollout for torchvision ViT (assumes .encoder.layers list)."""
    def __init__(self, model, head_fusion='mean'):
        self.model = model.eval()
        self.blocks = model.encoder.layers
        self.head_fusion = head_fusion
        self.attns = []
        self.hooks = [blk.attn.attn_drop.register_forward_hook(self._hook) for blk in self.blocks]

    def _hook(self, module, input, output):
        # We can't directly get attention matrices from torchvision ViT easily.
        # As a pragmatic alternative: capture attn probabilities via attn_drop input
        # input is a tuple; the first element is the attn probs pre-dropout
        attn = input[0].detach() if isinstance(input, tuple) else input.detach()
        self.attns.append(attn)  # [B, heads, tokens, tokens]

    def generate(self, x):
        self.attns = []
        _ = self.model(x)
        # fuse heads and rollout
        result = None
        for attn in self.attns:
            if self.head_fusion == 'mean':
                a = attn.mean(dim=1)  # [B, T, T]
            else:
                a = attn.max(dim=1).values
            # add identity and renormalize
            I = torch.eye(a.size(-1), device=a.device).unsqueeze(0).repeat(a.size(0),1,1)
            a = a + I
            a = a / a.sum(dim=-1, keepdim=True)
            result = a if result is None else torch.bmm(result, a)
        # attention from CLS to patches
        cls_attn = result[:, 0, 1:]  # [B, N_patches]
        # reshape to spatial map (assuming 14x14 for 224/16)
        s = int((cls_attn.size(1))**0.5)
        cam = cls_attn.view(-1, s, s)
        cam = F.interpolate(cam.unsqueeze(1), size=(224,224), mode='bilinear', align_corners=False).squeeze(1)
        cam_min = cam.flatten(1).min(dim=1)[0].view(-1,1,1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1,1,1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

    def close(self):
        for h in self.hooks: h.remove()
