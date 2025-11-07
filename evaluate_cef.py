import os, argparse, torch
from torch.utils.data import DataLoader
from data.datasets import FolderDataset
from models.efficientnet_b4 import EfficientNetB4
from models.vit_b16 import ViTB16
from models.convnext_tiny import ConvNeXtTiny
from models.inceptionresnetv2_sab import InceptionResNetV2_SAB_Surrogate
from xai.gradcam_pp import GradCAMPlusPlus
from xai.attention_rollout import AttentionRolloutViT
from cef.cef_metric import compute_cef_for_batch
from tqdm import tqdm

def get_model(name: str, num_classes: int):
    name = name.lower()
    if name == 'efficientnet_b4':
        return EfficientNetB4(num_classes)
    if name == 'vit_b16':
        return ViTB16(num_classes)
    if name == 'convnext_tiny':
        return ConvNeXtTiny(num_classes)
    if name == 'inceptionresnetv2_sab':
        return InceptionResNetV2_SAB_Surrogate(num_classes)
    raise ValueError(f'Unknown model {name}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--model', type=str, required=True, choices=['efficientnet_b4','vit_b16','convnext_tiny','inceptionresnetv2_sab'])
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--steps', type=int, default=10, help='Number of p-steps for curves')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    test_ds = FolderDataset(args.data_root, split='test', img_size=224)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_classes = len(test_ds.classes)
    model = get_model(args.model, num_classes).to(args.device)

    ckpt = torch.load(args.weights, map_location=args.device)
    model.load_state_dict(ckpt['state_dict'])

    # choose target layer for Grad-CAM++
    cam_generator = None
    if args.model in ['efficientnet_b4','convnext_tiny','inceptionresnetv2_sab']:
        # pick the last conv-like module
        target_layer = None
        if args.model == 'efficientnet_b4':
            target_layer = model.backbone.features[-1]
        elif args.model == 'convnext_tiny':
            target_layer = model.backbone.features[-1][-1].block[0] if hasattr(model.backbone.features[-1][-1], 'block') else model.backbone.features[-1][-1]
        else:
            target_layer = model.layer4[-1].conv3 if hasattr(model.layer4[-1], 'conv3') else model.layer4[-1]
        cam_generator = GradCAMPlusPlus(model, target_layer)
    else:
        cam_generator = AttentionRolloutViT(model.backbone)

    cef_vals = []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc='CEF'):
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)
            # generate CAMs
            if args.model == 'vit_b16':
                cams = cam_generator.generate(imgs)
            else:
                cams = cam_generator.generate(imgs, target_class=None)
            # normalize to probability-like saliency
            # ensure sum to 1 per image
            B, H, W = cams.shape
            cams = (cams - cams.view(B,-1).min(1)[0].view(B,1,1))
            cams = cams / (cams.view(B,-1).sum(1)[0].view(B,1,1) + 1e-8)

            cef_batch = compute_cef_for_batch(model, imgs, cams, labels, steps=args.steps, device=args.device)
            cef_vals.extend(cef_batch)

    import numpy as np
    print(f'CEF mean={np.mean(cef_vals):.4f} std={np.std(cef_vals):.4f} over {len(cef_vals)} samples.')

if __name__ == '__main__':
    main()
