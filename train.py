import os, argparse, torch
from torch import nn
from torch.utils.data import DataLoader
from data.datasets import FolderDataset
from data.transforms import get_transforms
from models.efficientnet_b4 import EfficientNetB4
from models.vit_b16 import ViTB16
from models.convnext_tiny import ConvNeXtTiny
from models.inceptionresnetv2_sab import InceptionResNetV2_SAB_Surrogate
from utils import set_seed, to_device, save_checkpoint
from tqdm import tqdm

def get_model(name: str, num_classes: int, pretrained: bool=False):
    name = name.lower()
    if name == 'efficientnet_b4':
        return EfficientNetB4(num_classes, pretrained)
    if name == 'vit_b16':
        return ViTB16(num_classes, pretrained)
    if name == 'convnext_tiny':
        return ConvNeXtTiny(num_classes, pretrained)
    if name == 'inceptionresnetv2_sab':
        return InceptionResNetV2_SAB_Surrogate(num_classes, pretrained)
    raise ValueError(f'Unknown model {name}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True, help='Root folder containing train/ val/ test/')
    ap.add_argument('--model', type=str, default='efficientnet_b4', choices=['efficientnet_b4','vit_b16','convnext_tiny','inceptionresnetv2_sab'])
    ap.add_argument('--epochs', type=int, default=5)  # keep small for demo
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--pretrained', action='store_true')
    ap.add_argument('--out', type=str, default='checkpoints')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # datasets
    train_ds = FolderDataset(args.data_root, split='train', img_size=224)
    val_ds = FolderDataset(args.data_root, split='val', img_size=224)
    num_classes = len(train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = get_model(args.model, num_classes, args.pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))
    crit = nn.CrossEntropyLoss()

    best_val = 0.0
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [train]')
        total, correct, loss_sum, n = 0, 0, 0.0, 0
        for batch in pbar:
            x, y = to_device(batch, device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            pred = logits.argmax(dim=1)
            correct += (pred==y).sum().item()
            total += y.numel()
            loss_sum += loss.item()*y.size(0)
            n += y.size(0)
            pbar.set_postfix(loss=loss_sum/n, acc=correct/total)

        # validate
        model.eval()
        tot, cor, lsum, n = 0,0,0.0,0
        with torch.no_grad():
            for x,y in DataLoader(val_ds, batch_size=args.batch_size):
                x,y = x.to(device), y.to(device)
                logits = model(x)
                l = crit(logits, y)
                pred = logits.argmax(dim=1)
                cor += (pred==y).sum().item()
                tot += y.numel()
                lsum += l.item()*y.size(0)
                n += y.size(0)
        val_acc = cor/tot if tot>0 else 0.0
        print(f'[val] loss={lsum/n:.4f} acc={val_acc:.4f}')

        if val_acc > best_val:
            best_val = val_acc
            save_checkpoint({'model': args.model,
                             'state_dict': model.state_dict(),
                             'num_classes': num_classes}, os.path.join(args.out, f'{args.model}_best.pt'))
    print('Done. Best val acc=', best_val)

if __name__ == '__main__':
    main()
