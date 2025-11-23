import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.signal import resample
import glob
import os
import json
import torch.nn.functional as F

# --- Config ---
LPC_ORDER = 30
CHANNELS_IN = LPC_ORDER + 1
CHANNELS_OUT = 1
DATA_DIR = './dataset'
SAVE_DIR = './checkpoints'
MODEL_NAME = 'vocalhr_unet.pth'
BATCH_SIZE = 8
EPOCHS = 200
LR = 0.0001
SEED = 42

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Model Architecture ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNet1D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet1D, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(512, 1024))
        
        self.up1 = nn.ConvTranspose1d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose1d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv1d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        # Handle padding for odd dimensions
        diff = x4.size()[2] - x.size()[2]
        x = F.pad(x, [diff // 2, diff - diff // 2])
        x = self.conv1(torch.cat([x4, x], dim=1))
        
        x = self.up2(x)
        diff = x3.size()[2] - x.size()[2]
        x = F.pad(x, [diff // 2, diff - diff // 2])
        x = self.conv2(torch.cat([x3, x], dim=1))
        
        x = self.up3(x)
        diff = x2.size()[2] - x.size()[2]
        x = F.pad(x, [diff // 2, diff - diff // 2])
        x = self.conv3(torch.cat([x2, x], dim=1))
        
        x = self.up4(x)
        diff = x1.size()[2] - x.size()[2]
        x = F.pad(x, [diff // 2, diff - diff // 2])
        x = self.conv4(torch.cat([x1, x], dim=1))
        
        return self.outc(x)

# --- Custom Loss ---
class VocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # Basic shape loss
        loss_l1 = self.l1(pred, target)
        
        # Gradient loss (force sharp R-peaks)
        grad_pred = pred[:, :, 1:] - pred[:, :, :-1]
        grad_true = target[:, :, 1:] - target[:, :, :-1]
        loss_grad = self.l1(grad_pred, grad_true)
        
        # Peak emphasis (squared error punishes large deviations more)
        loss_peak = self.mse(pred**2, target**2)
        
        total = 1.0 * loss_l1 + 0.5 * loss_grad + 0.3 * loss_peak
        return total, loss_l1.item()

# --- Dataset ---
class VocalHRDataset(Dataset):
    def __init__(self, data_dir, stats=None):
        self.files = glob.glob(os.path.join(data_dir, '*.npz'))
        self.stats = stats
        print(f"Dataset loaded: {len(self.files)} files.")
        
        if not self.stats and len(self.files) > 0:
            self.stats = self._get_stats()

    def _get_stats(self):
        print("Calculating dataset stats...")
        # Sample subset to save time
        subset = self.files[:min(100, len(self.files))]
        pharynx_all, ecg_all = [], []
        
        for f in subset:
            try:
                d = np.load(f)
                p, e = d['pharynx_features'], d['ecg_data']
                if p.shape[0] > 20: 
                    pharynx_all.append(p)
                    ecg_all.append(e)
            except: pass
            
        if not pharynx_all: return None
        
        p_concat = np.concatenate(pharynx_all, axis=0)
        e_concat = np.concatenate(ecg_all, axis=0)
        
        return {
            'p_mean': np.mean(p_concat, axis=0),
            'p_std': np.std(p_concat, axis=0) + 1e-6,
            'e_mean': np.mean(e_concat),
            'e_std': np.std(e_concat) + 1e-6
        }

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            data = np.load(self.files[idx])
            pharynx = data['pharynx_features'].astype(np.float32)
            ecg = data['ecg_data'].astype(np.float32)
        except: return None

        if pharynx.shape[0] < 20 or ecg.shape[0] < 100: return None

        # Resample ECG to match voice frames
        ecg = resample(ecg, pharynx.shape[0]).astype(np.float32)

        # Normalize
        if self.stats:
            pharynx = (pharynx - self.stats['p_mean']) / self.stats['p_std']
            ecg = (ecg - self.stats['e_mean']) / self.stats['e_std']
        else:
            # Fallback
            pharynx = (pharynx - pharynx.mean(0)) / (pharynx.std(0) + 1e-6)
            ecg = (ecg - ecg.mean()) / (ecg.std() + 1e-6)

        p_t = torch.tensor(np.nan_to_num(pharynx), dtype=torch.float32).permute(1, 0)
        e_t = torch.tensor(np.nan_to_num(ecg), dtype=torch.float32).unsqueeze(0)
        return p_t, e_t

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None, None
    
    inputs, targets = zip(*batch)
    max_len = max([x.shape[1] for x in inputs])
    
    # Pad to max length in batch
    pad_inputs = [F.pad(x, (0, max_len - x.shape[1])) for x in inputs]
    pad_targets = [F.pad(y, (0, max_len - y.shape[1])) for y in targets]
    
    return torch.stack(pad_inputs), torch.stack(pad_targets)

# --- Main ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    dataset = VocalHRDataset(DATA_DIR)
    if len(dataset) == 0: return
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    model = UNet1D(CHANNELS_IN, CHANNELS_OUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = VocalLoss()
    
    best_loss = float('inf')
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        count = 0
        
        for x, y in loader:
            if x is None: continue
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss, l1_val = loss_fn(pred, y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            count += 1
            
        avg_loss = epoch_loss / count if count > 0 else 0
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.5f}")
            
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            # Save stats specifically for the best model
            if dataset.stats:
                np.savez(os.path.join(SAVE_DIR, 'norm_stats.npz'), **dataset.stats)

    print("Training finished.")

if __name__ == "__main__":
    main()