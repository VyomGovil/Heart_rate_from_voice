import torch
import numpy as np
from scipy.signal import resample, find_peaks, welch
import glob
import os
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import model from your training script
from final_train import UNet1D, LPC_ORDER

# --- Config ---
DATA_DIR = r"D:\Assignments\HeartBeatFromVoice\test_dataset"  # Ensure this points to your test data
MODEL_PATH = './checkpoints/best_model.pth'
STATS_PATH = './checkpoints/norm_stats.npz'
RESULTS_DIR = './results'
ECG_FS = 512

def load_stats(path):
    try:
        data = np.load(path)
        return {k: data[k] for k in data.files}
    except:
        print("Warning: No norm stats found.")
        return None

def get_metrics(pred, true, fs=512):
    """Calculate Paper Metrics: R-peak error, Cycle error, SDSD, Band Power"""
    
    # Peak Detection
    thresh = np.mean(true) + 0.5 * np.std(true)
    # Reduced distance slightly to catch faster heart rates
    true_peaks, _ = find_peaks(true, height=thresh, distance=int(0.25*fs))
    pred_peaks, _ = find_peaks(pred, height=thresh, distance=int(0.25*fs))
    
    if len(true_peaks) < 2 or len(pred_peaks) < 2: return None

    # 1. R-Peak Error & Cycle Error
    true_rr = np.diff(true_peaks)
    cycle_errs = []
    peak_errs = []
    
    for t_p, t_rr in zip(true_peaks[:-1], true_rr):
        # Find nearest predicted peak
        window = int(0.2 * t_rr)
        candidates = pred_peaks[np.abs(pred_peaks - t_p) < window]
        
        if len(candidates) > 0:
            nearest = candidates[np.argmin(np.abs(candidates - t_p))]
            peak_errs.append(100 * abs(nearest - t_p) / t_rr) # Eq 15
            
            # Find cycle match (rough approx)
            idx = np.where(pred_peaks == nearest)[0][0]
            if idx < len(pred_peaks) - 1:
                p_rr = pred_peaks[idx+1] - pred_peaks[idx]
                cycle_errs.append(100 * abs(p_rr - t_rr) / t_rr) # Eq 14

    # 2. HRV metrics (SDSD)
    true_rr_ms = np.diff(true_peaks) / fs * 1000
    pred_rr_ms = np.diff(pred_peaks) / fs * 1000
    
    if len(true_rr_ms) > 1 and len(pred_rr_ms) > 1:
        sdsd_true = np.std(np.diff(true_rr_ms))
        sdsd_pred = np.std(np.diff(pred_rr_ms))
        sdsd_err = abs(sdsd_pred - sdsd_true)
    else:
        sdsd_err = 0

    # 3. Band Power (FIXED FUNCTION)
    def get_bp(rr_intervals):
        if len(rr_intervals) < 5: return 0, 0
        
        # Calculate time points
        t = np.cumsum(rr_intervals) / 1000
        
        # Create uniform time axis (4Hz) within the valid range
        if t[-1] <= t[0]: return 0, 0
        t_uni = np.arange(t[0], t[-1], 0.25)
        
        if len(t_uni) == 0: return 0, 0

        # Interpolate (Lengths now match: t vs rr_intervals)
        rr_uni = np.interp(t_uni, t, rr_intervals)
        
        # Welch's method
        nperseg = min(len(rr_uni), 256)
        if nperseg == 0: return 0, 0
        
        f, psd = welch(rr_uni, fs=4.0, nperseg=nperseg)
        
        # Integrate bands
        lf = np.trapezoid(psd[(f>=0.04) & (f<=0.15)], f[(f>=0.04) & (f<=0.15)])
        hf = np.trapezoid(psd[(f>=0.15) & (f<=0.4)], f[(f>=0.15) & (f<=0.4)])
        return lf, hf

    lf_t, hf_t = get_bp(true_rr_ms)
    lf_p, hf_p = get_bp(pred_rr_ms)

    return {
        'rpeak_err': np.mean(peak_errs) if peak_errs else 0,
        'cycle_err': np.mean(cycle_errs) if cycle_errs else 0,
        'sdsd_err': sdsd_err,
        'bp_lf_err': 2 * abs(lf_p - lf_t),
        'bp_hf_err': 2 * abs(hf_p - hf_t),
        'peaks_found': len(pred_peaks),
        'peaks_real': len(true_peaks)
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

    # Load Model
    model = UNet1D(LPC_ORDER + 1, 1).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded.")
    except:
        print("Error loading model. Run training first.")
        return
    model.eval()

    stats = load_stats(STATS_PATH)
    files = glob.glob(os.path.join(DATA_DIR, '*.npz'))
    results = []

    print(f"Evaluating {len(files)} files...")

    with torch.no_grad():
        for f in files:
            try:
                d = np.load(f)
                pharynx = d['pharynx_features'].astype(np.float32)
                ecg = d['ecg_data'].astype(np.float32)
                
                if pharynx.shape[0] < 20: continue

                # Prepare data
                ecg = resample(ecg, pharynx.shape[0])
                
                # Normalize
                if stats:
                    p_norm = (pharynx - stats['p_mean']) / stats['p_std']
                    e_norm = (ecg - stats['e_mean']) / stats['e_std']
                else:
                    p_norm = (pharynx - pharynx.mean(0))/(pharynx.std(0) + 1e-6)
                    e_norm = (ecg - ecg.mean())/(ecg.std() + 1e-6)

                t_in = torch.tensor(np.nan_to_num(p_norm), dtype=torch.float32).permute(1,0).unsqueeze(0).to(device)
                
                # Inference
                pred = model(t_in).squeeze().cpu().numpy()
                
                # Metrics
                m = get_metrics(pred, e_norm)
                if m:
                    m['file'] = os.path.basename(f)
                    results.append(m)
                    
                    # Save plot for first 5
                    if len(results) >= 5 and len(results) <= 10 :
                        plt.figure(figsize=(12,4))
                        plt.plot(e_norm, label='True', alpha=0.7)
                        plt.plot(pred, label='Pred', alpha=0.7)
                        plt.title(f"{m['file']} | Peak Err: {m['rpeak_err']:.2f}%")
                        plt.legend()
                        plt.savefig(os.path.join(RESULTS_DIR, f"plot_{m['file']}.png"))
                        plt.close()

            except Exception as e:
                print(f"Skipped {f}: {e}")

    # Summary
    # ... inside main() ...

    # Summary
    if results:
        avg_rpeak = np.mean([r['rpeak_err'] for r in results])
        avg_cycle = np.mean([r['cycle_err'] for r in results])
        
        # --- ADD THIS LINE ---
        avg_sdsd = np.mean([r['sdsd_err'] for r in results]) 
        
        print("\n" + "="*40)
        print(f"Evaluation Results ({len(results)} valid samples)")
        print("="*40)
        print(f"R-Peak Timing Error: {avg_rpeak:.2f}%  (Goal: <11%)")
        print(f"Cycle Duration Error: {avg_cycle:.2f}%  (Goal: <15%)")
        
        # --- ADD THIS LINE ---
        print(f"SDSD Error:          {avg_sdsd:.2f} ms (Goal: <30ms)") 
        
        print(f"Plots saved to: {RESULTS_DIR}")
        
        with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()