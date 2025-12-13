import numpy as np
import matplotlib.pyplot as plt

def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def plot_conf_hist_with_quantiles(calib, near, far, title, bins=60):
    calib = _to_numpy(calib).ravel()
    near  = _to_numpy(near).ravel()
    far   = _to_numpy(far).ravel()

    q = lambda a: (np.quantile(a, 0.05), np.quantile(a, 0.95))
    c05, c95 = q(calib)
    n05, n95 = q(near)
    f05, f95 = q(far)

    plt.figure(figsize=(8, 5))
    plt.hist(calib, bins=bins, alpha=0.55, label="Calib (ID)")
    plt.hist(near,  bins=bins, alpha=0.55, label="Near-OOD")
    plt.hist(far,   bins=bins, alpha=0.55, label="Far-OOD")

    # 5% / 95% lines
    plt.axvline(c05, linestyle="--", linewidth=2, label=f"ID 5%={c05:.3f}")

    plt.axvline(n95, linestyle=":", linewidth=2, label=f"Near 95%={n95:.3f}")

    plt.axvline(f95, linestyle="-.", linewidth=2, label=f"Far 95%={f95:.3f}")

    plt.title(title)
    plt.xlabel("Confidence (max softmax)")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)
    plt.show()