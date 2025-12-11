import pandas as pd
import glob
import os

# find all csv logs
csv_files = glob.glob("checkpoints/*_metrics.csv")

rows = []
for path in csv_files:
    df = pd.read_csv(path)
    # take the last validation epoch
    best = df[df["phase"] == "tune"].sort_values("val_loss").iloc[0]

    
    backbone = os.path.basename(path).replace("_metrics.csv", "")

    rows.append({
        "backbone": backbone,
        "val_super_acc": best["val_super_acc"],
        "val_sub_acc": best["val_sub_acc"],
        "val_loss": best["val_loss"]
    })

out = pd.DataFrame(rows)
print(out)

out.to_csv("checkpoints/model_comparison_table.csv", index=False)