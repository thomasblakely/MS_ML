
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd


class CNN(nn.Module):
    def __init__(self, input_len=30):
        super().__init__()
        self.c1 = nn.Conv1d(1, 16, 3, 1, 1)
        self.b1 = nn.BatchNorm1d(16)
        self.c2 = nn.Conv1d(16, 32, 3, 1, 1)
        self.b2 = nn.BatchNorm1d(32)
        self.c3 = nn.Conv1d(32, 64, 3, 1, 1)
        self.b3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2, 2)
        with torch.no_grad():
            d = torch.zeros(1, 1, input_len)
            d = self.pool(F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(d))))))))))
            flat = d.numel()
        self.f1 = nn.Linear(flat, 64)
        self.drop = nn.Dropout(0.3)
        self.f2 = nn.Linear(64, 16)
        self.f3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = F.relu(self.b3(self.c3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.f1(x))
        x = self.drop(x)
        x = F.relu(self.f2(x))
        x = self.drop(x)
        return self.f3(x)


def train_one_fold(Xtr, ytr, Xte, yte, epochs=50, batch=32, lr=1e-3, pos_weight=None, device='cpu'):
    # tensors
    Xt = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(1)
    yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    Xe = torch.tensor(Xte, dtype=torch.float32).unsqueeze(1)
    ye = torch.tensor(yte, dtype=torch.float32).unsqueeze(1)

    tr_loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=True)
    te_loader = DataLoader(TensorDataset(Xe, ye), batch_size=batch, shuffle=False)

    model = CNN(input_len=Xtr.shape[1]).to(device)
    if pos_weight is None:
        criterion = nn.BCEWithLogitsLoss()
    else:
        # pos_weight balances positive examples; it should be a scalar tensor on the correct device
        if torch.is_tensor(pos_weight):
            pw = pos_weight.to(device)
        else:
            pw = torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()

    model.eval()
    y_true, y_pred, probs = [], [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            p = torch.sigmoid(logits)
            pred = (logits > 0).float()
            y_true.extend(yb.view(-1).cpu().numpy().tolist())
            y_pred.extend(pred.view(-1).cpu().numpy().tolist())
            probs.extend(p.view(-1).cpu().numpy().tolist())

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = float('nan')

    return precision, recall, f1, auc


def main():
    # Load prepared training data
    x = np.load('output/ready/x_train.npy')
    y = np.load('output/ready/y_train.npy')

    # ********ISSUE*********
    # INAPPROPRIATE CV METHOD: Using StratifiedKFold with shuffle on temporal data
    # Filename says "lopo" (leave-one-patient-out) but this is NOT LOPO
    # Sliding windows from same patient can appear in both train and test (data leakage)
    # Should use GroupKFold with patient IDs or true LOPO
    # 20-fold stratified CV to preserve class balance in each fold
    skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    rows = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(x, y), start=1):
        Xtr, ytr = x[tr_idx], y[tr_idx]
        Xte, yte = x[te_idx], y[te_idx]

        # Compute positive class weight for BCEWithLogitsLoss: pos_weight = (N_neg / N_pos)
        pos_ratio = float((ytr == 1).mean()) if len(ytr) > 0 else 0.0
        if pos_ratio > 0 and pos_ratio < 1:
            pw = (1.0 - pos_ratio) / pos_ratio
        else:
            pw = None  # no weighting if only one class present

        p, r, f1, auc = train_one_fold(
            Xtr, ytr, Xte, yte,
            epochs=8, batch=32, lr=1e-3,
            pos_weight=pw,
            device=device,
        )

        rows.append({
            'fold': fold,
            'precision': p,
            'recall': r,
            'f1': f1,
            'auc': auc,
            'pos_ratio_train': pos_ratio,
        })

        print(f"fold {fold:02d}: P {p:.2f} R {r:.2f} F1 {f1:.2f} AUC {auc:.2f}")

    df = pd.DataFrame(rows)
    out_csv = 'result/cv20_metrics.csv'
    df.to_csv(out_csv, index=False)
    print(f"saved to {out_csv}")
    print(
        "Mean P {:.2f} R {:.2f} F1 {:.2f} AUC {:.2f}".format(
            df['precision'].mean(), df['recall'].mean(), df['f1'].mean(), df['auc'].mean()
        )
    )


if __name__ == "__main__":
    main()
