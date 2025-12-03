import os
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Put these near the top of your file, after imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Adjust folder name if yours is different (see note below)
DEFAULT_DATA_DIR = os.getenv(
    "HYDRAULIC_DATA_DIR",
    os.path.join(BASE_DIR, "data", "condition+monitoring+of+hydraulic+systems"),
)

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    input_dim: int,
    latent_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Autoencoder:
    model = Autoencoder(input_dim, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(X_train.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(X_val.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val = np.inf
    best_state = None

    for epoch in range(epochs):
        model.train()
        tr = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            x_hat = model(xb)
            loss = crit(x_hat, xb)
            loss.backward()
            opt.step()
            tr += loss.item() * xb.size(0)
        tr /= len(train_loader.dataset)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_hat = model(xb)
                loss = crit(x_hat, xb)
                vl += loss.item() * xb.size(0)
        vl /= len(val_loader.dataset)

        print(f"[AE] epoch {epoch+1:03d} | train {tr:.6f} | val {vl:.6f}")

        if vl < best_val:
            best_val = vl
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def reconstruction_errors(
    model: Autoencoder, X: np.ndarray, device: str = "cpu", batch_size: int = 256
) -> np.ndarray:
    model.eval()
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    errs = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            xh = model(xb)
            err = ((xh - xb) ** 2).mean(dim=1).cpu().numpy()
            errs.append(err)
    return np.concatenate(errs, axis=0)



def load_hydraulic_dataset(base_path: str = DEFAULT_DATA_DIR):

    sensor_files = [
        "PS1.txt", "PS2.txt", "PS3.txt", "PS4.txt", "PS5.txt", "PS6.txt",
        "EPS1.txt",
        "FS1.txt", "FS2.txt",
        "TS1.txt", "TS2.txt", "TS3.txt", "TS4.txt",
        "VS1.txt",
        "CE.txt", "CP.txt", "SE.txt",
    ]

    base_path = os.path.abspath(base_path)
    print(f"Loading hydraulic dataset from: {base_path}")

    sensor_dfs = []
    for fname in sensor_files:
        fpath = os.path.join(base_path, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Expected file not found: {fpath}")
        df = pd.read_csv(fpath, sep="\t", header=None)
        prefix = fname.replace(".txt", "")
        df.columns = [f"{prefix}_{i}" for i in range(df.shape[1])]
        sensor_dfs.append(df)

    X = pd.concat(sensor_dfs, axis=1)

    profile_path = os.path.join(base_path, "profile.txt")
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Expected file not found: {profile_path}")

    y = pd.read_csv(profile_path, sep="\t", header=None)
    y.columns = [
        "cooler_condition",
        "valve_condition",
        "pump_leakage",
        "accumulator_pressure",
        "stable_flag",
    ]

    cooler = y["cooler_condition"]
    valve = y["valve_condition"]
    pump_leak = y["pump_leakage"]
    accumulator = y["accumulator_pressure"]
    stable_flag = y["stable_flag"]

    healthy_mask = (
        (cooler >= 90)
        & (valve >= 90)
        & (pump_leak <= 1)
        & (accumulator >= 100)
        & (stable_flag == 0)
    )

    labels = (~healthy_mask).astype(int)

    assert X.shape[0] == y.shape[0] == labels.shape[0], "Row counts must match"

    print(f"Loaded features shape: {X.shape}")
    print(f"Loaded targets shape:  {y.shape}")
    print(f"Normal samples: {(labels == 0).sum()} | Anomalous samples: {(labels == 1).sum()}")

    dataset = None
    return X, y, labels, dataset


def prepare_data(
    X_df: pd.DataFrame,
    labels_series: pd.Series,
    test_size: float = 0.3,
    val_size: float = 0.2,
    random_state: int = 42,
    n_pca_components: int = 50,
):

    assert isinstance(X_df, pd.DataFrame)

    if isinstance(labels_series, pd.DataFrame):
        labels_series = labels_series.iloc[:, 0]
    assert isinstance(labels_series, pd.Series)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_df,
        labels_series,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_series,
    )

    train_normal_mask = y_train_full == 0
    X_train_normal = X_train_full[train_normal_mask]

    X_train, X_val = train_test_split(
        X_train_normal,
        test_size=val_size,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.transform(X_val.values)
    X_test_scaled = scaler.transform(X_test.values)

    max_possible = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
    n_components = min(n_pca_components, max_possible - 1)  # -1 to avoid degenerate case

    if n_components < n_pca_components:
        print(
            f"[WARN] Reducing PCA components from {n_pca_components} to {n_components} "
            f"because we only have {X_train_scaled.shape[0]} training samples."
        )

    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return {
        "X_train_pca": X_train_pca,
        "X_val_pca": X_val_pca,
        "X_test_pca": X_test_pca,
        "y_test": y_test.values,
        "scaler": scaler,
        "pca": pca,
        "X_test_raw": X_test,  
    }



def train_isolation_forest(
    X_train_pca: np.ndarray, contamination: float = 0.05, random_state: int = 42
) -> IsolationForest:
    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_train_pca)
    return iso


def evaluate_detector(scores: np.ndarray, y_true: np.ndarray, higher_is_anom: bool = True):
    if not higher_is_anom:
        scores = -scores

    thr = np.percentile(scores, 95)  
    y_pred = (scores >= thr).astype(int)

    print(f"Threshold: {thr:.6f}")
    print(classification_report(y_true, y_pred, digits=4))

    try:
        auc = roc_auc_score(y_true, scores)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception as e:
        print("Could not compute ROC-AUC:", e)

    return y_pred, thr


def build_anomaly_context(
    X_row_raw: pd.Series,
    y_row_targets: pd.Series,
    recon_error: float,
) -> dict:
    vals = X_row_raw.values.astype(float)
    ctx = {
        "targets": y_row_targets.to_dict(),
        "raw_features_example": {
            "mean_feature_value": float(np.mean(vals)),
            "std_feature_value": float(np.std(vals)),
        },
        "reconstruction_error": float(recon_error),
    }
    return ctx


LLM_PROMPT_TEMPLATE = """
You are an AI assistant helping an engineer monitor a hydraulic system in a smart factory.

You are given:
- High-level health labels for a 60-second cycle of the system (cooler condition, valve condition, internal pump leakage, accumulator pressure, stability flag).
- Basic statistics about the multi-sensor time series in this cycle (mean and std across all sensors).
- A scalar reconstruction error from an autoencoder anomaly detector (higher means more anomalous).

Respond with:
1. A short explanation of what seems to be wrong in this cycle.
2. 2–3 likely root causes in bullet points.
3. 3 concrete checks or actions the engineer should take next.

Be concise and technical, but understandable to a controls/maintenance engineer.

Here is the JSON input you should base your answer on:

{context_json}
"""


def generate_llm_explanation_with_ollama(
    context_dict: dict,
    model: str = "llama3",
    host: str = "http://localhost:11434",
    stream: bool = False,
):
    
    prompt = LLM_PROMPT_TEMPLATE.format(
        context_json=json.dumps(context_dict, indent=2)
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert in hydraulic systems and predictive maintenance.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": stream,
    }

    try:
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=None)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("\n[ERROR] Failed to call Ollama:", e)
        print("Make sure Ollama is running and the model is pulled, e.g.:")
        print("    ollama pull llama3")
        print("    ollama serve  (if needed)\n")
        print("Context that would have been sent:\n", json.dumps(context_dict, indent=2)[:1000])
        return None

    data = resp.json()

    # /api/chat returns: {"model": "...", "message": {"role": "assistant", "content": "..."}, "done": true, ...}
    if "message" in data and "content" in data["message"]:
        text = data["message"]["content"]
    else:
        # Fallback for any unexpected shape
        text = data.get("response", "")

    print("\n--- Ollama LLM Explanation ---")
    print(text)
    print("------------------------------\n")
    return text


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("Loading dataset...")
    X, y_targets, labels, dataset = load_hydraulic_dataset()
    feature_names = list(X.columns)

    print(f"Total samples: {len(X)}")
    print(f"Anomalies: {labels.sum()} | Normal: {(labels == 0).sum()}")

    print("Preparing data (scaler + PCA)...")
    data_dict = prepare_data(X, labels, n_pca_components=50)
    X_train_pca = data_dict["X_train_pca"]
    X_val_pca = data_dict["X_val_pca"]
    X_test_pca = data_dict["X_test_pca"]
    y_test = data_dict["y_test"]
    scaler = data_dict["scaler"]
    pca = data_dict["pca"]
    X_test_raw = data_dict["X_test_raw"]  

    print("\nTraining IsolationForest baseline...")
    iso = train_isolation_forest(X_train_pca, contamination=0.1)
    iso_scores = -iso.score_samples(X_test_pca)  
    print("\nIsolationForest performance on test set:")
    iso_pred, iso_thr = evaluate_detector(iso_scores, y_test, higher_is_anom=True)

    print("\nTraining Autoencoder advanced model...")
    input_dim = X_train_pca.shape[1]
    ae = train_autoencoder(
        X_train_pca,
        X_val_pca,
        input_dim=input_dim,
        latent_dim=32,
        epochs=50,
        batch_size=64,
        lr=1e-3,
        device=device,
    )

    print("\nScoring Autoencoder on test set...")
    ae_scores = reconstruction_errors(ae, X_test_pca, device=device)
    print("\nAutoencoder performance on test set:")
    ae_pred, ae_thr = evaluate_detector(ae_scores, y_test, higher_is_anom=True)

    k = 5
    top_idx = np.argsort(ae_scores)[::-1][:k]
    print(f"\nTop {k} anomalous cycles (by AE reconstruction error):", top_idx.tolist())

    for idx_in_test in top_idx:
        print("\n=== Anomalous cycle (test set index):", idx_in_test, "===")
        row_index = X_test_raw.index[idx_in_test]      
        x_raw = X.loc[row_index]
        y_row = y_targets.loc[row_index]

        ctx = build_anomaly_context(
            X_row_raw=x_raw,
            y_row_targets=y_row,
            recon_error=float(ae_scores[idx_in_test]),
        )

        _ = generate_llm_explanation_with_ollama(
            ctx,
            model=OLLAMA_MODEL,               
            host=OLLAMA_HOST,
            stream=False,
        )



if __name__ == "__main__":
    main()
