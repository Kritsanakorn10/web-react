import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATA_FILE = Path(__file__).resolve().parents[1] / "data_rainfall.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "public" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "MaxAirPressure",
    "MinAirPressure",
    "AvgAirPressure8Time",
    "MaxTemp",
    "MinTemp",
    "AvgTemp",
    "Evaporation",
    "MaxHumidity",
    "MinHumidity",
    "AvgHumidity",
]

TIME_STEP = 7
SHIFT = 1
EPOCHS = 200
BATCH_SIZE = 32
LR = 0.01


def load_data():
    df = pd.read_excel(DATA_FILE)
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    x = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["Rainfall"].to_numpy(dtype=np.float32)

    y_shifted = np.concatenate([y[SHIFT:], np.full(SHIFT, np.nan, dtype=np.float32)])
    valid_mask = ~np.isnan(y_shifted)
    x = x[valid_mask]
    y_shifted = y_shifted[valid_mask]

    sequences = []
    targets = []
    for i in range(TIME_STEP, x.shape[0]):
        window = x[i - TIME_STEP : i]
        sequences.append(window)
        targets.append(y_shifted[i])

    sequences = np.stack(sequences)
    targets = np.array(targets, dtype=np.float32)
    return sequences, targets


def split_train(sequences, targets, ratio=0.8):
    num_train = int(len(targets) * ratio)
    return (
        sequences[:num_train],
        targets[:num_train],
        sequences[num_train:],
        targets[num_train:],
    )


def normalize(train_seq, test_seq):
    x_min = train_seq.min(axis=(0, 1))
    x_max = train_seq.max(axis=(0, 1))
    denom = x_max - x_min
    denom[denom == 0] = 1.0
    train_norm = (train_seq - x_min) / denom
    test_norm = (test_seq - x_min) / denom
    return train_norm, test_norm, x_min, x_max


class LSTMRegressor(nn.Module):
    def __init__(self, num_features, hidden_size=50):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return self.fc(last)


def train_model(train_x, train_y, num_features):
    device = torch.device("cpu")
    model = LSTMRegressor(num_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32).unsqueeze(-1),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{EPOCHS} - Loss: {epoch_loss / len(loader):.4f}")

    return model


def export_onnx(model, num_features):
    model.eval()
    dummy = torch.randn(1, TIME_STEP, num_features, dtype=torch.float32)
    onnx_path = OUTPUT_DIR / "rainfall_lstm.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path.as_posix(),
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )
    return onnx_path


def save_metadata(x_min, x_max):
    meta = {
        "feature_cols": FEATURE_COLS,
        "time_step": TIME_STEP,
        "shift": SHIFT,
        "x_min": x_min.tolist(),
        "x_max": x_max.tolist(),
        "target": "Rainfall",
    }
    meta_path = OUTPUT_DIR / "rainfall_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path


def main():
    sequences, targets = load_data()
    train_x, train_y, test_x, test_y = split_train(sequences, targets)
    train_x, test_x, x_min, x_max = normalize(train_x, test_x)
    num_features = train_x.shape[2]

    print("Training samples:", train_x.shape[0])
    model = train_model(train_x, train_y, num_features)
    export_onnx(model, num_features)
    save_metadata(x_min, x_max)
    print("Export complete -> public/models/rainfall_lstm.onnx")


if __name__ == "__main__":
    main()
