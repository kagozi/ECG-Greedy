import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from models import CWT2DCNN, DualStreamCNN, ViTFusionECG, EfficientNetFusionECG
from train_models import CWTDataset, plot_confusion_matrix_all_classes, DEVICE, PROCESSED_PATH, WAVELETS_PATH, RESULTS_PATH
import pickle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
# MODEL LOADING
############################################

def load_model_from_config(config, num_classes):
    mode = config['mode']
    model_type = config['model']

    if model_type == 'DualStream':
        model = DualStreamCNN(num_classes=num_classes, num_channels=12)
    elif model_type == 'CWT2DCNN':
        model = CWT2DCNN(num_classes=num_classes, num_channels=(24 if mode=='fusion' else 12))
    elif model_type == 'ViTFusionECG':
        model = ViTFusionECG(num_classes=num_classes, pretrained=False)
    elif model_type == 'EfficientNetFusionECG':
        model = EfficientNetFusionECG(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(DEVICE)


def apply_thresholds(scores, thresholds):
    return (scores >= thresholds).astype(int)


############################################
# PREDICTION + METRICS
############################################

def predict_and_metrics(model_name, metadata, cached_data):
    print(f"\n🔍 Evaluating {model_name}...")

    json_path = os.path.join(RESULTS_PATH, f"results_{model_name}.json")
    ckpt_path = os.path.join(RESULTS_PATH, f"best_{model_name}.pth")
    results = json.load(open(json_path))
    config = results["config"]
    thresholds = np.array(results["optimal_thresholds"])
    auc = results.get("macro_auc", -np.inf)

    model = load_model_from_config(config, metadata["num_classes"])
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in cached_data["loader"]:
            if config["mode"] == "both":
                (x1, x2), _ = batch
                out = model(x1.to(DEVICE), x2.to(DEVICE))
            else:
                x, _ = batch
                out = model(x.to(DEVICE))
            preds.append(torch.sigmoid(out).cpu().numpy())

    preds = np.vstack(preds)
    pred_bin = apply_thresholds(preds, thresholds)

    # Compute metrics
    auc = roc_auc_score(cached_data["y_test"], preds, average="macro")
    f1 = f1_score(cached_data["y_test"], pred_bin, average="macro")

    return preds, thresholds, auc, f1


############################################
# CONFUSION MATRIX
############################################

def save_confusion(model_name, y_true, y_pred, metadata):
    save_path = os.path.join(RESULTS_PATH, f"confusion_{model_name}.png")
    plot_confusion_matrix_all_classes(
        y_true, y_pred, metadata["classes"],
        save_path=save_path,
        title=f"Confusion Matrix - {model_name}"
    )
    print(f"📁 Saved: {save_path}")


############################################
# ENSEMBLE LOGIC
############################################

def ensemble_top_k(model_results, k, metadata, y_test):
    top_k = sorted(model_results, key=lambda x: x["auc"], reverse=True)[:k]

    print(f"\n✅ Ensemble uses top {k}:")
    for r in top_k:
        print(f"- {r['name']} | AUC={r['auc']:.4f}")

    avg_scores = np.mean([r["preds"] for r in top_k], axis=0)
    avg_thresh = np.mean([r["thresholds"] for r in top_k], axis=0)
    pred_bin = apply_thresholds(avg_scores, avg_thresh)

    auc = roc_auc_score(y_test, avg_scores, average="macro")
    f1 = f1_score(y_test, pred_bin, average="macro")

    save_confusion(f"ensemble_top{k}", y_test, pred_bin, metadata)

    return auc, f1


############################################
# MAIN
############################################

if __name__ == "__main__":
    with open(os.path.join(PROCESSED_PATH, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    y_test = np.load(os.path.join(PROCESSED_PATH, "y_test.npy"))
    test_scalos = os.path.join(WAVELETS_PATH, "test_scalograms.npy")
    test_phasos = os.path.join(WAVELETS_PATH, "test_phasograms.npy")

    # Single dataset and loader reuse ✅
    dataset = CWTDataset(test_scalos, test_phasos, y_test, mode="both")
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    cached_data = {"loader": loader, "y_test": y_test}

    # Detect models
    model_names = [
        f.split("results_")[1].replace(".json", "")
        for f in os.listdir(RESULTS_PATH)
        if "results_" in f
    ]

    # Evaluate all models
    results = []
    for m in model_names:
        preds, thresh, auc, f1 = predict_and_metrics(m, metadata, cached_data)
        results.append({
            "name": m, "preds": preds,
            "thresholds": thresh,
            "auc": auc,
            "f1": f1
        })
        pred_bin = apply_thresholds(preds, thresh)
        save_confusion(m, y_test, pred_bin, metadata)

    # Plot model performance comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=[r["name"] for r in results],
        y=[r["auc"] for r in results]
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Macro AUC Comparison Across Models")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "model_auc_comparison.png"))
    plt.close()
    print("📈 Saved model performance plot")

    # Perform ensemble
    ens_auc, ens_f1 = ensemble_top_k(results, k=3, metadata=metadata, y_test=y_test)
    print(f"\n🏆 Ensemble Performance — AUC={ens_auc:.4f}, F1={ens_f1:.4f}")
