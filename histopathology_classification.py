# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
import os
import time
import copy
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    classification_report, confusion_matrix,
    cohen_kappa_score, roc_auc_score,
    accuracy_score, f1_score
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = "/kaggle/input/datasets/imrankhan77/nct-crc-he-100k/NCT-CRC-HE-100K"
OUTPUT_DIR = "/kaggle/working/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Global Hyperparameters ───────────────────────────────────────────────────
IMG_SIZE        = 224
BATCH_SIZE      = 64
NUM_CLASSES     = 9
TRAIN_SPLIT     = 0.72          # 72% train  (~72,000 images)
VAL_SPLIT       = 0.18          # 18% validation (~18,000 images)
TEST_SPLIT      = 0.10          # 10% test — held out, never seen during training (~10,000)

# Phase 1 — head only
P1_EPOCHS       = 5
P1_LR           = 1e-3

# Phase 2 — full fine-tuning
P2_EPOCHS       = 25
P2_LR           = 1e-4
WEIGHT_DECAY    = 1e-5
DROPOUT_RATE    = 0.3
GRAD_CLIP       = 1.0           # Max gradient norm (exploding gradient prevention)
PATIENCE        = 5             # Early stopping patience

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"  Device      : {DEVICE}")
print(f"  CUDA avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
print(f"{'='*60}\n")

CLASS_NAMES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA PIPELINE  (Section 3 — Data Augmentation as Regularization)
# ─────────────────────────────────────────────────────────────────────────────
# ImageNet normalization stats (required because model is pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Training transforms — augmentation for regularization
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),   # zoom variation
    transforms.RandomHorizontalFlip(p=0.5),                      # tissue has no fixed orientation
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),                       # rotational invariance
    transforms.ColorJitter(                                      # H&E stain variation
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.05
    ),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),   # focus variation
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Validation / Test — no augmentation, only normalization
val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_dataloaders(data_dir):
    """
    Load NCT-CRC-HE-100K from a single folder and split into train/val/test.
    This Kaggle source has no separate test set, so we do a 72/18/10 split.

    Returns: train_loader, val_loader, test_loader, class_names
    """
    print("[1/6] Building data loaders ...")

    # Load full dataset — we use train_transforms; val/test subsets get overridden below
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    total        = len(full_dataset)

    test_size  = int(total * TEST_SPLIT)
    val_size   = int(total * VAL_SPLIT)
    train_size = total - val_size - test_size

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Give val and test their own dataset copy with no augmentation
    val_ds_copy  = copy.deepcopy(full_dataset)
    test_ds_copy = copy.deepcopy(full_dataset)
    val_ds_copy.transform  = val_test_transforms
    test_ds_copy.transform = val_test_transforms

    # Wrap in Subset so indices still align
    from torch.utils.data import Subset
    val_ds  = Subset(val_ds_copy,  val_ds.indices)
    test_ds = Subset(test_ds_copy, test_ds.indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"   Total dataset      : {total:,} images across {len(full_dataset.classes)} classes")
    print(f"   Training samples   : {train_size:,}  (72%)")
    print(f"   Validation samples : {val_size:,}  (18%)")
    print(f"   Test samples       : {test_size:,}  (10% — held out)")
    print(f"   Classes            : {full_dataset.classes}\n")

    return train_loader, val_loader, test_loader, full_dataset.classes


def show_sample_images(data_dir, n=3):
    """Display sample images from each class (exploratory data analysis)."""
    ds = datasets.ImageFolder(root=data_dir, transform=val_test_transforms)
    fig, axes = plt.subplots(len(ds.classes), n, figsize=(n * 3, len(ds.classes) * 3))
    class_to_idx = ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    shown = {c: 0 for c in range(len(ds.classes))}
    for img, label in ds:
        if shown[label] < n:
            ax = axes[label][shown[label]]
            img_np = img.permute(1, 2, 0).numpy()
            img_np = img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img_np = np.clip(img_np, 0, 1)
            ax.imshow(img_np)
            ax.set_title(idx_to_class[label], fontsize=9)
            ax.axis('off')
            shown[label] += 1
        if all(v >= n for v in shown.values()):
            break

    plt.suptitle("Sample Images per Class — NCT-CRC-HE-100K", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sample_images.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("   Saved: sample_images.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL ARCHITECTURE  (Section 1 — Model Selection & Architecture)
# ─────────────────────────────────────────────────────────────────────────────
class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 with custom classifier head for 9-class tissue classification.

    Architecture justification:
      - MBConv blocks with Swish activation: smooth gradients, no saturation
      - Squeeze-and-Excitation: channel attention for fine-grained texture features
      - Compound scaling: optimal depth/width/resolution balance
      - ~5.3M parameters: 4.8x smaller than ResNet-50 with higher accuracy
      - ImageNet pretraining: strong initialization for histology feature extraction

    Forward pass formulation:
      z_l = W_l · a_{l-1} + b_l
      a_l = Swish(z_l) = z_l · σ(z_l)
      ŷ   = Softmax(W_fc · a_L)      (final layer, 9 classes)
    """

    def __init__(self, num_classes=9, dropout_rate=0.3, pretrained=True):
        super(EfficientNetClassifier, self).__init__()

        # Load pretrained EfficientNet-B0 backbone
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Retrieve in_features of the original classifier
        in_features = self.backbone.classifier[1].in_features  # 1280

        # Replace default classifier with custom head
        # Includes extra Dropout (regularization — Section 3)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x):
        """
        Forward Pass:
          Input  x : (B, 3, 224, 224)
          Output   : (B, 9) raw logits — cross-entropy loss expects logits, not softmax
        """
        return self.backbone(x)

    def freeze_backbone(self):
        """Phase 1: Freeze all layers except the classifier head."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        print("   Backbone FROZEN — training classifier head only.")

    def unfreeze_all(self):
        """Phase 2: Unfreeze all layers for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("   All layers UNFROZEN — full fine-tuning.")

    def count_parameters(self):
        total  = sum(p.numel() for p in self.parameters())
        train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - train
        return total, train, frozen


def print_architecture(model):
    """Print a concise architecture summary."""
    total, train, frozen = model.count_parameters()
    print("\n" + "="*60)
    print("  MODEL ARCHITECTURE: EfficientNet-B0")
    print("="*60)
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable params    : {train:,}")
    print(f"  Frozen params       : {frozen:,}")
    print(f"  Input size          : 3 × {IMG_SIZE} × {IMG_SIZE}")
    print(f"  Output classes      : {model.num_classes}")
    print(f"  Activation          : Swish (x · σ(x))")
    print(f"  Regularization      : Dropout(p={DROPOUT_RATE}), L2 (wd={WEIGHT_DECAY})")
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOSS FUNCTION & OPTIMIZER  (Section 2 — Mathematical Explanation)
# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer_and_scheduler(model, lr, weight_decay, total_epochs):
    """
    Loss Function: Cross-Entropy
      L = -Σ y_k · log(ŷ_k)
      where y_k = one-hot label, ŷ_k = softmax(z_k)

    Optimizer: AdamW (Adam + decoupled L2 weight decay)
      m_t = β₁·m_{t-1} + (1-β₁)·g_t          (1st moment — mean)
      v_t = β₂·v_{t-1} + (1-β₂)·g_t²         (2nd moment — variance)
      m̂_t = m_t / (1 - β₁ᵗ)                   (bias-corrected)
      v̂_t = v_t / (1 - β₂ᵗ)                   (bias-corrected)
      W_{t+1} = W_t - α · m̂_t/(√v̂_t + ε)     (parameter update)

    Scheduler: CosineAnnealingLR
      α_t = α_min + 0.5(α_max - α_min)(1 + cos(π·t/T_max))
      Reduces LR smoothly, avoids sharp drops, helps convergence.
    """
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay   # L2 regularization (Section 3)
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=1e-7
    )

    return criterion, optimizer, scheduler


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING ENGINE  (Sections 2, 3, 4)
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Stops training when validation loss stops improving.
    Saves the best model checkpoint automatically.
    """
    def __init__(self, patience=5, min_delta=1e-4, path='best_model.pth'):
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = path
        self.best_loss  = np.inf
        self.counter    = 0
        self.triggered  = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    """
    One full pass through training data.
    Includes gradient clipping (Section 5 — exploding gradient prevention).

    Backpropagation steps per mini-batch:
      1. Forward pass:  ŷ = model(x)
      2. Compute loss:  L = CrossEntropy(ŷ, y)
      3. Zero grads:    optimizer.zero_grad()
      4. Backward pass: L.backward()  → computes ∂L/∂W for all W
      5. Clip gradients (prevent exploding): clip_grad_norm_(params, max_norm)
      6. Update weights: optimizer.step()  → W = W - α·∂L/∂W (via Adam)
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()           # clear accumulated gradients

        outputs = model(imgs)           # forward pass
        loss = criterion(outputs, labels)  # compute cross-entropy loss

        loss.backward()                 # backpropagation

        # Gradient clipping — prevents exploding gradients (Section 5)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()               # Adam weight update

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on val/test set. Returns loss, accuracy."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)

    return running_loss / total, correct / total


def run_training_phase(model, train_loader, val_loader, criterion,
                       optimizer, scheduler, epochs, phase_name,
                       device, grad_clip, early_stop=None):
    """
    Generic training loop for Phase 1 (head only) and Phase 2 (full fine-tuning).
    Returns history dict with loss/accuracy curves.
    """
    print(f"\n{'─'*60}")
    print(f"  {phase_name}")
    print(f"{'─'*60}")

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip)
        val_loss,   val_acc   = evaluate(
            model, val_loader, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        elapsed = time.time() - t0
        print(f"  Epoch [{epoch:2d}/{epochs}] "
              f"| Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% "
              f"| LR: {current_lr:.2e} "
              f"| {elapsed:.1f}s")

        if early_stop:
            early_stop(val_loss, model)
            if early_stop.triggered:
                print(f"\n  Early stopping triggered at epoch {epoch}.")
                break

    return history


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOTTING  (Section 4 — Performance Evaluation)
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history_p1, history_p2):
    """
    Plot training & validation loss/accuracy curves across both phases.
    Satisfies assignment requirement for accuracy/loss curves.
    """
    # Concatenate both phases
    train_loss = history_p1['train_loss'] + history_p2['train_loss']
    val_loss   = history_p1['val_loss']   + history_p2['val_loss']
    train_acc  = history_p1['train_acc']  + history_p2['train_acc']
    val_acc    = history_p1['val_acc']    + history_p2['val_acc']
    lr_curve   = history_p1['lr']         + history_p2['lr']
    epochs     = range(1, len(train_loss) + 1)
    p1_end     = len(history_p1['train_loss'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('EfficientNet-B0 Training Curves — NCT-CRC-HE-100K',
                 fontsize=14, fontweight='bold')

    # Loss curve
    axes[0].plot(epochs, train_loss, label='Train Loss', color='#E74C3C', linewidth=2)
    axes[0].plot(epochs, val_loss,   label='Val Loss',   color='#2ECC71', linewidth=2)
    axes[0].axvline(x=p1_end, color='gray', linestyle='--', alpha=0.7, label='Phase boundary')
    axes[0].set_title('Loss Curve', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy curve
    axes[1].plot(epochs, [a*100 for a in train_acc], label='Train Acc', color='#E74C3C', linewidth=2)
    axes[1].plot(epochs, [a*100 for a in val_acc],   label='Val Acc',   color='#2ECC71', linewidth=2)
    axes[1].axvline(x=p1_end, color='gray', linestyle='--', alpha=0.7, label='Phase boundary')
    axes[1].set_title('Accuracy Curve', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Learning rate schedule
    axes[2].semilogy(epochs, lr_curve, color='#9B59B6', linewidth=2)
    axes[2].axvline(x=p1_end, color='gray', linestyle='--', alpha=0.7, label='Phase boundary')
    axes[2].set_title('Learning Rate Schedule (Cosine Annealing)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate (log scale)')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("   Saved: training_curves.png")


def plot_confusion_matrix(y_true, y_pred, class_names, model_name="EfficientNet-B0"):
    """Normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        [f'{model_name} — Confusion Matrix (Counts)',
         f'{model_name} — Confusion Matrix (Normalized)'],
        ['d', '.2f']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5, cbar=True)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_{model_name.replace('-','')}.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"   Saved: confusion_matrix_{model_name}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPREHENSIVE EVALUATION  (Section 4 — Performance Evaluation)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def full_evaluation(model, test_loader, criterion, class_names, device, model_name="EfficientNet-B0"):
  
    print(f"\n{'='*60}")
    print(f"  EVALUATION: {model_name}")
    print(f"{'='*60}")

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    running_loss, total = 0.0, 0

    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        running_loss += loss.item() * imgs.size(0)
        total        += labels.size(0)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # Core metrics
    acc     = accuracy_score(all_labels, all_preds)
    f1      = f1_score(all_labels, all_preds, average='macro')
    kappa   = cohen_kappa_score(all_labels, all_preds)
    avg_loss = running_loss / total

    # AUC-ROC (one-vs-rest)
    y_bin = label_binarize(all_labels, classes=list(range(len(class_names))))
    auc   = roc_auc_score(y_bin, all_probs, average='macro', multi_class='ovr')

    print(f"\n  Test Loss         : {avg_loss:.4f}")
    print(f"  Accuracy          : {acc*100:.2f}%")
    print(f"  Macro F1-Score    : {f1*100:.2f}%")
    print(f"  Cohen's Kappa (κ) : {kappa:.4f}")
    print(f"  Macro AUC-ROC     : {auc:.4f}")

    print("\n  Per-Class Classification Report:")
    print("  " + "-"*55)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # Confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_names, model_name)

    return {
        'accuracy': acc, 'f1': f1, 'kappa': kappa, 'auc': auc,
        'loss': avg_loss, 'preds': all_preds, 'labels': all_labels,
        'probs': all_probs
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. GRAD-CAM VISUALIZATION  (Section 6 — Application Perspective)
# ─────────────────────────────────────────────────────────────────────────────
def visualize_gradcam(model, test_loader, class_names, device, n_samples=3):
    """
    Gradient-weighted Class Activation Maps (Grad-CAM).
    Shows WHICH regions of the tissue patch the model uses to classify —
    critical for clinical interpretability and trust in the CAP system.

    Application: In a real pathology workflow, Grad-CAM overlays help
    pathologists verify that the model focuses on biologically meaningful
    regions (e.g., tumor glands, lymphocyte clusters) rather than artifacts.

    Requires: pip install grad-cam
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("   [Grad-CAM] Install via: pip install grad-cam")
        return

    # Target the last convolutional block of EfficientNet-B0
    target_layers = [model.backbone.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    model.eval()
    imgs_batch, labels_batch = next(iter(test_loader))
    imgs_batch = imgs_batch[:n_samples].to(device)
    labels_batch = labels_batch[:n_samples]

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 4))
    fig.suptitle('Grad-CAM Visualization — EfficientNet-B0\n'
                 'Highlights regions the model uses for tissue classification',
                 fontsize=12, fontweight='bold')

    with torch.no_grad():
        outputs = model(imgs_batch)
        _, preds = torch.max(outputs, 1)

    for i in range(n_samples):
        img_tensor = imgs_batch[i:i+1]
        true_label = labels_batch[i].item()
        pred_label = preds[i].item()

        targets = [ClassifierOutputTarget(pred_label)]
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

        # Denormalize image for display
        img_np = imgs_batch[i].cpu().permute(1, 2, 0).numpy()
        img_np = img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img_np = np.clip(img_np, 0, 1).astype(np.float32)

        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        axes[i][0].imshow(img_np)
        axes[i][0].set_title(f"Original\nTrue: {class_names[true_label]}", fontsize=9)
        axes[i][0].axis('off')

        axes[i][1].imshow(grayscale_cam, cmap='jet')
        axes[i][1].set_title("Grad-CAM Heatmap", fontsize=9)
        axes[i][1].axis('off')

        color = 'green' if pred_label == true_label else 'red'
        axes[i][2].imshow(cam_image)
        axes[i][2].set_title(f"Overlay\nPred: {class_names[pred_label]}",
                             fontsize=9, color=color)
        axes[i][2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/gradcam_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("   Saved: gradcam_visualization.png")
    cam.remove_hooks()


# ─────────────────────────────────────────────────────────────────────────────
# 8. BASELINE COMPARISON — ResNet-50  (Section 5 — Critical Analysis)
# ─────────────────────────────────────────────────────────────────────────────
def build_resnet50(num_classes=9, dropout_rate=0.3):
    """
    ResNet-50 baseline for comparison against EfficientNet-B0.
    Identical training protocol: pretrained, same head, same optimizer.

    ResNet-50 addresses vanishing gradients via residual connections:
      H(x) = F(x) + x   (identity shortcut)
    Gradient during backprop:
      ∂L/∂x_l = ∂L/∂x_L · Π_{k=l}^{L} (1 + ∂F_k/∂x_k)
    The +1 term ensures gradients are never fully zeroed out.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features   # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    return model


def compare_models(efficientnet_metrics, resnet_metrics):
    """
    Print and plot side-by-side comparison table.
    Satisfies Section 5 — Compare model performance with existing architecture.
    """
    print(f"\n{'='*65}")
    print("  CRITICAL ANALYSIS — Model Comparison")
    print(f"{'='*65}")
    print(f"  {'Metric':<25} {'EfficientNet-B0':>18} {'ResNet-50':>15}")
    print(f"  {'─'*58}")

    metrics = [
        ('Test Accuracy (%)',   efficientnet_metrics['accuracy']*100, resnet_metrics['accuracy']*100, '{:.2f}'),
        ('Macro F1-Score (%)',  efficientnet_metrics['f1']*100,       resnet_metrics['f1']*100,       '{:.2f}'),
        ("Cohen's Kappa",       efficientnet_metrics['kappa'],        resnet_metrics['kappa'],        '{:.4f}'),
        ('Macro AUC-ROC',       efficientnet_metrics['auc'],          resnet_metrics['auc'],          '{:.4f}'),
        ('Test Loss',           efficientnet_metrics['loss'],         resnet_metrics['loss'],         '{:.4f}'),
    ]

    for name, eff_val, res_val, fmt in metrics:
        winner = '<-- Better' if eff_val > res_val else '          '
        if name == 'Test Loss':
            winner = '<-- Better' if eff_val < res_val else '          '
        print(f"  {name:<25} {fmt.format(eff_val):>18} {fmt.format(res_val):>15}  {winner}")

    print(f"\n  {'Parameters':<25} {'5.3M':>18} {'25.6M':>15}  EfficientNet 4.8× smaller")
    print(f"  {'FLOPs (approx.)':<25} {'0.39B':>18} {'4.1B':>15}  EfficientNet 10.5× fewer")
    print(f"{'='*65}\n")

    # Bar chart comparison
    labels  = ['Accuracy (%)', 'Macro F1 (%)', "Kappa ×100", 'AUC-ROC ×100']
    eff_vals = [efficientnet_metrics['accuracy']*100,
                efficientnet_metrics['f1']*100,
                efficientnet_metrics['kappa']*100,
                efficientnet_metrics['auc']*100]
    res_vals = [resnet_metrics['accuracy']*100,
                resnet_metrics['f1']*100,
                resnet_metrics['kappa']*100,
                resnet_metrics['auc']*100]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, eff_vals, width, label='EfficientNet-B0',
                   color='#2E74B5', edgecolor='white')
    bars2 = ax.bar(x + width/2, res_vals, width, label='ResNet-50',
                   color='#E74C3C', edgecolor='white')

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('EfficientNet-B0 vs ResNet-50 — Performance Comparison\n(NCT-CRC-HE-7K Test Set)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.set_ylim([min(min(eff_vals), min(res_vals)) - 2, 101])
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.annotate(f'{bar.get_height():.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("   Saved: model_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. GRADIENT MONITORING  (Section 5 — Vanishing/Exploding Gradient Analysis)
# ─────────────────────────────────────────────────────────────────────────────
def monitor_gradients(model, loader, criterion, device, n_batches=5):
    """
    Computes per-layer mean absolute gradient to detect vanishing/exploding.
    Section 5 requires discussing this — this gives you concrete data to cite.

    Expected result for EfficientNet:
      - Gradients should be non-vanishing (>1e-7) in early layers due to skip connections
      - Gradients should not explode (>10) due to gradient clipping + BN
    """
    model.train()
    grad_stats = {}

    for batch_idx, (imgs, labels) in enumerate(loader):
        if batch_idx >= n_batches:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                g = param.grad.abs().mean().item()
                if name not in grad_stats:
                    grad_stats[name] = []
                grad_stats[name].append(g)

    # Average over batches and aggregate by layer block
    block_grads = {}
    for name, vals in grad_stats.items():
        block = name.split('.')[2] if len(name.split('.')) > 2 else name.split('.')[0]
        block_grads.setdefault(block, []).append(np.mean(vals))

    avg_by_block = {k: np.mean(v) for k, v in block_grads.items()}

    print("\n  Gradient Analysis (mean |grad| per layer block):")
    print("  " + "-"*50)
    for block, grad_val in list(avg_by_block.items())[:15]:
        status = "OK"
        if grad_val < 1e-7:
            status = "⚠ VANISHING"
        elif grad_val > 10:
            status = "⚠ EXPLODING"
        print(f"  {block:<30} {grad_val:.6f}  {status}")

    # Plot
    blocks = list(avg_by_block.keys())[:15]
    values = [avg_by_block[b] for b in blocks]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#E74C3C' if v < 1e-7 or v > 10 else '#2E74B5' for v in values]
    ax.bar(range(len(blocks)), values, color=colors, edgecolor='white')
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels(blocks, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean |Gradient|', fontsize=11)
    ax.set_title('Gradient Flow Analysis — EfficientNet-B0\n'
                 'Blue=Healthy, Red=Vanishing/Exploding',
                 fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/gradient_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("   Saved: gradient_analysis.png")

    return avg_by_block


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  DL ASSIGNMENT — NCT-CRC-HE-100K — EfficientNet-B0")
    print("="*60)

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_names = build_dataloaders(DATA_DIR)
    show_sample_images(DATA_DIR, n=3)

    # ── Step 2: Build EfficientNet-B0 Model ──────────────────────────────────
    print("\n[2/6] Building EfficientNet-B0 model ...")
    efficientnet = EfficientNetClassifier(
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        pretrained=True
    ).to(DEVICE)
    print_architecture(efficientnet)

    # ── Step 3: Phase 1 — Head Training ──────────────────────────────────────
    print("\n[3/6] Phase 1 — Classifier head training ...")
    efficientnet.freeze_backbone()
    criterion, optimizer_p1, scheduler_p1 = build_optimizer_and_scheduler(
        efficientnet, lr=P1_LR, weight_decay=WEIGHT_DECAY, total_epochs=P1_EPOCHS)

    history_p1 = run_training_phase(
        efficientnet, train_loader, val_loader, criterion,
        optimizer_p1, scheduler_p1,
        epochs=P1_EPOCHS, phase_name="PHASE 1: Head Training",
        device=DEVICE, grad_clip=GRAD_CLIP
    )

    # ── Step 4: Phase 2 — Full Fine-Tuning ───────────────────────────────────
    print("\n[4/6] Phase 2 — Full fine-tuning ...")
    efficientnet.unfreeze_all()
    early_stop = EarlyStopping(patience=PATIENCE,
                               path=f"{OUTPUT_DIR}/best_efficientnet.pth")
    criterion, optimizer_p2, scheduler_p2 = build_optimizer_and_scheduler(
        efficientnet, lr=P2_LR, weight_decay=WEIGHT_DECAY, total_epochs=P2_EPOCHS)

    history_p2 = run_training_phase(
        efficientnet, train_loader, val_loader, criterion,
        optimizer_p2, scheduler_p2,
        epochs=P2_EPOCHS, phase_name="PHASE 2: Full Fine-Tuning",
        device=DEVICE, grad_clip=GRAD_CLIP, early_stop=early_stop
    )

    # Load best checkpoint
    efficientnet.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_efficientnet.pth",
                                            map_location=DEVICE))
    print("   Loaded best EfficientNet checkpoint.")

    # ── Step 5: Plot Training Curves ─────────────────────────────────────────
    print("\n[5/6] Plotting training curves ...")
    plot_training_curves(history_p1, history_p2)

    # ── Step 6: Full Evaluation on Test Set ──────────────────────────────────
    print("\n[6/6] Evaluating on NCT-CRC-HE-7K test set ...")
    eff_metrics = full_evaluation(efficientnet, test_loader, criterion,
                                  class_names, DEVICE, "EfficientNet-B0")

    # ── Step 7: Gradient Analysis ─────────────────────────────────────────────
    print("\n[+] Gradient flow analysis ...")
    monitor_gradients(efficientnet, train_loader, criterion, DEVICE)

    # ── Step 8: Grad-CAM Visualization ────────────────────────────────────────
    print("\n[+] Grad-CAM visualization ...")
    visualize_gradcam(efficientnet, test_loader, class_names, DEVICE, n_samples=3)

    # ── Step 9: Baseline — ResNet-50 ─────────────────────────────────────────
    print("\n[+] Training ResNet-50 baseline for comparison ...")
    resnet = build_resnet50(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE).to(DEVICE)

    # Freeze backbone, train head
    for name, param in resnet.named_parameters():
        param.requires_grad = 'fc' in name
    crit_r, opt_r, sch_r = build_optimizer_and_scheduler(
        resnet, lr=P1_LR, weight_decay=WEIGHT_DECAY, total_epochs=P1_EPOCHS)
    run_training_phase(resnet, train_loader, val_loader, crit_r, opt_r, sch_r,
                       epochs=P1_EPOCHS, phase_name="ResNet-50 Phase 1",
                       device=DEVICE, grad_clip=GRAD_CLIP)

    # Full fine-tuning
    for param in resnet.parameters():
        param.requires_grad = True
    es_r = EarlyStopping(patience=PATIENCE, path=f"{OUTPUT_DIR}/best_resnet.pth")
    crit_r, opt_r, sch_r = build_optimizer_and_scheduler(
        resnet, lr=P2_LR, weight_decay=WEIGHT_DECAY, total_epochs=P2_EPOCHS)
    run_training_phase(resnet, train_loader, val_loader, crit_r, opt_r, sch_r,
                       epochs=P2_EPOCHS, phase_name="ResNet-50 Phase 2",
                       device=DEVICE, grad_clip=GRAD_CLIP, early_stop=es_r)

    resnet.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_resnet.pth",
                                      map_location=DEVICE))
    res_metrics = full_evaluation(resnet, test_loader, crit_r,
                                  class_names, DEVICE, "ResNet-50")

    # ── Step 10: Model Comparison ─────────────────────────────────────────────
    compare_models(eff_metrics, res_metrics)

    print("\n" + "="*60)
    print("  ALL DONE! Outputs saved to:", OUTPUT_DIR)
    print("  Files generated:")
    print("    sample_images.png         — EDA visualization")
    print("    training_curves.png       — Loss/accuracy/LR curves")
    print("    confusion_matrix_*.png    — Confusion matrices")
    print("    gradient_analysis.png     — Vanishing/exploding gradient")
    print("    gradcam_visualization.png — Interpretability (Grad-CAM)")
    print("    model_comparison.png      — EfficientNet vs ResNet-50")
    print("    best_efficientnet.pth     — Best model weights")
    print("    best_resnet.pth           — ResNet-50 weights")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
