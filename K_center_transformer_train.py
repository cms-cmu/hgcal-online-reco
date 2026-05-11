import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.data import ConcatDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – enable 3-D
from k_center_model import GPTConfig, SetTransformer

# Debug mode – process only a small subset of events for quick iteration
DEBUG_MODE = False
DEBUG_SAMPLES = 100

# ===========================
# Dataset Definition
# ===========================
class SyntheticSet2GraphDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        # Per-node features: [node_distance_sum, node_distance_mean, sum_energy, mean_energy, std_energy, x, y, z] ⇒ 8 dims
        raw = data["X_all_mod"].astype(np.float32)
        # Use all 8 features
        self.input_data = raw
        
        # Event-level cluster summary labels (N, max_k, 9) where first 3 are center coordinates
        self.cluster_summary_labels = data["y_all"].astype(np.float32)
        # active flags (N,E) where 1 = active, 0 = inactive
        self.active_flags = data["active_flags"].astype(np.float32)
        # Store per-event k value (number of clusters) for stratified splits
        self.k_values = data["k_all"].astype(np.int64)

    def __len__(self):
        if DEBUG_MODE:
            return min(DEBUG_SAMPLES, len(self.input_data))
        return len(self.input_data)

    def __getitem__(self, idx):
        if DEBUG_MODE:
            idx = idx % min(DEBUG_SAMPLES, len(self.input_data))
        input_tensor = torch.tensor(self.input_data[idx], dtype=torch.float32)
        cluster_summary_labels = torch.tensor(self.cluster_summary_labels[idx], dtype=torch.float32)
        active_flags = torch.tensor(self.active_flags[idx], dtype=torch.float32)
        k_value = torch.tensor(self.k_values[idx], dtype=torch.long)  # k value for this event
        return input_tensor, cluster_summary_labels, active_flags, k_value

# ===========================
# Dynamic datasets & loader
# ===========================
class DynamicSummaryDataset(Dataset):
    """Dataset wrapper for a single total_points configuration without padding.

    Expects NPZ with keys (same generator as used in offset_version_final dynamic mode):
      - X_all_mod or X_all: [N, T, 8]
      - y_all: [N, Kmax, 9] (first 3 are centers, next 6 are upper-tri cov)
      - active_flags: [N, T]
      - k_all: [N]
    """
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        if 'X_all_mod' in data:
            self.X_all = data['X_all_mod'].astype(np.float32)
        elif 'X_all' in data:
            self.X_all = data['X_all'].astype(np.float32)
        else:
            raise KeyError('Neither X_all_mod nor X_all found in dataset.')

        if 'y_all' not in data:
            raise KeyError('y_all not found in dataset for set transformer dynamic mode.')
        self.cluster_summary_labels = data['y_all'].astype(np.float32)
        self.active_flags = data['active_flags'].astype(np.float32)
        self.k_values = data['k_all'].astype(np.int64)

        # Convenience attributes used by the multi-config loader
        self.num_events = self.X_all.shape[0]
        self.num_nodes = self.X_all.shape[1]
        self.total_points = self.num_nodes

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        x = torch.tensor(self.X_all[idx], dtype=torch.float32)
        y = torch.tensor(self.cluster_summary_labels[idx], dtype=torch.float32)
        flags = torch.tensor(self.active_flags[idx], dtype=torch.float32)
        kval = torch.tensor(self.k_values[idx], dtype=torch.long)
        return x, y, flags, kval

class MultiConfigDataLoader:
    """Iterates multiple DynamicSummaryDataset objects without padding.
    
    Each __next__ returns a dict: { total_points: (inputs, cluster_summary_labels, flags, k) }
    where every batch inside the dict has a uniform number of nodes.
    """
    def __init__(self, datasets, batch_size, shuffle=True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.loaders = {}
        for dataset in datasets:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            self.loaders[dataset.total_points] = loader

        # total iterations = max number of batches across configs
        self.total_iterations = 0
        for dataset in datasets:
            ds_len = len(dataset)
            num_batches = ds_len // batch_size
            if ds_len % batch_size != 0:
                num_batches += 1
            self.total_iterations = max(self.total_iterations, num_batches)

        self.iterators = {}
        self.current_iteration = 0
        self.reset_iterators()
        print(f"MultiConfigDataLoader: {len(datasets)} configs, {self.total_iterations} total iterations")

    def reset_iterators(self):
        for total_points, loader in self.loaders.items():
            self.iterators[total_points] = iter(loader)
        self.current_iteration = 0

    def __iter__(self):
        self.reset_iterators()
        return self

    def __next__(self):
        if self.current_iteration >= self.total_iterations:
            raise StopIteration
        batches = {}
        for total_points, iterator in self.iterators.items():
            try:
                batch = next(iterator)
                batches[total_points] = batch
            except StopIteration:
                continue
        self.current_iteration += 1
        if not batches:
            raise StopIteration
        return batches

def load_dynamic_datasets(data_dir: str, split: str, total_points_list: list):
    datasets = []
    split_dir = os.path.join(data_dir, split)
    for total_points in total_points_list:
        npz_path = os.path.join(split_dir, f'synthetic_detector_data_{total_points}pts.npz')
        if os.path.exists(npz_path):
            ds = DynamicSummaryDataset(npz_path)
            datasets.append(ds)
            print(f"Loaded {split} dataset {total_points}pts: {len(ds)} events, {ds.num_nodes} nodes")
        else:
            print(f"Warning: missing {npz_path}, skipping")
    return datasets

# ===========================
# ROOT Dataset
# ===========================
class RootWaferDataset(Dataset):
    """Dataset that reads HGCal wafer features and MergedSimCluster labels from ROOT files.

    Per-wafer feature vector (8 dims): [x, y, z, energy, mipPt, eta, phi, layer]
    Label vector per event (max_k, 9): [cx, cy, cz, 0,0,0,0,0,0]  (covariance zeros)
    active_flags (max_wafers,): 1 for real wafers, 0 for padding
    k_values: number of trainable MergedSimClusters (capped at max_k)

    Only events with at least min_trainable trainable clusters are kept.
    """

    _WAFER_BRANCHES = [
        'L1THGCAL_wafer_x', 'L1THGCAL_wafer_y', 'L1THGCAL_wafer_z',
        'L1THGCAL_wafer_energy', 'L1THGCAL_wafer_mipPt',
        'L1THGCAL_wafer_eta', 'L1THGCAL_wafer_phi', 'L1THGCAL_wafer_layer',
    ]
    _CLUSTER_BRANCHES = [
        'MergedSimCluster_impactPoint_x', 'MergedSimCluster_impactPoint_y',
        'MergedSimCluster_impactPoint_z', 'MergedSimCluster_isTrainable',
    ]

    def __init__(self, root_files, max_wafers: int = 600, max_k: int = 10, min_trainable: int = 1):
        try:
            import uproot
            import awkward as ak
        except ImportError as e:
            raise ImportError(f"uproot and awkward are required for RootWaferDataset: {e}")

        if isinstance(root_files, str):
            root_files = [root_files]

        all_X, all_y, all_flags, all_k = [], [], [], []

        for root_path in root_files:
            print(f"Loading {root_path} ...")
            with uproot.open(root_path) as f:
                tree = f['Events']
                data = tree.arrays(self._WAFER_BRANCHES + self._CLUSTER_BRANCHES, library='ak')
                n_events = tree.num_entries

            for ev in range(n_events):
                is_trainable = ak.to_numpy(data['MergedSimCluster_isTrainable'][ev]).astype(bool)
                k = int(is_trainable.sum())
                if k < min_trainable:
                    continue
                k = min(k, max_k)

                cx = ak.to_numpy(data['MergedSimCluster_impactPoint_x'][ev]).astype(np.float32)[is_trainable][:k]
                cy = ak.to_numpy(data['MergedSimCluster_impactPoint_y'][ev]).astype(np.float32)[is_trainable][:k]
                cz = ak.to_numpy(data['MergedSimCluster_impactPoint_z'][ev]).astype(np.float32)[is_trainable][:k]

                wx = ak.to_numpy(data['L1THGCAL_wafer_x'][ev]).astype(np.float32)
                wy = ak.to_numpy(data['L1THGCAL_wafer_y'][ev]).astype(np.float32)
                wz = ak.to_numpy(data['L1THGCAL_wafer_z'][ev]).astype(np.float32)
                we = ak.to_numpy(data['L1THGCAL_wafer_energy'][ev]).astype(np.float32)
                wm = ak.to_numpy(data['L1THGCAL_wafer_mipPt'][ev]).astype(np.float32)
                weta = ak.to_numpy(data['L1THGCAL_wafer_eta'][ev]).astype(np.float32)
                wphi = ak.to_numpy(data['L1THGCAL_wafer_phi'][ev]).astype(np.float32)
                wl = ak.to_numpy(data['L1THGCAL_wafer_layer'][ev]).astype(np.float32)

                feats = np.stack([we, wm, weta, wphi, wl, wx, wy, wz], axis=1)  # (N, 8): non-spatial first, xyz last
                T = min(len(wx), max_wafers)

                X_pad = np.zeros((max_wafers, 8), dtype=np.float32)
                X_pad[:T] = feats[:T]
                flags = np.zeros(max_wafers, dtype=np.float32)
                flags[:T] = 1.0

                y = np.zeros((max_k, 9), dtype=np.float32)
                y[:k, 0] = cx
                y[:k, 1] = cy
                y[:k, 2] = cz

                all_X.append(X_pad)
                all_y.append(y)
                all_flags.append(flags)
                all_k.append(k)

        if not all_X:
            raise RuntimeError("RootWaferDataset: no valid events found (check min_trainable and file paths)")

        self.input_data = np.stack(all_X, axis=0)
        self.cluster_summary_labels = np.stack(all_y, axis=0)
        self.active_flags = np.stack(all_flags, axis=0)
        self.k_values = np.array(all_k, dtype=np.int64)
        print(f"RootWaferDataset: {len(self.input_data)} events loaded from {len(root_files)} file(s)")

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_data[idx], dtype=torch.float32),
            torch.tensor(self.cluster_summary_labels[idx], dtype=torch.float32),
            torch.tensor(self.active_flags[idx], dtype=torch.float32),
            torch.tensor(self.k_values[idx], dtype=torch.long),
        )


def _print_k_distribution(k_all: np.ndarray):
    if k_all is None:
        print('k_all not found in dataset. Skipping k distribution summary.')
        return
    unique, counts = np.unique(k_all, return_counts=True)
    mapping = dict(zip(unique.tolist(), counts.tolist()))
    print('K distribution:')
    for k in sorted(mapping.keys()):
        print(f'  k={k}: {mapping[k]} events')

# ===========================
# K Prediction Loss Function
# ===========================
def _compute_k_prediction_loss(k_logits, k_gt, max_k=10):
    """Compute cross-entropy loss for k-prediction.
    
    k_logits: [B, max_k] - model predictions for k=1 to max_k
    k_gt: [B] - ground truth k values (1-indexed, so subtract 1 for 0-indexed)
    max_k: maximum k value to predict
    """
    # Convert k_gt to 0-indexed (since k=1 should correspond to index 0)
    k_gt_0indexed = k_gt - 1
    
    # Ensure k_gt is within valid range [0, max_k-1]
    k_gt_0indexed = torch.clamp(k_gt_0indexed, 0, max_k - 1)
    
    # Cross-entropy loss
    k_loss = F.cross_entropy(k_logits, k_gt_0indexed, reduction='mean')
    return k_loss

# ===========================
# Training Function
# ===========================
def _stratified_split_indices_by_k(k_values: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42):
    """Create stratified train/val/test indices so each split contains all ks when possible.

    Returns three lists of indices: train_indices, val_indices, test_indices.
    """
    rng = np.random.RandomState(seed)
    unique_k = np.unique(k_values)
    train_idx, val_idx, test_idx = [], [], []

    for k in unique_k:
        idx_k = np.where(k_values == k)[0]
        rng.shuffle(idx_k)
        n = len(idx_k)

        # Base allocations (floors)
        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        n_test = n - n_train - n_val

        # Ensure each split has at least 1 sample when feasible
        if n >= 3:
            if n_train == 0:
                n_train = 1
            if n_val == 0:
                n_val = 1
            n_test = max(1, n - n_train - n_val)
            # If adjustments caused over-allocation, reduce train first
            while n_train + n_val + n_test > n:
                if n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    n_test -= 1
        elif n == 2:
            # Split as 1 train, 1 test
            n_train, n_val, n_test = 1, 0, 1
        else:  # n == 1
            # Put single sample into train
            n_train, n_val, n_test = 1, 0, 0

        train_idx.extend(idx_k[:n_train].tolist())
        val_idx.extend(idx_k[n_train:n_train + n_val].tolist())
        test_idx.extend(idx_k[n_train + n_val:].tolist())

    # Shuffle combined indices to avoid ordered by k
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx

def train_model(
    model,
    dataset,
    device='cuda:0',
    n_epochs=8,
    batch_size=4,
    resume_from=None,
    final_test_size: int = 15000,
    fixed_lr: bool = False,
    base_lr: float = 1e-4,
    max_lr: float = 1e-3,
):
    device = torch.device(device)
    model = model.to(device)
    
    # Create checkpoint directory
    import os
    os.makedirs('checkpoint', exist_ok=True)

    # Stratified split by k so each split has all k values when possible
    k_values_np = np.array(dataset.k_values)
    train_indices, val_indices, test_indices = _stratified_split_indices_by_k(k_values_np, 0.7, 0.15, seed=42)
    
    # Apply debug mode to indices if needed
    if DEBUG_MODE:
        print(f"🐛 DEBUG MODE: limiting to {DEBUG_SAMPLES} events")
        # Take first DEBUG_SAMPLES from each split
        train_indices = train_indices[:DEBUG_SAMPLES//3]
        val_indices = val_indices[:DEBUG_SAMPLES//3] 
        test_indices = test_indices[:DEBUG_SAMPLES//3]
    
    # Optionally carve out a stratified final hold-out from the training indices
    final_test_set = None
    final_val_set = None
    if (not DEBUG_MODE) and final_test_size and final_test_size > 0:
        rng = np.random.RandomState(42)
        k_vals_train = np.array(dataset.k_values)[train_indices]
        unique_k_vals = np.unique(k_vals_train)
        target_size = min(final_test_size, len(train_indices) // 2)
        if target_size < unique_k_vals.size:
            print(f"Warning: requested final_test_size={final_test_size} but there are {unique_k_vals.size} unique k values in train; cannot guarantee coverage of every k.")

        # Build mapping: k -> shuffled list of absolute indices taken from train_indices
        k_to_indices = {}
        train_indices_arr = np.array(train_indices)
        for k in unique_k_vals:
            mask = (k_vals_train == k)
            idx_for_k = train_indices_arr[mask]
            idx_for_k = idx_for_k.copy()
            rng.shuffle(idx_for_k)
            k_to_indices[int(k)] = list(idx_for_k)

        # Round-robin selection across k values to fill up to target_size
        selected_final = []
        k_list = sorted(k_to_indices.keys())
        while len(selected_final) < target_size and any(len(k_to_indices[k]) > 0 for k in k_list):
            for k in k_list:
                if len(selected_final) >= target_size:
                    break
                if k_to_indices[k]:
                    selected_final.append(k_to_indices[k].pop())

        selected_final = np.array(selected_final, dtype=int)
        selected_final_set = set(selected_final.tolist())
        remaining_train_indices = [idx for idx in train_indices if idx not in selected_final_set]

        # Split final hold-out into validation-half and test-half
        split_point = len(selected_final) // 2
        final_val_indices = selected_final[:split_point]
        final_test_indices = selected_final[split_point:]

        train_set = Subset(dataset, remaining_train_indices)
        final_val_set = Subset(dataset, final_val_indices.tolist())
        final_test_set = Subset(dataset, final_test_indices.tolist())
    else:
        train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    # Print k distribution for each split
    def print_k_distribution(split_name, indices):
        k_vals = np.array(dataset.k_values)[np.asarray(indices, dtype=np.intp)]
        unique_k, counts = np.unique(k_vals, return_counts=True)
        print(f"{split_name} split k distribution:")
        for k, count in zip(unique_k, counts):
            print(f"  k={k}: {count} events")
        print(f"  Total: {len(indices)} events")
        print()

    print_k_distribution("Train", train_indices)
    print_k_distribution("Validation", val_indices)
    print_k_distribution("Test", test_indices)
    # Print distributions for final hold-out and updated train if applicable
    if final_test_set is not None:
        print("Final hold-out (from train) k distribution:")
        print_k_distribution("FinalHoldOut", np.array(final_test_set.indices))
        print("Train-set k distribution (after final hold-out):")
        print_k_distribution("Train(Updated)", np.array(train_set.indices))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # Merge final-val half into validation and keep a dedicated loader for it
    if final_val_set is not None:
        val_combined = ConcatDataset([val_set, final_val_set])
        val_loader = DataLoader(val_combined, batch_size=batch_size, shuffle=False)
        final_val_loader = DataLoader(final_val_set, batch_size=batch_size, shuffle=False)
    else:
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        final_val_loader = None
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    final_test_loader = None
    if final_test_set is not None:
        final_test_loader = DataLoader(final_test_set, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = None
    if not fixed_lr:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=len(train_loader),
            epochs=n_epochs,
            pct_start=0.15,
            anneal_strategy='cos'
        )
    criterion = nn.SmoothL1Loss()
    logs = []
    
    # Resume from checkpoint if specified
    start_epoch = 11
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logs = checkpoint.get('logs', [])
        print(f"Resuming from epoch {start_epoch}")
    
    # NEW: Track interval losses and event counts for plotting
    interval_event_counts = []
    interval_train_mae_losses = []
    interval_val_mae_losses = []
    interval_train_cos_losses = []
    interval_val_cos_losses = []
    # NEW: covariance interval tracking
    interval_train_cov_mae_losses = []
    interval_val_cov_mae_losses = []
    interval_train_cov_cos_losses = []
    interval_val_cov_cos_losses = []
    total_events_processed = 0

    for epoch in range(start_epoch, n_epochs):
        model.train()

        total_loss = total_mae = total_cos = total_k_acc = 0.0
        total_cov_mae = 0.0
        total_cov_cos = 0.0
        total_act_bce = 0.0
        total_act_acc = 0.0
        total_act_bce = 0.0
        total_act_acc = 0.0
        
        # Add event counter and interval loss for printing every 300 events
        event_counter = 0
        interval_loss = 0.0
        interval_mae_loss = 0.0  # Track MAE loss for intervals
        interval_cos_loss = 0.0  # Track cosine similarity for intervals
        interval_cov_mae_loss = 0.0  # Track covariance MAE for intervals
        interval_cov_cos_loss = 0.0  # Track covariance cosine similarity for intervals
        interval_act_bce_loss = 0.0
        interval_act_acc = 0.0
        interval_act_bce_loss = 0.0  # Track activity BCE for intervals
        interval_act_acc = 0.0  # Track activity accuracy for intervals
        interval_events = 0

        for inputs, cluster_summary_labels, active_flags, k_value in train_loader:
            inputs = inputs.to(device)
            cluster_summary_labels = cluster_summary_labels.to(device)
            active_flags = active_flags.to(device)
            k_value = k_value.to(device)
            
            optimizer.zero_grad()

            preds, cov_preds, k_logits, activity_logits = model(inputs)  # (B,Kmax,3), (B,Kmax,6), (B,max_k), (B,T,1)

            # K-prediction loss
            k_loss = _compute_k_prediction_loss(k_logits, k_value, model.k)
            
            # Compute k-prediction accuracy
            k_pred = k_logits.argmax(dim=1) + 1  # Convert back to 1-indexed
            k_acc = (k_pred == k_value).float().mean().item()

            # Compute loss and metrics per-sample using that sample's k
            per_sample_losses = []
            per_sample_mae = []
            per_sample_cos = []
            per_sample_cov_losses = []
            per_sample_cov_mae = []
            per_sample_cov_cos = []

            batch_size_curr = preds.size(0)
            for b in range(batch_size_curr):
                ki = int(k_value[b].item())
                targets_b = cluster_summary_labels[b, :ki, :3]  # (ki,3)
                preds_b = preds[b]  # (Kmax,3)
                cov_targets_b = cluster_summary_labels[b, :ki, 3:9]  # (ki,6)
                cov_preds_b = cov_preds[b]  # (Kmax,6)

                # Hungarian assignment on rectangular cost (Kmax x ki)
                cost = torch.cdist(preds_b.detach(), targets_b.detach(), p=1).cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost)

                matched_preds = preds_b[row_ind]
                matched_targets = targets_b[col_ind]
                matched_cov_preds = cov_preds_b[row_ind]
                matched_cov_targets = cov_targets_b[col_ind]

                per_sample_losses.append(criterion(matched_preds, matched_targets))
                per_sample_mae.append(F.l1_loss(matched_preds, matched_targets))
                per_sample_cov_losses.append(criterion(matched_cov_preds, matched_cov_targets))
                per_sample_cov_mae.append(F.l1_loss(matched_cov_preds, matched_cov_targets))
                # Cosine similarity on flattened vectors
                cos_b = F.cosine_similarity(matched_preds.reshape(1, -1), matched_targets.reshape(1, -1), dim=1).mean()
                per_sample_cos.append(cos_b)
                cov_cos_b = F.cosine_similarity(matched_cov_preds.reshape(1, -1), matched_cov_targets.reshape(1, -1), dim=1).mean()
                per_sample_cov_cos.append(cov_cos_b)

            # Aggregate across samples in batch
            reg_loss = torch.stack(per_sample_losses).mean()
            cov_loss = torch.stack(per_sample_cov_losses).mean()
            
            # Activity BCE loss (node-level)
            activity_bce = F.binary_cross_entropy_with_logits(
                activity_logits.squeeze(-1), active_flags
            )

            # Combine losses with k prediction loss and activity loss
            k_loss_weight = 0.01  # Weight for k prediction loss
            cov_loss_weight = 1.0  # Weight for covariance regression
            act_loss_weight = 0.2  # Initial weight for activity supervision
            loss = reg_loss + cov_loss_weight * cov_loss + k_loss_weight * k_loss + act_loss_weight * activity_bce

            # Backpropagation
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Metrics logging (use means across batch)
            total_loss += loss.item()
            batch_mae = torch.stack(per_sample_mae).mean().item()
            batch_cov_mae = torch.stack(per_sample_cov_mae).mean().item()
            total_mae += batch_mae
            total_cov_mae += batch_cov_mae
            cos_sim = torch.stack(per_sample_cos).mean().item()
            total_cos += cos_sim
            total_k_acc += k_acc
            # Activity interval/global aggregates
            total_act_bce += activity_bce.item()
            act_pred = (torch.sigmoid(activity_logits).squeeze(-1) >= 0.5).float()
            batch_act_acc = (act_pred == active_flags).float().mean().item()
            total_act_acc += batch_act_acc
            # Activity interval/global aggregates
            total_act_bce += activity_bce.item()
            act_pred = (torch.sigmoid(activity_logits).squeeze(-1) >= 0.5).float()
            batch_act_acc = (act_pred == active_flags).float().mean().item()
            total_act_acc += batch_act_acc
            
            # Track interval losses
            interval_loss += loss.item() * inputs.size(0)
            interval_mae_loss += batch_mae * inputs.size(0)
            interval_cos_loss += cos_sim * inputs.size(0)
            # covariance intervals
            batch_cov_cos = torch.stack(per_sample_cov_cos).mean().item()
            interval_cov_mae_loss += batch_cov_mae * inputs.size(0)
            interval_cov_cos_loss += batch_cov_cos * inputs.size(0)
            interval_act_bce_loss += activity_bce.item() * inputs.size(0)
            interval_act_acc += batch_act_acc * inputs.size(0)
            interval_act_bce_loss += activity_bce.item() * inputs.size(0)
            interval_act_acc += batch_act_acc * inputs.size(0)
            event_counter += inputs.size(0)
            interval_events += inputs.size(0)
            total_events_processed += inputs.size(0)
            total_cov_cos += batch_cov_cos

            # Print every 300 events
            if event_counter >= 300:
                avg_interval_loss = interval_loss / interval_events
                avg_interval_mae_loss = interval_mae_loss / interval_events
                avg_interval_cos_loss = interval_cos_loss / interval_events
                
                # Compute validation metrics for 300 events
                val_mae_sum = 0.0
                val_cos_sum = 0.0
                val_cov_mae_sum = 0.0
                val_cov_cos_sum = 0.0
                val_events_processed = 0
                val_events_needed = 300
                
                # Use a fresh validation loader
                val_loader_iter = iter(val_loader)
                while val_events_processed < val_events_needed:
                    try:
                        val_inputs, val_cluster_summary_labels, val_active_flags, val_k_value = next(val_loader_iter)
                    except StopIteration:
                        # If we run out of validation data, restart
                        val_loader_iter = iter(val_loader)
                        val_inputs, val_cluster_summary_labels, val_active_flags, val_k_value = next(val_loader_iter)
                    
                    val_inputs = val_inputs.to(device)
                    val_cluster_summary_labels = val_cluster_summary_labels.to(device)
                    val_active_flags = val_active_flags.to(device)
                    val_k_value = val_k_value.to(device)

                    with torch.no_grad():
                        val_preds, val_cov_preds, val_k_logits, val_act_logits = model(val_inputs)  # (B,Kmax,3), (B,Kmax,6), (B,max_k)

                        # Compute per-sample metrics using that sample's k
                        per_val_mae = []
                        per_val_cos = []
                        per_val_cov_mae = []
                        per_val_cov_cos = []
                        for vb in range(val_preds.size(0)):
                            ki_v = int(val_k_value[vb].item())
                            val_targets_b = val_cluster_summary_labels[vb, :ki_v, :3]
                            val_cov_targets_b = val_cluster_summary_labels[vb, :ki_v, 3:9]
                            val_preds_b = val_preds[vb]
                            val_cov_preds_b = val_cov_preds[vb]
                            cost = torch.cdist(val_preds_b.detach(), val_targets_b.detach(), p=1).cpu().numpy()
                            row_ind, col_ind = linear_sum_assignment(cost)
                            matched_preds = val_preds_b[row_ind]
                            matched_targets = val_targets_b[col_ind]
                            matched_cov_preds = val_cov_preds_b[row_ind]
                            matched_cov_targets = val_cov_targets_b[col_ind]
                            per_val_mae.append(F.l1_loss(matched_preds, matched_targets))
                            cos_b = F.cosine_similarity(matched_preds.reshape(1, -1), matched_targets.reshape(1, -1), dim=1).mean()
                            per_val_cos.append(cos_b)
                            per_val_cov_mae.append(F.l1_loss(matched_cov_preds, matched_cov_targets))
                            cov_cos_b = F.cosine_similarity(matched_cov_preds.reshape(1, -1), matched_cov_targets.reshape(1, -1), dim=1).mean()
                            per_val_cov_cos.append(cov_cos_b)

                        batch_val_mae = torch.stack(per_val_mae).mean().item()
                        batch_val_cos = torch.stack(per_val_cos).mean().item()
                        batch_val_cov_mae = torch.stack(per_val_cov_mae).mean().item()
                        batch_val_cov_cos = torch.stack(per_val_cov_cos).mean().item()

                    # Activity metrics (per batch)
                    val_act_bce = F.binary_cross_entropy_with_logits(val_act_logits.squeeze(-1), val_active_flags)
                    val_act_pred = (torch.sigmoid(val_act_logits).squeeze(-1) >= 0.5).float()
                    val_act_acc = (val_act_pred == val_active_flags).float().mean().item()

                    batch_size = val_inputs.size(0)
                    take = min(batch_size, val_events_needed - val_events_processed)
                    val_mae_sum += batch_val_mae * take
                    val_cos_sum += batch_val_cos * take
                    val_cov_mae_sum += batch_val_cov_mae * take
                    val_cov_cos_sum += batch_val_cov_cos * take
                    # Activity sums for interval val
                    # Weight by number taken to keep same scale
                    val_act_bce_sum = val_act_bce_sum + (val_act_bce.item() * take) if 'val_act_bce_sum' in locals() else (val_act_bce.item() * take)
                    val_act_acc_sum = val_act_acc_sum + (val_act_acc * take) if 'val_act_acc_sum' in locals() else (val_act_acc * take)
                    val_events_processed += take
                
                avg_val_mae_loss = val_mae_sum / val_events_needed
                avg_val_cos_loss = val_cos_sum / val_events_needed
                avg_val_cov_mae_loss = val_cov_mae_sum / val_events_needed
                avg_val_cov_cos_loss = val_cov_cos_sum / val_events_needed
                avg_val_act_bce = (val_act_bce_sum / val_events_needed) if 'val_act_bce_sum' in locals() else 0.0
                avg_val_act_acc = (val_act_acc_sum / val_events_needed) if 'val_act_acc_sum' in locals() else 0.0
                
                print(f"[Epoch {epoch:03d}] Processed {total_events_processed} events | Train Avg Loss (last 300 events): {avg_interval_loss:.6f} | Train Avg MAE: {avg_interval_mae_loss:.6f} | Val Avg MAE: {avg_val_mae_loss:.6f} | Train Avg CosSim: {avg_interval_cos_loss:.4f} | Val Avg CosSim: {avg_val_cos_loss:.4f} | Train Avg CovMAE: { (interval_cov_mae_loss/interval_events) if interval_events>0 else 0.0 :.6f} | Val Avg CovMAE: {avg_val_cov_mae_loss:.6f} | Train Avg CovCosSim: { (interval_cov_cos_loss/interval_events) if interval_events>0 else 0.0 :.4f} | Val Avg CovCosSim: {avg_val_cov_cos_loss:.4f} | Train ActBCE: { (interval_act_bce_loss/interval_events) if interval_events>0 else 0.0 :.6f} | Val ActBCE: {avg_val_act_bce:.6f} | Train ActAcc: { (interval_act_acc/interval_events) if interval_events>0 else 0.0 :.4f} | Val ActAcc: {avg_val_act_acc:.4f}")
                
                # Store interval data for plotting
                interval_event_counts.append(total_events_processed)
                interval_train_mae_losses.append(avg_interval_mae_loss)
                interval_val_mae_losses.append(avg_val_mae_loss)
                interval_train_cos_losses.append(avg_interval_cos_loss)
                interval_val_cos_losses.append(avg_val_cos_loss)
                interval_train_cov_mae_losses.append(interval_cov_mae_loss / interval_events)
                interval_val_cov_mae_losses.append(avg_val_cov_mae_loss)
                interval_train_cov_cos_losses.append(interval_cov_cos_loss / interval_events)
                interval_val_cov_cos_losses.append(avg_val_cov_cos_loss)
                
                # Reset interval counters
                interval_loss = 0.0
                interval_mae_loss = 0.0
                interval_cos_loss = 0.0
                interval_cov_mae_loss = 0.0
                interval_cov_cos_loss = 0.0
                interval_act_bce_loss = 0.0
                interval_act_acc = 0.0
                interval_events = 0
                event_counter = 0

        # If there are leftover events in the last interval, average and store them too
        if interval_events > 0:
            avg_interval_loss = interval_loss / interval_events
            avg_interval_mae_loss = interval_mae_loss / interval_events
            avg_interval_cos_loss = interval_cos_loss / interval_events
            
            # Store interval data for plotting
            interval_event_counts.append(total_events_processed)
            interval_train_mae_losses.append(avg_interval_mae_loss)
            interval_val_mae_losses.append(avg_interval_mae_loss)  # For consistency, use train values for leftover
            interval_train_cos_losses.append(avg_interval_cos_loss)
            interval_val_cos_losses.append(avg_interval_cos_loss)  # For consistency, use train values for leftover
        
        steps = len(train_loader)
        train_metrics = {
            'epoch': epoch,
            'loss': total_loss / steps,
            'mae': total_mae / steps,
            'cov_mae': total_cov_mae / steps,
            'cos_sim': total_cos / steps,
            'cov_cos': total_cov_cos / steps,
            'k_accuracy': total_k_acc / steps,
            'act_bce': total_act_bce / steps,
            'act_acc': total_act_acc / steps
        }

        val_metrics = evaluate_model(model, val_loader, device, criterion)
        breakout_val_metrics = None
        if final_val_loader is not None:
            breakout_val_metrics = evaluate_model(model, final_val_loader, device, criterion)

        print(f"[Epoch {epoch:03d}] Loss: {train_metrics['loss']:.6f} | MAE: {train_metrics['mae']:.6f} | CovMAE: {train_metrics['cov_mae']:.6f} | CosSim: {train_metrics['cos_sim']:.4f} | CovCosSim: {train_metrics['cov_cos']:.4f} | K Acc: {train_metrics['k_accuracy']:.4f} | ActBCE: {train_metrics.get('act_bce',0.0):.6f} | ActAcc: {train_metrics.get('act_acc',0.0):.4f} || "
              f"Val Loss: {val_metrics['loss']:.6f} | Val MAE: {val_metrics['mae']:.6f} | Val CovMAE: {val_metrics.get('cov_mae', 0.0):.6f} | Val CosSim: {val_metrics['cos_sim']:.4f} | Val CovCosSim: {val_metrics.get('cov_cos', 0.0):.4f} | Val K Acc: {val_metrics['k_accuracy']:.4f} | Val ActBCE: {val_metrics.get('act_bce',0.0):.6f} | Val ActAcc: {val_metrics.get('act_acc',0.0):.4f} || ")
        if breakout_val_metrics is not None:
            print(f"[Epoch {epoch:03d}] Val (break-out) | Loss: {breakout_val_metrics['loss']:.6f} | MAE: {breakout_val_metrics['mae']:.6f} | CovMAE: {breakout_val_metrics.get('cov_mae', 0.0):.6f} | CosSim: {breakout_val_metrics['cos_sim']:.4f} | CovCosSim: {breakout_val_metrics.get('cov_cos', 0.0):.4f} | K Acc: {breakout_val_metrics['k_accuracy']:.4f}")

        logs.append({**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}})
        df = pd.DataFrame(logs)
        df.set_index('epoch', inplace=True)
        df.plot(figsize=(10, 6), title="Training and Validation Metrics")
        plt.grid(True)
        plt.savefig("training_metrics_with_val.png")
        
        # Save model weights after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'logs': logs
        }, f'checkpoint/model_epoch_{epoch:03d}.pth')
        
        # Also save the latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'logs': logs
        }, 'checkpoint/model_latest.pth')

    # Final test evaluation
    test_metrics = evaluate_model(model, test_loader, device, criterion)
    print("\nFinal Test Performance:")
    print(f"Test Loss: {test_metrics['loss']:.6f} | MAE: {test_metrics['mae']:.6f} | CovMAE: {test_metrics.get('cov_mae', 0.0):.6f} | CosSim: {test_metrics['cos_sim']:.4f} | CovCosSim: {test_metrics.get('cov_cos', 0.0):.4f} | K Acc: {test_metrics['k_accuracy']:.4f} || ")

    torch.save(model.state_dict(), 'nano_encoder_model.pth')

    # Optional: evaluate on the final hold-out split carved from train (TEST half)
    if final_test_loader is not None:
        final_hold_metrics = evaluate_model(model, final_test_loader, device, criterion)
        print("\nFinal Test (hold-out from train) | "
              f"Loss: {final_hold_metrics['loss']:.6f} | MAE: {final_hold_metrics['mae']:.6f} | CovMAE: {final_hold_metrics.get('cov_mae', 0.0):.6f} | "
              f"Cos: {final_hold_metrics['cos_sim']:.4f} | CovCos: {final_hold_metrics.get('cov_cos', 0.0):.4f} | "
              f"K Acc: {final_hold_metrics['k_accuracy']:.4f}")
    
    # =========================
    # Plot synchronized training and val MAE loss (interval-based)
    # =========================
    plt.figure(figsize=(12,7))
    plt.plot(interval_event_counts, interval_train_mae_losses, label='Train MAE (per 300 events)', marker='o')
    plt.plot(interval_event_counts, interval_val_mae_losses, label='Val MAE (per 300 events)', marker='x')
    plt.xlabel('Number of Events Processed')
    plt.ylabel('Loss (MAE)')
    plt.title('Training and Validation MAE Curves (per 300 events)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('train_val_mae_loss_curve_intervals.png')
    plt.close()
    
    # =========================
    # Plot synchronized training and val cosine similarity (interval-based)
    # =========================
    plt.figure(figsize=(12,7))
    plt.plot(interval_event_counts, interval_train_cos_losses, label='Train CosSim (per 300 events)', marker='o')
    plt.plot(interval_event_counts, interval_val_cos_losses, label='Val CosSim (per 300 events)', marker='x')
    plt.xlabel('Number of Events Processed')
    plt.ylabel('Cosine Similarity')
    plt.title('Training and Validation Cosine Similarity Curves (per 300 events)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('train_val_cos_sim_curve_intervals.png')
    plt.close()

    # =========================
    # Plot synchronized training and val Covariance MAE (interval-based)
    # =========================
    plt.figure(figsize=(12,7))
    plt.plot(interval_event_counts, interval_train_cov_mae_losses, label='Train CovMAE (per 300 events)', marker='o')
    plt.plot(interval_event_counts, interval_val_cov_mae_losses, label='Val CovMAE (per 300 events)', marker='x')
    plt.xlabel('Number of Events Processed')
    plt.ylabel('Loss (Cov MAE)')
    plt.title('Training and Validation Covariance MAE Curves (per 300 events)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('train_val_cov_mae_loss_curve_intervals.png')
    plt.close()

    # =========================
    # Plot synchronized training and val Covariance cosine similarity (interval-based)
    # =========================
    plt.figure(figsize=(12,7))
    plt.plot(interval_event_counts, interval_train_cov_cos_losses, label='Train Cov CosSim (per 300 events)', marker='o')
    plt.plot(interval_event_counts, interval_val_cov_cos_losses, label='Val Cov CosSim (per 300 events)', marker='x')
    plt.xlabel('Number of Events Processed')
    plt.ylabel('Covariance Cosine Similarity')
    plt.title('Training and Validation Covariance Cosine Similarity Curves (per 300 events)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('train_val_cov_cos_sim_curve_intervals.png')
    plt.close()

    # ----------------------------------------------------------
    # Quick 3-D visualisation of predicted vs ground-truth centres
    # ----------------------------------------------------------

    def visualise_events_simple(model, dataset, device, num_events=10, out_dir="event_vis_pool"):
        """Save single plots showing GT vs predicted centres with matching lines.

        GT centres: red × markers
        Predicted centres: blue ○ markers  
        Matching lines: grey lines connecting matched pairs
        A grey point cloud of detector nodes provides spatial context.
        """
        os.makedirs(out_dir, exist_ok=True)

        model.eval()
        from scipy.optimize import linear_sum_assignment

        # Ensure we visualize events from all k values
        k_values = np.array(dataset.k_values)
        unique_k = np.unique(k_values)
        print(f"Visualizing events with k values: {unique_k}")
        
        # Sample events from each k value
        events_per_k = max(1, num_events // len(unique_k))
        idxs = []
        
        for k in unique_k:
            k_indices = np.where(k_values == k)[0]
            if len(k_indices) > 0:
                # Take evenly spaced samples from this k value
                k_samples = np.linspace(0, len(k_indices)-1, min(events_per_k, len(k_indices)), dtype=int)
                idxs.extend(k_indices[k_samples])
        
        # If we have room for more events, add some random ones
        if len(idxs) < num_events:
            remaining = num_events - len(idxs)
            all_indices = np.arange(len(dataset))
            available = np.setdiff1d(all_indices, idxs)
            if len(available) > 0:
                additional = np.random.choice(available, min(remaining, len(available)), replace=False)
                idxs.extend(additional)
        
        idxs = idxs[:num_events]  # Ensure we don't exceed num_events
        print(f"Selected event indices: {idxs}")
        print(f"Corresponding k values: {[int(dataset.k_values[i]) for i in idxs]}")

        with torch.no_grad():
            for idx in idxs:
                inputs, cluster_summary_labels, active_flags, k_value = dataset[idx]

                k = int(k_value.item())

                # predict
                inputs_device = inputs.unsqueeze(0).to(device)
                centres_pred_all, _, k_logits, _ = model(inputs_device)
                centres_pred_all = centres_pred_all.cpu().squeeze(0).numpy()  # (Kmax,3)
                centres_gt = cluster_summary_labels[:k, :3].numpy()  # (k,3)

                # Build rectangular cost matrix (Kmax x k)
                Kmax = centres_pred_all.shape[0]
                cost_matrix = np.zeros((Kmax, k), dtype=np.float32)
                for i in range(Kmax):
                    for j in range(k):
                        cost_matrix[i, j] = np.linalg.norm(centres_pred_all[i] - centres_gt[j])

                # Hungarian assignment returns row indices (pred) and col indices (gt)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                # Keep only the k assignments that map to the k GT centres, ordered by GT index
                order = np.argsort(col_ind)
                matched_rows = row_ind[order][:k]
                centres_pred_matched = centres_pred_all[matched_rows]

                detector_xyz = inputs[:, -3:].numpy()  # xyz columns

                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')

                # Plot detector nodes as background
                ax.scatter(detector_xyz[:, 0], detector_xyz[:, 1], detector_xyz[:, 2],
                           c='lightgrey', s=4, alpha=0.25, label='Detector nodes')

                # Plot GT centers
                ax.scatter(centres_gt[:, 0], centres_gt[:, 1], centres_gt[:, 2],
                           c='red', marker='x', s=100, linewidths=3, label='GT centers')

                # Plot predicted centers
                ax.scatter(centres_pred_matched[:, 0], centres_pred_matched[:, 1], centres_pred_matched[:, 2],
                           c='blue', marker='o', s=100, linewidths=2, label='Predicted centers')

                # Draw lines connecting matched pairs (GT index order)
                for i in range(k):
                    ax.plot([centres_gt[i, 0], centres_pred_matched[i, 0]],
                            [centres_gt[i, 1], centres_pred_matched[i, 1]],
                            [centres_gt[i, 2], centres_pred_matched[i, 2]],
                            'gray', alpha=0.6, linewidth=1)

                ax.set_title(f"Event {idx}: GT vs Predicted Centers (k={k})")
                ax.set_xlabel('X [mm]')
                ax.set_ylabel('Y [mm]')
                ax.set_zlabel('Z [mm]')
                ax.set_xlim(-120, 120)
                ax.set_ylim(-120, 120)
                ax.set_zlim(-100, 100)
                ax.view_init(elev=25, azim=135)
                ax.legend()

                plt.tight_layout()
                outfile = os.path.join(out_dir, f"event_{idx}.png")
                plt.savefig(outfile, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved visualisation to {outfile}")

    visualise_events_simple(model, dataset, device='cuda:0', num_events=10, out_dir="event_vis_pool")

# ===========================
# Evaluation Function
# ===========================
def evaluate_model(model, loader, device, criterion):
    model.eval()
    total_loss = total_mae = total_cos = total_k_acc = 0.0
    total_cov_mae = 0.0
    total_cov_cos = 0.0
    total_act_bce = 0.0
    total_act_acc = 0.0

    with torch.no_grad():
        for inputs, cluster_summary_labels, active_flags, k_value in loader:
            inputs = inputs.to(device)
            cluster_summary_labels = cluster_summary_labels.to(device)
            active_flags = active_flags.to(device)
            k_value = k_value.to(device)
            
            preds, cov_preds, k_logits, activity_logits = model(inputs)  # (B,Kmax,3), (B,Kmax,6), (B,max_k), (B,T,1)

            # Compute per-sample losses/metrics
            per_sample_losses = []
            per_sample_mae = []
            per_sample_cos = []
            per_sample_cov_losses = []
            per_sample_cov_mae = []

            for b in range(preds.size(0)):
                ki = int(k_value[b].item())
                targets_b = cluster_summary_labels[b, :ki, :3]
                cov_targets_b = cluster_summary_labels[b, :ki, 3:9]
                preds_b = preds[b]
                cov_preds_b = cov_preds[b]
                cost = torch.cdist(preds_b.detach(), targets_b.detach(), p=1).cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost)
                matched_preds = preds_b[row_ind]
                matched_targets = targets_b[col_ind]
                matched_cov_preds = cov_preds_b[row_ind]
                matched_cov_targets = cov_targets_b[col_ind]

                per_sample_losses.append(criterion(matched_preds, matched_targets))
                per_sample_mae.append(F.l1_loss(matched_preds, matched_targets))
                per_sample_cov_losses.append(criterion(matched_cov_preds, matched_cov_targets))
                per_sample_cov_mae.append(F.l1_loss(matched_cov_preds, matched_cov_targets))
                cos_b = F.cosine_similarity(matched_preds.reshape(1, -1), matched_targets.reshape(1, -1), dim=1).mean()
                per_sample_cos.append(cos_b)

            total_loss += torch.stack(per_sample_losses).mean().item()
            total_mae += torch.stack(per_sample_mae).mean().item()
            batch_cov_mae = torch.stack(per_sample_cov_mae).mean().item()
            total_cov_mae += batch_cov_mae
            total_cos += torch.stack(per_sample_cos).mean().item()
            # Covariance cosine similarity per batch
            per_sample_cov_cos = []
            for b in range(preds.size(0)):
                ki = int(k_value[b].item())
                preds_b = preds[b]
                cov_preds_b = cov_preds[b]
                targets_b = cluster_summary_labels[b, :ki, :3]
                cov_targets_b = cluster_summary_labels[b, :ki, 3:9]
                cost = torch.cdist(preds_b.detach(), targets_b.detach(), p=1).cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost)
                matched_cov_preds = cov_preds_b[row_ind]
                matched_cov_targets = cov_targets_b[col_ind]
                cov_cos_b = F.cosine_similarity(matched_cov_preds.reshape(1, -1), matched_cov_targets.reshape(1, -1), dim=1).mean()
                per_sample_cov_cos.append(cov_cos_b)
            total_cov_cos += torch.stack(per_sample_cov_cos).mean().item()
            
            # K-prediction accuracy
            k_pred = k_logits.argmax(dim=1) + 1  # Convert back to 1-indexed
            k_acc = (k_pred == k_value).float().mean().item()
            total_k_acc += k_acc

            # Activity metrics
            act_bce = F.binary_cross_entropy_with_logits(activity_logits.squeeze(-1), active_flags)
            total_act_bce += act_bce.item()
            act_pred = (torch.sigmoid(activity_logits).squeeze(-1) >= 0.5).float()
            act_acc = (act_pred == active_flags).float().mean().item()
            total_act_acc += act_acc

    steps = len(loader)
    return {
        'loss': total_loss / steps,
        'mae': total_mae / steps,
        'cov_mae': total_cov_mae / steps,
        'cos_sim': total_cos / steps,
        'cov_cos': total_cov_cos / steps,
        'k_accuracy': total_k_acc / steps,
        'act_bce': total_act_bce / steps,
        'act_acc': total_act_acc / steps,
    }

# ===========================
# Dynamic train/eval (no padding)
# ===========================
def evaluate_model_dynamic(model, loader, device):
    model.eval()
    total_mae = 0.0
    total_cov_mae = 0.0
    total_cos = 0.0
    total_k_acc = 0.0
    total_cov_cos = 0.0
    total_act_bce = 0.0
    total_act_acc = 0.0
    steps = 0

    with torch.no_grad():
        for batches in loader:
            for _, batch in batches.items():
                inputs, cluster_summary_labels, active_flags, k_value = batch
                inputs = inputs.to(device)
                cluster_summary_labels = cluster_summary_labels.to(device)
                active_flags = active_flags.to(device)
                k_value = k_value.to(device)

                preds, cov_preds, k_logits, activity_logits = model(inputs)

                per_sample_mae = []
                per_sample_cov_mae = []
                per_sample_cos = []

                for b in range(preds.size(0)):
                    ki = int(k_value[b].item())
                    targets_b = cluster_summary_labels[b, :ki, :3]
                    cov_targets_b = cluster_summary_labels[b, :ki, 3:9]
                    preds_b = preds[b]
                    cov_preds_b = cov_preds[b]

                    cost = torch.cdist(preds_b.detach(), targets_b.detach(), p=1).cpu().numpy()
                    row_ind, col_ind = linear_sum_assignment(cost)

                    matched_preds = preds_b[row_ind]
                    matched_targets = targets_b[col_ind]
                    matched_cov_preds = cov_preds_b[row_ind]
                    matched_cov_targets = cov_targets_b[col_ind]

                    per_sample_mae.append(F.l1_loss(matched_preds, matched_targets))
                    per_sample_cov_mae.append(F.l1_loss(matched_cov_preds, matched_cov_targets))
                    cos_b = F.cosine_similarity(matched_preds.reshape(1, -1), matched_targets.reshape(1, -1), dim=1).mean()
                    per_sample_cos.append(cos_b)

                batch_mae = torch.stack(per_sample_mae).mean().item()
                batch_cov_mae = torch.stack(per_sample_cov_mae).mean().item()
                batch_cos = torch.stack(per_sample_cos).mean().item()
                # covariance cosine per batch
                per_sample_cov_cos = []
                for b in range(preds.size(0)):
                    ki = int(k_value[b].item())
                    preds_b = preds[b]
                    cov_preds_b = cov_preds[b]
                    targets_b = cluster_summary_labels[b, :ki, :3]
                    cov_targets_b = cluster_summary_labels[b, :ki, 3:9]
                    cost = torch.cdist(preds_b.detach(), targets_b.detach(), p=1).cpu().numpy()
                    row_ind, col_ind = linear_sum_assignment(cost)
                    matched_cov_preds = cov_preds_b[row_ind]
                    matched_cov_targets = cov_targets_b[col_ind]
                    cov_cos_b = F.cosine_similarity(matched_cov_preds.reshape(1, -1), matched_cov_targets.reshape(1, -1), dim=1).mean()
                    per_sample_cov_cos.append(cov_cos_b)
                batch_cov_cos = torch.stack(per_sample_cov_cos).mean().item()

                total_mae += batch_mae
                total_cov_mae += batch_cov_mae
                total_cos += batch_cos
                total_cov_cos += batch_cov_cos

                k_pred = k_logits.argmax(dim=1) + 1
                k_acc = (k_pred == k_value).float().mean().item()
                total_k_acc += k_acc
                # Activity metrics
                act_bce = F.binary_cross_entropy_with_logits(activity_logits.squeeze(-1), active_flags)
                total_act_bce += act_bce.item()
                act_pred = (torch.sigmoid(activity_logits).squeeze(-1) >= 0.5).float()
                act_acc = (act_pred == active_flags).float().mean().item()
                total_act_acc += act_acc
                steps += 1

    if steps == 0:
        steps = 1
    return {
        'mae': total_mae / steps,
        'cov_mae': total_cov_mae / steps,
        'cos_sim': total_cos / steps,
        'cov_cos': total_cov_cos / steps,
        'k_accuracy': total_k_acc / steps,
        'act_bce': total_act_bce / steps,
        'act_acc': total_act_acc / steps,
    }

def print_val_metrics_per_config_dynamic(model, loader, device):
    """Print per-configuration validation metrics (center MAE, cov MAE, center cos, cov cos)."""
    model.eval()
    # Accumulate by total_points
    sums = {}  # tp -> dict of sums
    counts = {}  # tp -> int
    with torch.no_grad():
        for batches in loader:
            for total_points, batch in batches.items():
                inputs, centres_labels, active_flags, k_value = batch
                inputs = inputs.to(device)
                centres_labels = centres_labels.to(device)
                active_flags = active_flags.to(device)
                k_value = k_value.to(device)

                preds, cov_preds, k_logits, act_logits = model(inputs)

                per_sample_mae = []
                per_sample_cov_mae = []
                per_sample_cos = []
                per_sample_cov_cos = []

                for b in range(preds.size(0)):
                    ki = int(k_value[b].item())
                    targets_b = centres_labels[b, :ki, :3]
                    cov_targets_b = centres_labels[b, :ki, 3:9]
                    preds_b = preds[b]
                    cov_preds_b = cov_preds[b]

                    cost = torch.cdist(preds_b.detach(), targets_b.detach(), p=1).cpu().numpy()
                    row_ind, col_ind = linear_sum_assignment(cost)

                    matched_preds = preds_b[row_ind]
                    matched_targets = targets_b[col_ind]
                    matched_cov_preds = cov_preds_b[row_ind]
                    matched_cov_targets = cov_targets_b[col_ind]

                    per_sample_mae.append(F.l1_loss(matched_preds, matched_targets))
                    per_sample_cov_mae.append(F.l1_loss(matched_cov_preds, matched_cov_targets))
                    per_sample_cos.append(F.cosine_similarity(matched_preds.reshape(1, -1), matched_targets.reshape(1, -1), dim=1).mean())
                    per_sample_cov_cos.append(F.cosine_similarity(matched_cov_preds.reshape(1, -1), matched_cov_targets.reshape(1, -1), dim=1).mean())

                batch_mae = torch.stack(per_sample_mae).mean().item()
                batch_cov_mae = torch.stack(per_sample_cov_mae).mean().item()
                batch_cos = torch.stack(per_sample_cos).mean().item()
                batch_cov_cos = torch.stack(per_sample_cov_cos).mean().item()

                if total_points not in sums:
                    sums[total_points] = {'mae': 0.0, 'cov_mae': 0.0, 'cos': 0.0, 'cov_cos': 0.0, 'act_bce': 0.0, 'act_acc': 0.0}
                    counts[total_points] = 0
                # Accumulate per-event averages weighted by batch size
                bs = inputs.size(0)
                sums[total_points]['mae'] += batch_mae * bs
                sums[total_points]['cov_mae'] += batch_cov_mae * bs
                sums[total_points]['cos'] += batch_cos * bs
                sums[total_points]['cov_cos'] += batch_cov_cos * bs
                # Activity metrics per batch
                act_bce = F.binary_cross_entropy_with_logits(act_logits.squeeze(-1), active_flags)
                act_pred = (torch.sigmoid(act_logits).squeeze(-1) >= 0.5).float()
                act_acc = (act_pred == active_flags).float().mean().item()
                sums[total_points]['act_bce'] += act_bce.item() * bs
                sums[total_points]['act_acc'] += act_acc * bs
                counts[total_points] += bs

    # Print summary
    if sums:
        print("Validation per configuration:")
        for tp in sorted(sums.keys()):
            c = max(counts[tp], 1)
            avg_mae = sums[tp]['mae'] / c
            avg_cov_mae = sums[tp]['cov_mae'] / c
            avg_cos = sums[tp]['cos'] / c
            avg_cov_cos = sums[tp]['cov_cos'] / c
            avg_act_bce = sums[tp]['act_bce'] / c
            avg_act_acc = sums[tp]['act_acc'] / c
            print(f"  {tp} points ({counts[tp]} events): CtrMAE={avg_mae:.6f} | CovMAE={avg_cov_mae:.6f} | CtrCos={avg_cos:.4f} | CovCos={avg_cov_cos:.4f} | ActBCE={avg_act_bce:.6f} | ActAcc={avg_act_acc:.4f}")

def visualise_events_dynamic(model, loader, device, num_events_per_config: int = 1, out_dir: str = "event_vis_dynamic"):
    """Visualize GT vs predicted centres on test set per configuration (variable nodes).

    For each total_points configuration (e.g., 600, 700, 800, 900), take up to
    num_events_per_config examples from the first available batch and plot:
      - detector points (background)
      - GT centres (red ×)
      - predicted centres (blue ○) after Hungarian matching
      - grey match lines
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    from scipy.optimize import linear_sum_assignment
    with torch.no_grad():
        try:
            first_batches = next(iter(loader))
        except StopIteration:
            print("No data in loader to visualize.")
            return
        for total_points, batch in first_batches.items():
            inputs, centres_labels, active_flags, k_value = batch
            inputs = inputs.to(device)
            centres_labels = centres_labels.to(device)
            k_value = k_value.to(device)

            # Forward pass for a small subset
            take = min(num_events_per_config, inputs.size(0))
            preds, _, _, _ = model(inputs[:take])  # (take,Kmax,3)

            for i in range(take):
                k = int(k_value[i].item())
                centres_pred_all = preds[i].cpu().numpy()
                centres_gt = centres_labels[i, :k, :3].cpu().numpy()

                # Rectangular cost matrix (Kmax x k)
                Kmax = centres_pred_all.shape[0]
                cost_matrix = np.zeros((Kmax, k), dtype=np.float32)
                for r in range(Kmax):
                    for c in range(k):
                        cost_matrix[r, c] = np.linalg.norm(centres_pred_all[r] - centres_gt[c])
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                order = np.argsort(col_ind)
                matched_rows = row_ind[order][:k]
                centres_pred_matched = centres_pred_all[matched_rows]

                xyz = inputs[i].detach().cpu().numpy()[:, -3:]

                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='lightgrey', s=4, alpha=0.25, label='Detector nodes')
                ax.scatter(centres_gt[:, 0], centres_gt[:, 1], centres_gt[:, 2], c='red', marker='x', s=100, linewidths=3, label='GT centers')
                ax.scatter(centres_pred_matched[:, 0], centres_pred_matched[:, 1], centres_pred_matched[:, 2], c='blue', marker='o', s=100, linewidths=2, label='Predicted centers')
                for j in range(k):
                    ax.plot([centres_gt[j, 0], centres_pred_matched[j, 0]],
                            [centres_gt[j, 1], centres_pred_matched[j, 1]],
                            [centres_gt[j, 2], centres_pred_matched[j, 2]],
                            'gray', alpha=0.6, linewidth=1)
                ax.set_title(f"T={total_points} | k={k} | sample {i}")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim(-120, 120)
                ax.set_ylim(-120, 120)
                ax.set_zlim(-100, 100)
                ax.view_init(elev=25, azim=135)
                ax.legend()
                plt.tight_layout()
                outfile = os.path.join(out_dir, f"dynamic_T{total_points}_sample_{i}.png")
                plt.savefig(outfile, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved dynamic visualisation: {outfile}")

def visualise_events_dynamic_by_k(
    model,
    loader,
    device,
    desired_k=(3, 4, 5),
    target_points=(600, 700, 800, 900, 1000),
    images_per_config=1,
    override_images_per_config=None,
    out_dir: str = "event_vis_dynamic_k",
):
    """Ensure visuals cover specific ks across specified total_points configs.

    For each k in desired_k and each total_points in target_points, capture a
    number of examples (default 1), with optional overrides per total_points
    (e.g., {600: 2}).
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    from scipy.optimize import linear_sum_assignment

    # Build quota per (T, k) - typeerror here
    quotas = {}
    for tp in target_points:
        imgs = images_per_config
        if override_images_per_config and tp in override_images_per_config:
            imgs = override_images_per_config[tp]
        for k in desired_k:
            quotas[(tp, k)] = imgs

    selected = []  # list of dicts with keys: total_points, x, labels, k

    with torch.no_grad():
        # Iterate through the loader until quotas are satisfied or data exhausted
        for batches in loader:
            for total_points, batch in batches.items():
                if total_points not in target_points:
                    continue
                inputs, centres_labels, active_flags, k_value = batch
                for i in range(inputs.size(0)):
                    k_i = int(k_value[i].item())
                    key = (total_points, k_i)
                    if key in quotas and quotas[key] > 0:
                        selected.append({
                            'total_points': total_points,
                            'x': inputs[i].to(device),
                            'labels': centres_labels[i].to(device),
                            'k': k_i,
                        })
                        quotas[key] -= 1
                # Early-exit if all quotas met
                if all(v == 0 for v in quotas.values()):
                    break
            if all(v == 0 for v in quotas.values()):
                break

        # Report unmet quotas, if any
        unmet = [(tp, k, q) for (tp, k), q in quotas.items() if q > 0]
        if unmet:
            for tp, k, q in unmet:
                print(f"Warning: unmet visual quota for T={tp}, k={k}: missing {q} samples")

        # Generate and save figures for selected samples
        for idx, item in enumerate(selected):
            x = item['x'].unsqueeze(0)  # (1, T, 8)
            labels = item['labels']
            k = int(item['k'])
            tp = item['total_points']

            preds, _, _, _ = model(x)
            centres_pred_all = preds[0].detach().cpu().numpy()  # (Kmax,3)
            centres_gt = labels[:k, :3].detach().cpu().numpy()  # (k,3)

            # Rectangular cost matrix (Kmax x k) + Hungarian matching
            Kmax = centres_pred_all.shape[0]
            cost_matrix = np.zeros((Kmax, k), dtype=np.float32)
            for r in range(Kmax):
                for c in range(k):
                    cost_matrix[r, c] = np.linalg.norm(centres_pred_all[r] - centres_gt[c])
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            order = np.argsort(col_ind)
            matched_rows = row_ind[order][:k]
            centres_pred_matched = centres_pred_all[matched_rows]

            xyz = item['x'].detach().cpu().numpy()[:, -3:]

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='lightgrey', s=4, alpha=0.25, label='Detector nodes')
            ax.scatter(centres_gt[:, 0], centres_gt[:, 1], centres_gt[:, 2], c='red', marker='x', s=100, linewidths=3, label='GT centers')
            ax.scatter(centres_pred_matched[:, 0], centres_pred_matched[:, 1], centres_pred_matched[:, 2], c='blue', marker='o', s=100, linewidths=2, label='Predicted centers')
            for j in range(k):
                ax.plot([centres_gt[j, 0], centres_pred_matched[j, 0]],
                        [centres_gt[j, 1], centres_pred_matched[j, 1]],
                        [centres_gt[j, 2], centres_pred_matched[j, 2]],
                        'gray', alpha=0.6, linewidth=1)
            ax.set_title(f"T={tp} | k={k}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(-120, 120)
            ax.set_ylim(-120, 120)
            ax.set_zlim(-100, 100)
            ax.view_init(elev=25, azim=135)
            ax.legend()
            plt.tight_layout()
            # index per (T,k)
            existing = sum(1 for it in selected[:idx] if it['total_points'] == tp and it['k'] == k)
            outfile = os.path.join(out_dir, f"dynamic_T{tp}_k{k}_idx{existing}.png")
            plt.savefig(outfile, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved dynamic (by-k) visualisation: {outfile}")

def train_model_dynamic(model, train_loader, val_loader, device='cuda:0', n_epochs=8, k_loss_weight: float = 0.1, cov_loss_weight: float = 1.0, checkpoint_dir: str = 'checkpoint_dynamic', breakout_val_loader=None, breakout_test_loader=None):
    device = torch.device(device)
    model = model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    criterion = nn.SmoothL1Loss()

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        total_mae = 0.0
        total_cov_mae = 0.0
        total_cos = 0.0
        total_cov_cos = 0.0
        total_k_acc = 0.0
        steps = 0

        # Interval accumulators for 300-event logging with quick validation
        event_counter = 0
        interval_events = 0
        interval_loss = 0.0
        interval_mae = 0.0
        interval_cov_mae = 0.0
        interval_cos = 0.0
        interval_k_acc = 0.0
        interval_act_bce = 0.0
        interval_act_acc = 0.0

        for batches in train_loader:
            for _, batch in batches.items():
                inputs, cluster_summary_labels, active_flags, k_value = batch
                inputs = inputs.to(device)
                cluster_summary_labels = cluster_summary_labels.to(device)
                active_flags = active_flags.to(device)
                k_value = k_value.to(device)

                optimizer.zero_grad()
                preds, cov_preds, k_logits, activity_logits = model(inputs)

                per_sample_losses = []
                per_sample_mae = []
                per_sample_cov_losses = []
                per_sample_cov_mae = []
                per_sample_cos = []

                for b in range(preds.size(0)):
                    ki = int(k_value[b].item())
                    targets_b = cluster_summary_labels[b, :ki, :3]
                    cov_targets_b = cluster_summary_labels[b, :ki, 3:9]
                    preds_b = preds[b]
                    cov_preds_b = cov_preds[b]

                    cost = torch.cdist(preds_b.detach(), targets_b.detach(), p=1).cpu().numpy()
                    row_ind, col_ind = linear_sum_assignment(cost)

                    matched_preds = preds_b[row_ind]
                    matched_targets = targets_b[col_ind]
                    matched_cov_preds = cov_preds_b[row_ind]
                    matched_cov_targets = cov_targets_b[col_ind]

                    per_sample_losses.append(criterion(matched_preds, matched_targets))
                    per_sample_mae.append(F.l1_loss(matched_preds, matched_targets))
                    per_sample_cov_losses.append(criterion(matched_cov_preds, matched_cov_targets))
                    per_sample_cov_mae.append(F.l1_loss(matched_cov_preds, matched_cov_targets))
                    cos_b = F.cosine_similarity(matched_preds.reshape(1, -1), matched_targets.reshape(1, -1), dim=1).mean()
                    per_sample_cos.append(cos_b)

                reg_loss = torch.stack(per_sample_losses).mean()
                cov_loss = torch.stack(per_sample_cov_losses).mean()
                k_loss = _compute_k_prediction_loss(k_logits, k_value, model.k)
                # Activity BCE loss (node-level)
                activity_bce = F.binary_cross_entropy_with_logits(
                    activity_logits.squeeze(-1), active_flags
                )
                act_loss_weight = 0.5
                loss = reg_loss + cov_loss_weight * cov_loss + k_loss_weight * k_loss + act_loss_weight * activity_bce

                loss.backward()
                optimizer.step()

                batch_mae = torch.stack(per_sample_mae).mean().item()
                batch_cov_mae = torch.stack(per_sample_cov_mae).mean().item()
                batch_cos = torch.stack(per_sample_cos).mean().item()
                k_pred = k_logits.argmax(dim=1) + 1
                k_acc = (k_pred == k_value).float().mean().item()

                total_loss += loss.item()
                total_mae += batch_mae
                total_cov_mae += batch_cov_mae
                total_cos += batch_cos
                total_k_acc += k_acc
                total_cov_cos += batch_cov_mae if False else batch_cos  # placeholder to satisfy lints if needed
                steps += 1

                # Interval accumulations
                bs_now = inputs.size(0)
                interval_events += bs_now
                event_counter += bs_now
                interval_loss += loss.item() * bs_now
                interval_mae += batch_mae * bs_now
                interval_cov_mae += batch_cov_mae * bs_now
                interval_cos += batch_cos * bs_now
                interval_k_acc += k_acc * bs_now
                # activity interval accumulations
                act_pred_interval = (torch.sigmoid(activity_logits).squeeze(-1) >= 0.5).float()
                batch_act_acc_interval = (act_pred_interval == active_flags).float().mean().item()
                interval_act_acc += batch_act_acc_interval * bs_now
                interval_act_bce += activity_bce.item() * bs_now

                # Print every ~300 events with quick validation sample
                if event_counter >= 300:
                    avg_int_loss = interval_loss / max(interval_events, 1)
                    avg_int_mae = interval_mae / max(interval_events, 1)
                    avg_int_cov_mae = interval_cov_mae / max(interval_events, 1)
                    avg_int_cos = interval_cos / max(interval_events, 1)

                    # Quick validation over ~300 events
                    val_events_needed = 300
                    val_events_processed = 0
                    val_mae_sum = 0.0
                    val_cov_mae_sum = 0.0
                    val_cos_sum = 0.0
                    val_k_acc_sum = 0.0
                    val_act_bce_sum = 0.0
                    val_act_acc_sum = 0.0
                    val_loader_iter = iter(val_loader)
                    while val_events_processed < val_events_needed:
                        try:
                            val_batches = next(val_loader_iter)
                        except StopIteration:
                            val_loader_iter = iter(val_loader)
                            val_batches = next(val_loader_iter)
                        for _, vbatch in val_batches.items():
                            v_inputs, v_labels, v_flags, v_k = vbatch
                            v_inputs = v_inputs.to(device)
                            v_labels = v_labels.to(device)
                            v_flags = v_flags.to(device)
                            v_k = v_k.to(device)
                            with torch.no_grad():
                                v_preds, v_cov_preds, v_k_logits, v_act_logits = model(v_inputs)
                            # Per-sample match and metrics
                            per_val_mae = []
                            per_val_cov_mae = []
                            per_val_cos = []
                            for vb in range(v_preds.size(0)):
                                ki_v = int(v_k[vb].item())
                                v_tgt = v_labels[vb, :ki_v, :3]
                                v_cov_tgt = v_labels[vb, :ki_v, 3:9]
                                v_pb = v_preds[vb]
                                v_cb = v_cov_preds[vb]
                                cost = torch.cdist(v_pb.detach(), v_tgt.detach(), p=1).cpu().numpy()
                                row_ind, col_ind = linear_sum_assignment(cost)
                                m_p = v_pb[row_ind]; m_t = v_tgt[col_ind]
                                m_cp = v_cb[row_ind]; m_ct = v_cov_tgt[col_ind]
                                per_val_mae.append(F.l1_loss(m_p, m_t))
                                per_val_cov_mae.append(F.l1_loss(m_cp, m_ct))
                                per_val_cos.append(F.cosine_similarity(m_p.reshape(1, -1), m_t.reshape(1, -1), dim=1).mean())
                            v_batch_mae = torch.stack(per_val_mae).mean().item()
                            v_batch_cov_mae = torch.stack(per_val_cov_mae).mean().item()
                            v_batch_cos = torch.stack(per_val_cos).mean().item()
                            # k acc and activity metrics
                            v_k_acc = ((v_k_logits.argmax(dim=1) + 1) == v_k).float().mean().item()
                            v_act_bce = F.binary_cross_entropy_with_logits(v_act_logits.squeeze(-1), v_flags)
                            v_act_pred = (torch.sigmoid(v_act_logits).squeeze(-1) >= 0.5).float()
                            v_act_acc = (v_act_pred == v_flags).float().mean().item()
                            bsz = v_inputs.size(0)
                            take = min(bsz, val_events_needed - val_events_processed)
                            val_mae_sum += v_batch_mae * take
                            val_cov_mae_sum += v_batch_cov_mae * take
                            val_cos_sum += v_batch_cos * take
                            val_k_acc_sum += v_k_acc * take
                            val_act_bce_sum += v_act_bce.item() * take
                            val_act_acc_sum += v_act_acc * take
                            val_events_processed += take
                            if val_events_processed >= val_events_needed:
                                break

                    avg_val_mae = val_mae_sum / val_events_needed
                    avg_val_cov_mae = val_cov_mae_sum / val_events_needed
                    avg_val_cos = val_cos_sum / val_events_needed
                    avg_val_k_acc = val_k_acc_sum / val_events_needed
                    avg_val_act_bce = val_act_bce_sum / val_events_needed
                    avg_val_act_acc = val_act_acc_sum / val_events_needed
                    print(f"[Epoch {epoch:03d}] Train Avg Loss (last 300 events): {avg_int_loss:.6f} | Train MAE: {avg_int_mae:.6f} | Val MAE: {avg_val_mae:.6f} | Train CosSim: {avg_int_cos:.4f} | Val CosSim: {avg_val_cos:.4f} | Train CovMAE: {avg_int_cov_mae:.6f} | Val CovMAE: {avg_val_cov_mae:.6f} | Train K Acc: {(interval_k_acc/max(interval_events,1)):.4f} | Val K Acc: {avg_val_k_acc:.4f} | Train ActBCE: {(interval_act_bce/max(interval_events,1)):.6f} | Val ActBCE: {avg_val_act_bce:.6f} | Train ActAcc: {(interval_act_acc/max(interval_events,1)):.4f} | Val ActAcc: {avg_val_act_acc:.4f}")
                    # Reset interval accumulators
                    event_counter = 0
                    interval_events = 0
                    interval_loss = 0.0
                    interval_mae = 0.0
                    interval_cov_mae = 0.0
                    interval_cos = 0.0
                    interval_k_acc = 0.0
                    interval_act_bce = 0.0
                    interval_act_acc = 0.0

        if steps == 0:
            steps = 1
        train_metrics = {
            'loss': total_loss / steps,
            'mae': total_mae / steps,
            'cov_mae': total_cov_mae / steps,
            'cos_sim': total_cos / steps,
            'cov_cos': total_cov_cos / steps if steps > 0 else 0.0,
            'k_accuracy': total_k_acc / steps,
        }

        val_metrics = evaluate_model_dynamic(model, val_loader, device)
        # Per-configuration validation breakdown
        print_val_metrics_per_config_dynamic(model, val_loader, device)
        scheduler.step(val_metrics['mae'])

        print(f"[Epoch {epoch:03d}] Loss: {train_metrics['loss']:.6f} | MAE: {train_metrics['mae']:.6f} | CovMAE: {train_metrics['cov_mae']:.6f} | CosSim: {train_metrics['cos_sim']:.4f} | CovCosSim: {train_metrics['cov_cos']:.4f} | K Acc: {train_metrics['k_accuracy']:.4f} || "
              f"Val MAE: {val_metrics['mae']:.6f} | Val CovMAE: {val_metrics['cov_mae']:.6f} | Val CosSim: {val_metrics['cos_sim']:.4f} | Val CovCosSim: {val_metrics.get('cov_cos', 0.0):.4f} | Val K Acc: {val_metrics['k_accuracy']:.4f}")

        # Optional: evaluate explicitly on break-out validation if provided
        if breakout_val_loader is not None:
            bo_val = evaluate_model_dynamic(model, breakout_val_loader, device)
            print(f"[Epoch {epoch:03d}] Val (break-out) | MAE: {bo_val['mae']:.6f} | CovMAE: {bo_val['cov_mae']:.6f} | CosSim: {bo_val['cos_sim']:.4f} | CovCos: {bo_val.get('cov_cos', 0.0):.4f} | K Acc: {bo_val['k_accuracy']:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, os.path.join(checkpoint_dir, f'model_dynamic_epoch_{epoch:03d}.pth'))


# ===========================
# Run Training
# ===========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SetTransformer training (static or dynamic mini-batch mode)')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic mode (variable nodes, no padding)')
    parser.add_argument('--root_data_dir', type=str, default='./data', help='Directory containing *.root files')
    parser.add_argument('--max_files', type=int, default=None, help='Limit number of ROOT files loaded (useful for quick tests)')
    parser.add_argument('--root_max_wafers', type=int, default=600, help='Pad each event to this many wafers (root_mode)')
    parser.add_argument('--root_max_k', type=int, default=10, help='Maximum number of clusters to predict (root_mode)')
    parser.add_argument('--root_min_trainable', type=int, default=1, help='Minimum trainable clusters; events below threshold are skipped (root_mode)')
    parser.add_argument('--data_dir', type=str, default='./synthetic_events', help='Base directory with train/val/test splits')
    parser.add_argument('--train_points', type=str, default='3000', help='Comma-separated train total_points list')
    parser.add_argument('--val_points', type=str, default='3000', help='Comma-separated val total_points list')
    parser.add_argument('--test_points', type=str, default='3000', help='Comma-separated test total_points list')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0', help='Training device')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume checkpoint path (static mode)')
    parser.add_argument('--final_test_size', type=int, default=5000, help='Hold-out size from training for final test (stratified by k)')
    # Convenience: pass an epoch number to resume from a specific checkpoint without specifying a path
    parser.add_argument('--resume_epoch', type=int, default=None, help='Resume from epoch N (infers checkpoint path)')
    # For dynamic mode weight-only resume, this is where dynamic checkpoints are stored
    parser.add_argument('--dynamic_checkpoint_dir', type=str, default='checkpoint_dynamic', help='Directory for dynamic mode checkpoints')
    # LR control (static mode)
    parser.add_argument('--fixed_lr', action='store_true', help='Use a fixed learning rate (disables OneCycle)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate for Adam')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Max LR for OneCycle (ignored if --fixed_lr)')
    args = parser.parse_args()

    if args.dynamic:
        train_points = [int(x) for x in args.train_points.split(',') if x]
        val_points = [int(x) for x in args.val_points.split(',') if x]
        test_points = [int(x) for x in args.test_points.split(',') if x]

        print("Dynamic mode enabled")
        print(f"Train points: {train_points}")
        print(f"Val   points: {val_points}")
        print(f"Test  points: {test_points}")

        train_datasets = load_dynamic_datasets(args.data_dir, 'train', train_points)
        val_datasets = load_dynamic_datasets(args.data_dir, 'val', val_points)
        test_datasets = load_dynamic_datasets(args.data_dir, 'test', test_points)
        if not train_datasets:
            raise ValueError('No training datasets found for dynamic mode')

        # Print k distribution per configuration for each split
        print("K distribution check per split/configuration:")
        for ds_list, split_name in [(train_datasets, 'Train'), (val_datasets, 'Val'), (test_datasets, 'Test')]:
            for ds in ds_list:
                total_points = getattr(ds, 'total_points', 'unknown')
                print(f"  {split_name} {total_points}pts:")
                _print_k_distribution(getattr(ds, 'k_values', None))

        train_loader = MultiConfigDataLoader(train_datasets, batch_size=args.batch_size, shuffle=True)
        val_loader = MultiConfigDataLoader(val_datasets, batch_size=args.batch_size, shuffle=False)
        test_loader = MultiConfigDataLoader(test_datasets, batch_size=args.batch_size, shuffle=False)

        # Build dynamic break-out holdout from training datasets (~10%), split 50/50 val/test
        rng = np.random.RandomState(42)
        breakout_val_datasets = []
        breakout_test_datasets = []
        reduced_train_datasets = []
        for ds in train_datasets:
            k_vals = getattr(ds, 'k_values', None)
            if k_vals is None:
                reduced_train_datasets.append(ds)
                continue
            indices = np.arange(len(ds))
            k_unique = np.unique(k_vals)
            k_to_idxs = {}
            for k in k_unique:
                idxs = indices[k_vals == k]
                idxs = idxs.copy()
                rng.shuffle(idxs)
                k_to_idxs[int(k)] = list(idxs)
            target_size = max(1, len(indices) // 10)
            selected = []
            k_list = sorted(k_to_idxs.keys())
            while len(selected) < target_size and any(len(k_to_idxs[kk]) > 0 for kk in k_list):
                for kk in k_list:
                    if len(selected) >= target_size:
                        break
                    if k_to_idxs[kk]:
                        selected.append(k_to_idxs[kk].pop())
            selected = np.array(selected, dtype=int)
            remain_mask = np.ones(len(indices), dtype=bool)
            remain_mask[selected] = False
            remain_idx = indices[remain_mask]
            split_p = len(selected) // 2
            sel_val = selected[:split_p]
            sel_test = selected[split_p:]
            from torch.utils.data import Subset as TorchSubset
            ds_rem = TorchSubset(ds, remain_idx.tolist())
            ds_val = TorchSubset(ds, sel_val.tolist()) if len(sel_val) > 0 else None
            ds_test = TorchSubset(ds, sel_test.tolist()) if len(sel_test) > 0 else None
            for dss in [ds_rem, ds_val, ds_test]:
                if dss is None:
                    continue
                dss.total_points = ds.total_points
                dss.num_nodes = ds.num_nodes
                dss.k_values = k_vals[dss.indices]
            reduced_train_datasets.append(ds_rem)
            if ds_val is not None:
                breakout_val_datasets.append(ds_val)
            if ds_test is not None:
                breakout_test_datasets.append(ds_test)

        # Replace train loader with reduced version and create optional break-out loaders
        train_loader = MultiConfigDataLoader(reduced_train_datasets, batch_size=args.batch_size, shuffle=True)
        breakout_val_loader = MultiConfigDataLoader(breakout_val_datasets, batch_size=args.batch_size, shuffle=False) if breakout_val_datasets else None
        breakout_test_loader = MultiConfigDataLoader(breakout_test_datasets, batch_size=args.batch_size, shuffle=False) if breakout_test_datasets else None

        # Print dataset sizes
        total_train_events = sum(len(dataset) for dataset in train_datasets)
        total_val_events = sum(len(dataset) for dataset in val_datasets)
        total_test_events = sum(len(dataset) for dataset in test_datasets)
        print(f"Training dataset: {total_train_events} events")
        print(f"Validation dataset: {total_val_events} events")
        print(f"Test dataset: {total_test_events} events")

        # Determine max_k across all training datasets
        max_k = int(max(ds.cluster_summary_labels.shape[1] for ds in train_datasets))

        config = GPTConfig(block_size=3333, n_layer=6, n_head=8, n_embd=256, dropout=0.1)
        model = SetTransformer(config, input_dim=8, k=max_k)

        # Optional: resume weights in dynamic mode from a given epoch (weights-only)
        if args.resume_epoch is not None:
            resume_dyn_path = os.path.join(args.dynamic_checkpoint_dir, f'model_dynamic_epoch_{args.resume_epoch:03d}.pth')
            if os.path.exists(resume_dyn_path):
                print(f"Loading dynamic checkpoint weights from {resume_dyn_path}")
                dyn_ckpt = torch.load(resume_dyn_path, map_location=args.device)
                model.load_state_dict(dyn_ckpt['model_state_dict'])
            else:
                print(f"Warning: dynamic resume checkpoint not found at {resume_dyn_path}; proceeding without resume.")

        print("Training (dynamic) with mini-batches across variable-node configs")
        train_model_dynamic(
            model,
            train_loader,
            val_loader,
            device=args.device,
            n_epochs=args.epochs,
            breakout_val_loader=breakout_val_loader,
            breakout_test_loader=breakout_test_loader,
        )

        # Final evaluation on test (dynamic)
        test_metrics = evaluate_model_dynamic(model, test_loader, args.device)
        print("\nFinal Test (dynamic) | "
              f"MAE: {test_metrics['mae']:.6f} | CovMAE: {test_metrics['cov_mae']:.6f} | Cos: {test_metrics['cos_sim']:.4f} | CovCos: {test_metrics.get('cov_cos', 0.0):.4f} | "
              f"K Acc: {test_metrics['k_accuracy']:.4f}")
        # Per-configuration test breakdown
        print_val_metrics_per_config_dynamic(model, test_loader, args.device)
        # Also per-configuration for break-out test if present
        if breakout_test_loader is not None:
            print("Break-out Test per configuration (dynamic mode):")
            print_val_metrics_per_config_dynamic(model, breakout_test_loader, args.device)
        # Visualise a few test events per configuration (e.g., 600/700/800/900/1000)
        visualise_events_dynamic(model, test_loader, args.device, num_events_per_config=5, out_dir="event_vis_dynamic_test")
        # Ensure visuals cover specific ks across specified total_points; 600 gets 2 images per k
        visualise_events_dynamic_by_k(
            model,
            test_loader,
            args.device,
            desired_k=(3, 4, 5),
            target_points=(1000,), #edited to treat as a tuple
            images_per_config=1,
            override_images_per_config={600: 6},
            out_dir="event_vis_dynamic_test_k"
        )
    else:
        import glob
        root_files = sorted(glob.glob(os.path.join(args.root_data_dir, '*.root')))
        if not root_files:
            raise FileNotFoundError(f"No *.root files found in {args.root_data_dir}")
        if args.max_files is not None:
            root_files = root_files[:args.max_files]
        print(f"Found {len(root_files)} ROOT file(s) in {args.root_data_dir}")
        dataset = RootWaferDataset(
            root_files,
            max_wafers=args.root_max_wafers,
            max_k=args.root_max_k,
            min_trainable=args.root_min_trainable,
        )
        max_k = args.root_max_k
        print(f"Dataset: {len(dataset)} events, max_wafers={args.root_max_wafers}, max_k={max_k}")
        _print_k_distribution(dataset.k_values)

        config = GPTConfig(block_size=3333, n_layer=6, n_head=8, n_embd=256, dropout=0.1)
        model = SetTransformer(config, input_dim=8, k=max_k)
        if args.resume_epoch is not None and (args.resume_from is None or args.resume_from == ''):
            auto_resume_path = os.path.join('checkpoint', f'model_epoch_{args.resume_epoch:03d}.pth')
            print(f"Auto-inferred resume checkpoint from epoch {args.resume_epoch}: {auto_resume_path}")
            args.resume_from = auto_resume_path
        train_model(
            model,
            dataset,
            device=args.device,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            resume_from=args.resume_from,
            final_test_size=args.final_test_size,
            fixed_lr=args.fixed_lr,
            base_lr=args.lr,
            max_lr=args.max_lr,
        )