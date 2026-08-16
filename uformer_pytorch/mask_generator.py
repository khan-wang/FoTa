"""Mask-bank sampling for FoTa-Net training.

The mask conventions follow LaMa and ZITS++ training practice. See
THIRD_PARTY_NOTICES.md for attribution and license information.
"""

import cv2
import numpy as np
import random
import torch
from pathlib import Path
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace


class ZITSStyleMask:
    """
    ZITS++-style mask-bank sampler with curriculum and caching support.
    """

    def __init__(self, config, device='cpu'):
        # --- Core Parameters ---
        self.img_size = config.IMG_SIZE
        self.device = device
        self.config = config

        # --- Unify MASK_BANK config to a SimpleNamespace for consistent access ---
        cfg_bank = getattr(config, "MASK_BANK", None)
        if isinstance(cfg_bank, dict):
            cfg_bank = SimpleNamespace(**cfg_bank)
        self.cfg_bank = cfg_bank

        # --- Performance & Caching (LRU) ---
        self.cache_limit = getattr(config, "MASK_CACHE_LIMIT", 1200)
        self._cache = OrderedDict()
        # Lazily initialize thread pool for DataLoader worker compatibility
        self.pool = None

        # --- Data Loading from MASK_BANK ---
        self.bin_to_paths = {'irr': defaultdict(list), 'coco': defaultdict(list)}
        self._load_masks_from_bank()

        # --- Logging ---
        self.histogram = defaultdict(int)

    def _load_masks_from_bank(self):
        """
        Scans directories defined in MASK_BANK config and populates the mask path dictionary.
        """
        cfg_bank = self.cfg_bank
        if not cfg_bank:
            print(" MASK_BANK not found in config. Mask generator will be empty.")
            return

        print("Initializing Mask Bank...")
        for source in cfg_bank.TRAIN_SOURCES:
            source_key = source.lower()
            root_path = Path(getattr(cfg_bank, f"{source.upper()}_ROOT", ""))

            if not root_path.is_dir():
                print(f"  -  Warning: {source.upper()}_ROOT not found at '{root_path}', skipping.")
                continue

            print(f"  - Scanning source: '{source_key}' at '{root_path}'")
            total_found_in_source = 0
            for bin_name in cfg_bank.BINS:
                bin_dir = root_path / f"mask_rates_{bin_name}"
                if bin_dir.is_dir():
                    paths = []
                    for pat in ("*.png", "*.jpg", "*.jpeg"):
                        paths.extend([str(p) for p in bin_dir.glob(pat)])

                    if paths:
                        self.bin_to_paths[source_key][bin_name].extend(paths)
                        print(f"    - Found {len(paths)} masks in bin '{bin_name}'")
                        total_found_in_source += len(paths)
            print(f"   Loaded {total_found_in_source} masks for source '{source_key}'.")

    def _load_mask_from_path(self, path, H, W):
        """Loads a single mask file, resizes, binarizes, and caches it with LRU policy."""
        key = (path, H, W)
        if key in self._cache:
            mask = self._cache.pop(key)
            self._cache[key] = mask
            return mask

        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"cv2.imread returned None for path: {path}")

            mask = (img > 127).astype(np.uint8)
            if mask.shape[0] != H or mask.shape[1] != W:
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

            if len(self._cache) >= self.cache_limit:
                self._cache.popitem(last=False)  # Evict least recently used
            self._cache[key] = mask
            return mask
        except Exception as e:
            print(f"Error loading mask '{path}': {e}. Returning an empty mask.")
            return np.zeros((H, W), dtype=np.uint8)

    def sample(self, H, W, allowed_bins=None):
        """
        Samples a single mask according to the curriculum and mix ratio with robust fallbacks.
        (Modified: Implements weighted sampling for 'irr' to prioritize large masks)
        """
        cfg_bank = self.cfg_bank
        if not allowed_bins:
            allowed_bins = cfg_bank.BINS


        assert len(cfg_bank.MIX_RATIO) == len(cfg_bank.TRAIN_SOURCES), \
            "MASK_BANK.MIX_RATIO length must match MASK_BANK.TRAIN_SOURCES length."

        source_type = random.choices(cfg_bank.TRAIN_SOURCES, weights=cfg_bank.MIX_RATIO, k=1)[0]

        valid_bins = [b for b in allowed_bins if b in self.bin_to_paths[source_type]]

        if not valid_bins:
            # Try the other source first before giving up on the curriculum for this step
            other_source = next((s for s in cfg_bank.TRAIN_SOURCES if s != source_type), None)
            if other_source:
                alternative_bins = [b for b in allowed_bins if b in self.bin_to_paths[other_source]]
                if alternative_bins:
                    source_type, valid_bins = other_source, alternative_bins

        if not valid_bins:
            # Final fallback: use all available bins for the original source type
            valid_bins = list(self.bin_to_paths[source_type].keys())
            if not valid_bins:
                raise RuntimeError(f"No masks found for source '{source_type}' in any bin.")



        if source_type == 'irr':
            weights = []
            for b in valid_bins:
                if b == '40_50':
                    weights.append(5.0)
                elif b == '30_40':
                    weights.append(2.0)
                else:
                    weights.append(1.0)
            chosen_bin = random.choices(valid_bins, weights=weights, k=1)[0]
        else:

            chosen_bin = random.choice(valid_bins)


        mask_path = random.choice(self.bin_to_paths[source_type][chosen_bin])

        self.histogram[f"{source_type}/{chosen_bin}"] += 1

        return self._load_mask_from_path(mask_path, H, W)
    def __call__(self, batch_size, allowed_bins=None, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        H, W = self.img_size, self.img_size

        if self.pool is None:

            max_threads = int(getattr(self.config, "MASK_IO_THREADS", 4))
            # Ensure we don't create more threads than tasks
            num_workers = min(max_threads, batch_size) if batch_size > 0 else max_threads
            self.pool = ThreadPoolExecutor(max_workers=num_workers)

        futures = [self.pool.submit(self.sample, H, W, allowed_bins) for _ in range(batch_size)]
        masks = [f.result() for f in futures]

        mask_array = np.stack(masks, axis=0).astype(np.float32)
        mask_array = mask_array[:, None, ...]
        mask_batch = torch.from_numpy(mask_array).to(self.device)

        avg_ratio = np.mean([m.mean() for m in masks]) if masks else 0.0

        return mask_batch, avg_ratio

    def report_and_reset_histogram(self, epoch):
        if not self.histogram:
            return {}

        print(f"\n--- Mask Bucket Histogram (End of Epoch {epoch}) ---")
        hist_copy = self.histogram.copy()
        total_samples = sum(hist_copy.values())

        sorted_keys = sorted(hist_copy.keys())
        for key in sorted_keys:
            count = hist_copy[key]
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"  Bucket [{key}]: {count} samples ({percentage:.1f}%)")

        print("-------------------------------------------------")
        self.histogram.clear()
        return hist_copy

    def __del__(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False)
