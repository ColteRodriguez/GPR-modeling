"""
Column-Connection Clustering (C3) Algorithm for GPR B-scan data.
Based on: Dou et al. (2017) "Real-Time Hyperbola Recognition and Fitting in GPR Data"
IEEE Transactions on Geoscience and Remote Sensing, 55(1), pp. 51-62.

Assumes the B-scan has already been pre-processed (e.g. mean subtraction, filtering).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Cluster:
    """A cluster produced by the C3 algorithm."""
    id: int
    # Maps column index -> list of row indices belonging to this cluster
    columns: dict = field(default_factory=dict)

    @property
    def all_points(self) -> np.ndarray:
        """Return all (row, col) points in the cluster as an (N, 2) array."""
        pts = [(r, c) for c, rows in self.columns.items() for r in rows]
        return np.array(pts, dtype=int) if pts else np.empty((0, 2), dtype=int)

    @property
    def central_string(self) -> np.ndarray:
        """
        Return the central string: the midpoint row for each column,
        as an (M, 2) array of (row, col) sorted by column.
        """
        cols = sorted(self.columns.keys())
        return np.array(
            [(int(np.mean(self.columns[c])), c) for c in cols], dtype=int
        )

    @property
    def width(self) -> int:
        return len(self.columns)

    @property
    def size(self) -> int:
        return sum(len(v) for v in self.columns.values())


def threshold_bscan(bscan: np.ndarray, rho_b: float = 0.1) -> np.ndarray:
    """
    Adaptive thresholding (Section II-A of the paper).

    Computes a threshold from edge pixel intensities:
        threshold = mean{ Ie | Ie > rho_b * max(Ie) }

    Args:
        bscan:  2-D float array, already pre-processed.
        rho_b:  Fraction in (0, 1). Paper uses 0.1 by default.

    Returns:
        Binary 2-D uint8 array (1 = region of interest, 0 = background).
    """
    # Simple gradient-magnitude edge map
    gy = np.abs(np.diff(bscan, axis=0, prepend=bscan[:1]))
    gx = np.abs(np.diff(bscan, axis=1, prepend=bscan[:, :1]))
    edge_mag = np.hypot(gx, gy)

    edge_vals = edge_mag[edge_mag > 0]
    if edge_vals.size == 0:
        return np.zeros_like(bscan, dtype=np.uint8)

    max_ie = edge_vals.max()
    selected = edge_vals[edge_vals > rho_b * max_ie]
    thresh = selected.mean() if selected.size > 0 else max_ie

    binary = (bscan >= thresh).astype(np.uint8)
    return binary


def _get_column_segments(col: np.ndarray, s: int) -> list[list[int]]:
    """
    Extract column segments: runs of consecutive non-zero pixels of length >= s.

    Args:
        col: 1-D binary array for one column.
        s:   Minimum run length.

    Returns:
        List of segments, each segment is a list of row indices.
    """
    segments = []
    n = len(col)
    i = 0
    while i < n:
        if col[i]:
            j = i
            while j < n and col[j]:
                j += 1
            run_rows = list(range(i, j))
            if len(run_rows) >= s:
                segments.append(run_rows)
            i = j
        else:
            i += 1
    return segments


def _connecting_elements(seg_a: list[int], seg_b: list[int]) -> int:
    """Count elements (rows) shared between two column segments."""
    return len(set(seg_a) & set(seg_b))


def c3_clustering(
    bscan: np.ndarray,
    s: int = 3,
    threshold: Optional[float] = None,
    rho_b: float = 0.1,
    min_cluster_width: int = 3,
    apply_derivative_split: bool = True,
) -> tuple[list[Cluster], np.ndarray]:
    """
    Column-Connection Clustering (C3) algorithm (Section II-B).

    Args:
        bscan:                 2-D float array (rows = time samples, cols = traces).
                               Should already be pre-processed.
        s:                     Minimum consecutive-pixel run length for a column
                               segment. Paper uses s=3. Acts as a noise gate.
        threshold:             If provided, use this fixed threshold value to binarise
                               the bscan instead of the adaptive method.
        rho_b:                 Parameter for the adaptive thresholding (ignored when
                               `threshold` is given). Paper default: 0.1.
        min_cluster_width:     Discard clusters that span fewer columns than this.
        apply_derivative_split: If True, apply the second-pass derivative-based split
                               for fused double-hyperbola clusters (Section II-B).

    Returns:
        clusters:   List of Cluster objects.
        binary:     The binary image produced by thresholding.
    """
    if bscan.ndim != 2:
        raise ValueError("bscan must be a 2-D array.")

    # ------------------------------------------------------------------ #
    # 1. Binarise
    # ------------------------------------------------------------------ #
    if threshold is not None:
        binary = (bscan >= threshold).astype(np.uint8)
    else:
        binary = threshold_bscan(bscan, rho_b=rho_b)

    n_rows, n_cols = binary.shape

    # ------------------------------------------------------------------ #
    # 2. C3 scan (left → right)
    # ------------------------------------------------------------------ #
    # active_clusters is a list of Cluster objects currently being extended.
    active_clusters: list[Cluster] = []
    cluster_id = 0

    for col_idx in range(n_cols):
        col_segs = _get_column_segments(binary[:, col_idx], s)

        if col_idx == 0:
            # Seed one cluster per column segment in the first column
            for seg in col_segs:
                c = Cluster(id=cluster_id)
                c.columns[col_idx] = seg
                active_clusters.append(c)
                cluster_id += 1
            continue

        # For each new column segment, find which active clusters connect to it
        new_active: list[Cluster] = []
        seg_handled = [False] * len(col_segs)

        for clust in active_clusters:
            # Does this cluster's last column connect to any new segment?
            last_col = max(clust.columns.keys())
            if last_col != col_idx - 1:
                # Cluster already stopped; keep it dormant (we'll flush later)
                new_active.append(clust)
                continue

            prev_seg = clust.columns[last_col]
            matched_segs = [
                (j, seg) for j, seg in enumerate(col_segs)
                if _connecting_elements(prev_seg, seg) >= 1
            ]

            if not matched_segs:
                # No connection → cluster stops here
                new_active.append(clust)
            elif len(matched_segs) == 1:
                # Exactly one match → extend the cluster
                j, seg = matched_segs[0]
                clust.columns[col_idx] = seg
                seg_handled[j] = True
                new_active.append(clust)
            else:
                # Multiple matches → split into child clusters
                for j, seg in matched_segs:
                    child = Cluster(id=cluster_id)
                    cluster_id += 1
                    # Copy parent history
                    for cc, rows in clust.columns.items():
                        child.columns[cc] = list(rows)
                    child.columns[col_idx] = seg
                    seg_handled[j] = True
                    new_active.append(child)

        # Any unmatched new segments start fresh clusters
        for j, seg in enumerate(col_segs):
            if not seg_handled[j]:
                c = Cluster(id=cluster_id)
                c.columns[col_idx] = seg
                new_active.append(c)
                cluster_id += 1

        active_clusters = new_active

    all_clusters = active_clusters  # everything is now finalised

    # ------------------------------------------------------------------ #
    # 3. Filter by minimum width
    # ------------------------------------------------------------------ #
    all_clusters = [c for c in all_clusters if c.width >= min_cluster_width]

    # ------------------------------------------------------------------ #
    # 4. Optional: derivative-based split for fused double-hyperbola shapes
    #    (Section II-B, Figure 7)
    #    Detects a local minimum in the central string (dy/dx = 0, d²y/dx² > 0)
    #    and breaks the cluster there.
    # ------------------------------------------------------------------ #
    if apply_derivative_split:
        extra: list[Cluster] = []
        for clust in all_clusters:
            cs = clust.central_string  # shape (W, 2) — (row, col)
            if len(cs) < 5:
                continue
            rows = cs[:, 0].astype(float)
            # First finite difference as proxy for dy/dx
            dy = np.diff(rows)
            # Look for a sign change from negative→positive (valley = local min in row)
            split_col_idx = None
            for k in range(1, len(dy)):
                if dy[k - 1] < 0 and dy[k] >= 0:
                    # Second derivative check (positive = valley)
                    if k + 1 < len(dy) and (dy[k] - dy[k - 1]) > 0:
                        split_col_idx = k  # index into cs
                        break

            if split_col_idx is None:
                continue

            split_col = cs[split_col_idx, 1]  # actual column number
            cols_sorted = sorted(clust.columns.keys())

            left_cols = [c for c in cols_sorted if c <= split_col]
            right_cols = [c for c in cols_sorted if c > split_col]

            if len(left_cols) < min_cluster_width or len(right_cols) < min_cluster_width:
                continue  # not worth splitting

            left_c = Cluster(id=cluster_id)
            cluster_id += 1
            for c in left_cols:
                left_c.columns[c] = clust.columns[c]

            right_c = Cluster(id=cluster_id)
            cluster_id += 1
            for c in right_cols:
                right_c.columns[c] = clust.columns[c]

            extra.extend([left_c, right_c])

        all_clusters.extend(extra)

    # ------------------------------------------------------------------ #
    # 5. Re-assign contiguous IDs and return
    # ------------------------------------------------------------------ #
    for new_id, clust in enumerate(all_clusters):
        clust.id = new_id

    return all_clusters, binary


# --------------------------------------------------------------------------- #
# Convenience helpers
# --------------------------------------------------------------------------- #

def cluster_label_image(clusters: list[Cluster], shape: tuple[int, int]) -> np.ndarray:
    """
    Return an integer label image of shape `shape`.
    Background = 0; cluster i gets label i+1.
    """
    label_img = np.zeros(shape, dtype=np.int32)
    for clust in clusters:
        for col, rows in clust.columns.items():
            for r in rows:
                if 0 <= r < shape[0] and 0 <= col < shape[1]:
                    label_img[r, col] = clust.id + 1
    return label_img


def mask_cluster(bscan: np.ndarray, cluster: Cluster) -> np.ndarray:
    """Return a boolean mask (same shape as bscan) for a single cluster."""
    mask = np.zeros(bscan.shape, dtype=bool)
    for col, rows in cluster.columns.items():
        for r in rows:
            if 0 <= r < bscan.shape[0] and 0 <= col < bscan.shape[1]:
                mask[r, col] = True
    return mask


# --------------------------------------------------------------------------- #
# Example usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Synthetic B-scan with two overlapping hyperbola-shaped blobs ---
    rows, cols = 200, 300
    bscan = np.random.rand(rows, cols) * 0.1  # noise floor

    def add_hyperbola(img, x0, y0, a=20, b=40, amp=1.0, width=3):
        for x in range(img.shape[1]):
            dx = x - x0
            val = y0 + a * np.sqrt(1 + (dx / b) ** 2)
            y_int = int(round(val))
            for dy in range(-width, width + 1):
                yy = y_int + dy
                if 0 <= yy < img.shape[0]:
                    img[yy, x] += amp * max(0, 1 - abs(dy) / width)

    add_hyperbola(bscan, x0=80,  y0=30, a=15, b=35, amp=1.0)
    add_hyperbola(bscan, x0=200, y0=50, a=20, b=50, amp=0.9)

    clusters, binary = c3_clustering(
        bscan,
        s=3,
        rho_b=0.1,
        min_cluster_width=10,
        apply_derivative_split=True,
    )

    print(f"Found {len(clusters)} clusters")
    for c in clusters:
        pts = c.all_points
        print(f"  Cluster {c.id}: {pts.shape[0]} points, "
              f"columns {min(c.columns)}-{max(c.columns)}")

    label_img = cluster_label_image(clusters, bscan.shape)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(bscan, cmap="gray", aspect="auto")
    axes[0].set_title("B-scan (synthetic)")
    axes[1].imshow(binary, cmap="gray", aspect="auto")
    axes[1].set_title("Binary (after thresholding)")
    axes[2].imshow(label_img, cmap="tab20", aspect="auto")
    # Overlay central strings
    for c in clusters:
        cs = c.central_string
        if len(cs) > 1:
            axes[2].plot(cs[:, 1], cs[:, 0], "w-", linewidth=1)
    axes[2].set_title("Cluster labels + central strings")
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/c3_demo.png", dpi=120)
    plt.show()
    print("Demo plot saved to c3_demo.png")
