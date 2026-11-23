import numpy as np

from .layout import FACE_CLASSES, L_EYE_OUTER, LIP_CLASSES, N_CLASSES, R_EYE_OUTER, REGIONS


def nme_per_point(pred, gt):
    """Error of each point divided by the outer eye corner distance (points 66 and 79)."""
    pred = np.asarray(pred, np.float64)
    gt = np.asarray(gt, np.float64)
    eye_dist = np.linalg.norm(gt[:, R_EYE_OUTER] - gt[:, L_EYE_OUTER], axis=-1).clip(1e-6)
    return np.linalg.norm(pred - gt, axis=-1) / eye_dist[:, None]


def nme_report(err, fail_thr=0.08):
    """Summary of the landmark error (err comes from nme_per_point)."""
    per_face = err.mean(1)
    out = {}
    for region, idx in REGIONS.items():
        out[f"nme_{region}"] = float(100 * err[:, idx].mean())
    out["nme_all"] = float(100 * per_face.mean())
    out["nme_median"] = float(100 * np.median(per_face))
    out["failure_rate"] = float((per_face > fail_thr).mean())
    out["n_faces"] = int(len(per_face))
    return out


def confusion(pred, gt, n=N_CLASSES):
    """n x n confusion matrix. Rows = true class, columns = predicted class."""
    pred = np.asarray(pred).reshape(-1).astype(np.int64)
    gt = np.asarray(gt).reshape(-1).astype(np.int64)
    ok = (gt >= 0) & (gt < n)
    counts = np.bincount(n * gt[ok] + pred[ok], minlength=n * n)
    return counts.reshape(n, n)


def iou_f1(cm):
    """IoU and F1 for every class from a confusion matrix."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    row = cm.sum(1)
    col = cm.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / (row + col - tp)
        f1 = 2 * tp / (row + col)
    return iou, f1


def merge_classes(cm, group):
    """Merge the classes in `group` into the first one."""
    group = list(group)
    keep = [c for c in range(cm.shape[0]) if c not in group[1:]]
    merged = cm.copy()
    for c in group[1:]:
        merged[group[0], :] += merged[c, :]
    for c in group[1:]:
        merged[:, group[0]] += merged[:, c]
    return merged[np.ix_(keep, keep)], keep.index(group[0])


def parsing_report(cm, class_names):
    iou, f1 = iou_f1(cm)
    lips_cm, lips_idx = merge_classes(cm, LIP_CLASSES)
    lips_iou = iou_f1(lips_cm)[0][lips_idx]
    per_class = {}
    for c in range(len(iou)):
        per_class[class_names[c]] = {"iou": float(iou[c]), "f1": float(f1[c])}
    return {
        "miou_face10": float(np.nanmean(iou[FACE_CLASSES])),
        "lips_iou": float(lips_iou),
        "per_class": per_class,
        "pixel_acc": float(np.trace(cm) / max(cm.sum(), 1)),
    }


def roc_curve(scores, labels):
    """ROC curve: false positive rate, true positive rate and thresholds (high score to low)."""
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels).astype(bool)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_pos = np.cumsum(sorted_labels)
    false_pos = np.cumsum(~sorted_labels)
    distinct = np.r_[np.nonzero(np.diff(sorted_scores))[0], len(sorted_scores) - 1]
    num_pos = max(sorted_labels.sum(), 1)
    num_neg = max((~sorted_labels).sum(), 1)
    tpr = np.r_[0.0, true_pos[distinct] / num_pos]
    fpr = np.r_[0.0, false_pos[distinct] / num_neg]
    thresholds = np.r_[np.inf, sorted_scores[distinct]]
    return fpr, tpr, thresholds


def auc(fpr, tpr):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    return float(np.trapz(tpr, fpr))
