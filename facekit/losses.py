import torch
import torch.nn.functional as F

from .layout import L_EYE_OUTER, N_CLASSES, R_EYE_OUTER


def focal_loss(logits, gt):
    """CenterNet focal loss for the heatmap."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    is_centre = gt.eq(1).float()
    not_centre = 1.0 - is_centre
    centre_loss = -torch.log(p) * (1 - p) ** 2 * is_centre
    other_loss = -torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * not_centre
    num_faces = is_centre.sum().clamp_min(1.0)
    return (centre_loss.sum() + other_loss.sum()) / num_faces


def detector_loss(out, hm, reg, mask):
    """Detector loss = focal loss on the heatmap + L1 loss on size and offset."""
    focal = focal_loss(out[:, 0:1], hm)
    num_faces = mask.sum().clamp_min(1.0)
    l1 = ((out[:, 1:5] - reg).abs() * mask).sum() / num_faces
    return focal + l1, {"focal": float(focal.detach()), "l1": float(l1.detach())}


def landmark_loss(pred, gt):
    """L1 distance between predicted and real points, averaged over the 106 points."""
    dist = (pred - gt).abs().sum(-1).mean(-1)
    eye_dist = (gt[:, R_EYE_OUTER] - gt[:, L_EYE_OUTER]).norm(dim=-1).clamp_min(1e-3)
    return (dist / eye_dist).mean()


def soft_dice_loss(logits, labels, n_classes=N_CLASSES, eps=1.0):
    """1 - average soft Dice score over the classes (softmax vs one-hot labels)."""
    prob = F.softmax(logits, 1)
    onehot = F.one_hot(labels.long(), n_classes).permute(0, 3, 1, 2).to(prob.dtype)
    dims = (0, 2, 3)
    overlap = (prob * onehot).sum(dims)
    total = prob.sum(dims) + onehot.sum(dims)
    dice = (2 * overlap + eps) / (total + eps)
    return 1.0 - dice.mean()


def ce_dice_loss(logits, labels, class_weight=None):
    """Parser loss = cross entropy + soft Dice."""
    ce = F.cross_entropy(logits, labels.long(), weight=class_weight)
    dice = soft_dice_loss(logits, labels, logits.shape[1])
    return ce + dice, {"ce": float(ce.detach()), "dice": float(dice.detach())}
