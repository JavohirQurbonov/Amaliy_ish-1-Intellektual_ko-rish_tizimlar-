# -*- coding: utf-8 -*-
"""
evaluation.py - natijalarni etalon (ground truth) bilan solishtirib baholash.

Ko'rsatkichlar:
  * IoU = |A ∩ B| / |A ∪ B|  - varaq niqobi (distorsiyasiz kadrda) va matn niqobi (tekislangan kadrda)
  * Burchak xatosi (px)      - topilgan va haqiqiy burchaklar orasidagi Evklid masofa
  * QC aniqligi              - sifat nazorati xulosasining etalon yorlig'iga mos kelishi
"""
import json
import cv2
import numpy as np
from config import *
from preprocessing import order_points


def load_meta(name):
    p = GT_DIR / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def iou(a, b):
    a, b = a > 0, b > 0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def _plane_from_undist_px(q, newK, meta):
    """Distorsiyasiz piksel (newK kadrida) -> tekislik koordinatasi (mm), haqiqiy poza bo'yicha."""
    R = cv2.Rodrigues(np.array(meta["rvec"]))[0]
    t = np.array(meta["t"])
    H0 = np.column_stack([R[:, 0], R[:, 1], t])
    n = np.c_[q, np.ones(len(q))] @ np.linalg.inv(newK).T
    ph = n @ np.linalg.inv(H0).T
    return ph[:, :2] / ph[:, 2:3]


def gt_corners_undist(meta, cal):
    """Haqiqiy burchaklarning distorsiyasiz kadrdagi koordinatalari."""
    pts = np.array(meta["burchaklar_px_distorsiyali"], np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, cal["K"], cal["dist"], P=cal["newK"]).reshape(-1, 2)
    return order_points(und)


def gt_sheet_mask_undist(meta, cal, shape):
    """Etalon varaq niqobi: ko'pburchak mm -> distorsiyasiz piksel (newK) -> to'ldirish."""
    R = cv2.Rodrigues(np.array(meta["rvec"]))[0]
    t = np.array(meta["t"])
    poly = np.array(meta["varaq_kopburchak_mm"])
    p3 = R[:, :2] @ poly.T + t[:, None]                 # kamera koordinatalari
    uv = (cal["newK"] @ p3).T
    uv = uv[:, :2] / uv[:, 2:3]
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(uv * 16).astype(np.int32)], 255, lineType=cv2.LINE_8, shift=4)
    return m


def gt_ink_rectified(meta, cal, M, name, size=(OUT_W, OUT_H)):
    """Etalon matn niqobini tekislangan kadrga o'tkazish (teskari xaritalash)."""
    W, H = size
    ink = cv2.imread(str(GT_DIR / f"{name}_matn.png"), 0)
    th, tw = ink.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([xs, ys, np.ones_like(xs)], -1).reshape(-1, 3).astype(np.float64)
    q = pix @ np.linalg.inv(M).T
    q = q[:, :2] / q[:, 2:3]
    mm = _plane_from_undist_px(q, cal["newK"], meta)
    mx = (mm[:, 0] * 5 + tw / 2).reshape(H, W).astype(np.float32)
    my = (mm[:, 1] * 5 + th / 2).reshape(H, W).astype(np.float32)
    # matn ingichka: GT ni biroz kengaytirilgan emas, aniq (nearest) holatda olamiz
    return cv2.remap(ink, mx, my, cv2.INTER_LINEAR, borderValue=0) > 100


def confusion(rows):
    tp = sum(1 for r in rows if r["rad_etildi_haqiqiy"] and r["rad_etildi_bashorat"])
    tn = sum(1 for r in rows if not r["rad_etildi_haqiqiy"] and not r["rad_etildi_bashorat"])
    fp = sum(1 for r in rows if not r["rad_etildi_haqiqiy"] and r["rad_etildi_bashorat"])
    fn = sum(1 for r in rows if r["rad_etildi_haqiqiy"] and not r["rad_etildi_bashorat"])
    return tp, tn, fp, fn
