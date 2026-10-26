# -*- coding: utf-8 -*-
"""
perspective.py - 3-bosqich: to'rt nuqtali perspektiv almashtirish (four-point transform).

Topilgan 4 burchak (tl, tr, br, bl) to'g'ri to'rtburchak burchaklariga
(0,0), (W-1,0), (W-1,H-1), (0,H-1) o'tkaziladi. Gomografiya M (3x3) 4 juft
nuqtadan DLT usulida (cv2.getPerspectiveTransform) topiladi:
      [x', y', w']^T = M [x, y, 1]^T ;  (u, v) = (x'/w', y'/w')
"""
import cv2
import numpy as np
from config import OUT_W, OUT_H


def four_point_transform(img, corners, size=(OUT_W, OUT_H), border=cv2.BORDER_REPLICATE):
    """Qaytaradi: (tekislangan_tasvir, M)."""
    W, H = size
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst)
    warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=border)
    return warped, M


def warp_mask(mask, M, size=(OUT_W, OUT_H)):
    """Ikkilik niqobni xuddi shu M bilan o'tkazish (tashqarisi = 0)."""
    W, H = size
    w = cv2.warpPerspective(mask, M, (W, H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return ((w > 127) * 255).astype(np.uint8)
