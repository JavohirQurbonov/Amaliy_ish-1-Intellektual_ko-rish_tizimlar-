# -*- coding: utf-8 -*-
"""
segmentation.py - 4-bosqich: segmentatsiya (4-mavzu).

A) Varaqni fondan ajratish (distorsiyasi to'g'irlangan tasvirda):
     Otsu global bo'sag'asi -> ishonchli old/orqa fon markerlari -> watershed.
B) Matn (siyoh) niqobi (tekislangan tasvirda):
     moslashuvchan (adaptive Gauss) bo'sag'a - notekis yoritilganlikka chidamli.
"""
import cv2
import numpy as np


def _largest_component_filled(binm):
    """Eng katta bog'langan komponentani qoldirib, ichki teshiklarini (dog', matn) to'ldirish."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    if n <= 1:
        return binm
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    comp = np.uint8(lab == k) * 255
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
    return filled


def segment_sheet(bgr, valid):
    """
    Varaq niqobi (255 = varaq). Otsu + watershed.
    Qaytaradi: (niqob, otsu_bo'sag'a, otsu_niqobi)
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    t, _ = cv2.threshold(blur[valid > 0].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = np.uint8((blur > t) & (valid > 0)) * 255
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    base = _largest_component_filled(otsu)

    ell = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    sure_fg = cv2.erode(base, ell(17))
    sure_bg = cv2.bitwise_not(cv2.dilate(base, ell(25)))
    markers = np.zeros(gray.shape, np.int32)
    markers[sure_bg > 0] = 1
    markers[valid == 0] = 1
    markers[sure_fg > 0] = 2
    cv2.watershed(bgr, markers)                  # chegarani gradient bo'yicha aniqlashtiradi
    mask = np.uint8(markers == 2) * 255
    mask = _largest_component_filled(mask)
    return mask, float(t), otsu


def segment_ink(rectified_bgr, sheet_mask=None, block=51, C=12):
    """Matn niqobi: adaptive Gauss bo'sag'a. 255 = siyoh."""
    gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, C)
    if sheet_mask is not None:
        ink = cv2.bitwise_and(ink, cv2.erode(sheet_mask, np.ones((9, 9), np.uint8)))
    return ink


def region_growing(gray, seed, tol=12, max_pix=3_000_000):
    """
    Oddiy region growing (qo'shimcha usul): urug'dan boshlab, o'rtacha yorqinlikdan
    +-tol dan oshmaydigan 4-qo'shni piksellarni qo'shib boradi. Taqqoslash uchun.
    """
    h, w = gray.shape
    seg = np.zeros((h, w), np.uint8)
    seg[seed[1], seed[0]] = 255
    stack = [seed]
    mean, cnt = float(gray[seed[1], seed[0]]), 1
    while stack and cnt < max_pix:
        x, y = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not seg[ny, nx] and abs(float(gray[ny, nx]) - mean) <= tol:
                seg[ny, nx] = 255
                mean = (mean * cnt + gray[ny, nx]) / (cnt + 1); cnt += 1
                stack.append((nx, ny))
    return seg
