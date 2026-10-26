# -*- coding: utf-8 -*-
"""
quality_control.py - 5-bosqich: qoidaviy (rule-based) sifat nazorati.

Kirish: tekislangan tasvir va tekislangan varaq niqobi (255 = varaq).
Mezonlar:
  1) Burchak yaxlitligi  - har bir burchak oynasining varaq bilan to'lish ulushi
  2) Burchak to'g'riligi - tomonlarga o'tkazilgan chiziqlar orasidagi burchakning 90 gradusdan og'ishi
  3) Chet tekisligi      - chet profilining to'g'ri chiziqdan RMS va (p95-p5) og'ishi
  4) Soya                - yoritilganlik xaritasi (morfologik yopish) bo'yicha qorong'i soha ulushi
  5) Dog'                - rang (xromatiklik) bo'yicha farq qiluvchi yoki lokal fondan qoraygan yirik dog'simon sohalar
Natija: "Qabul qilindi" yoki "Rad etildi: <sabablar>".
"""
import cv2
import numpy as np
from config import QC

CORNER_NAMES = ["yuqori chap", "yuqori o'ng", "pastki o'ng", "pastki chap"]


def _edge_profiles(mask):
    """To'rt tomon uchun chetdan ichkariga birinchi varaq pikseligacha masofa profili."""
    H, W = mask.shape
    m, b = QC["edge_margin"], QC["edge_band"]
    top = np.array([np.argmax(mask[:b, x] > 0) if (mask[:b, x] > 0).any() else b for x in range(m, W - m)], float)
    bot = np.array([np.argmax(mask[::-1][:b, x] > 0) if (mask[::-1][:b, x] > 0).any() else b for x in range(m, W - m)], float)
    lef = np.array([np.argmax(mask[y, :b] > 0) if (mask[y, :b] > 0).any() else b for y in range(m, H - m)], float)
    rig = np.array([np.argmax(mask[y, ::-1][:b] > 0) if (mask[y, ::-1][:b] > 0).any() else b for y in range(m, H - m)], float)
    return dict(top=(np.arange(m, W - m), top), bottom=(np.arange(m, W - m), bot),
                left=(np.arange(m, H - m), lef), right=(np.arange(m, H - m), rig))


def check_edges_and_angles(mask):
    """Chet tekisligi va burchak to'g'riligi. Qaytaradi (metrikalar, sabablar)."""
    H, W = mask.shape
    prof = _edge_profiles(mask)
    fits, rms, rng_, dirs = {}, {}, {}, {}
    for side, (s, p) in prof.items():
        a, c = np.polyfit(s, p, 1)
        res = p - (a * s + c)
        rms[side] = float(np.sqrt(np.mean(res ** 2)))
        rng_[side] = float(np.percentile(res, 95) - np.percentile(res, 5))
        # tomonning haqiqiy yo'nalishi (tasvir koordinatalarida)
        dirs[side] = np.array([1.0, a]) if side in ("top", "bottom") else np.array([a, 1.0])
    ang = {}
    for name, (s1, s2) in {"yuqori chap": ("top", "left"), "yuqori o'ng": ("top", "right"),
                           "pastki o'ng": ("bottom", "right"), "pastki chap": ("bottom", "left")}.items():
        u, v = dirs[s1], dirs[s2]
        cosang = abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
        ang[name] = float(np.degrees(np.arccos(np.clip(cosang, 0, 1))))
    reasons = []
    side_uz = dict(top="yuqori chet", bottom="pastki chet", left="chap chet", right="o'ng chet")
    for side in prof:
        if rms[side] > QC["edge_rms_max"] or rng_[side] > QC["edge_range_max"]:
            reasons.append(("chet", f"{side_uz[side]}i notekis (RMS={rms[side]:.1f} px)"))
    for name, a in ang.items():
        if abs(90 - a) > QC["angle_dev_max"]:
            reasons.append(("burchak", f"{name} burchak noto'g'ri ({a:.1f} deg)"))
    return dict(edge_rms=rms, edge_range=rng_, angles=ang), reasons


def check_corners(mask):
    """Burchak yaxlitligi: burchak oynasining varaq bilan to'lish ulushi."""
    H, W = mask.shape
    p = QC["corner_patch"]
    patches = [mask[:p, :p], mask[:p, W - p:], mask[H - p:, W - p:], mask[H - p:, :p]]
    fills = [float((q > 0).mean()) for q in patches]
    reasons = [("burchak", f"{CORNER_NAMES[i]} burchakda buklanish/yirtilish aniqlandi (to'lish={f:.2f})")
               for i, f in enumerate(fills) if f < QC["corner_fill_min"]]
    return dict(corner_fill=fills), reasons


def illumination_and_stains(rect_bgr, mask):
    """Soya va dog' tahlili. Qaytaradi (metrikalar, sabablar, soya_xaritasi, dog'_niqobi)."""
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    inner = cv2.erode(mask, np.ones((31, 31), np.uint8)) > 0
    if inner.sum() < 1000:
        return dict(shadow_frac=np.nan, stain_area=np.nan), [("chet", "varaq niqobi juda kichik")], None, None
    paper = float(np.percentile(gray[inner], 90))
    g2 = gray.copy()
    g2[~(mask > 0)] = np.uint8(paper)
    # yoritilganlik xaritasi: katta (251 px) morfologik yopish matn va dog'larni yo'qotadi
    bg = cv2.morphologyEx(g2, cv2.MORPH_CLOSE, np.ones((251, 251), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), 25).astype(np.float32)
    ref = float(np.percentile(bg[inner], 90))
    shadow = (bg < QC["shadow_ratio"] * ref) & inner
    shadow_frac = float(shadow.sum() / inner.sum())

    # ---- dog'lar. Ikki mezon:
    #  (a) XROMATIKLIK: dog' varaq rangidan farq qiladi, soya esa faqat yorqinlikni o'zgartiradi
    #      (xromatiklik c = BGR / (B+G+R) soyada deyarli o'zgarmaydi) -> soyaga chidamli;
    #  (b) QORAYISH: lokal fonga nisbatan qoraygan soha (rangsiz dog'lar uchun). Soya mavjud
    #      bo'lganda yoritilganlik xaritasi ishonchsiz bo'lgani uchun bu mezon o'chiriladi.
    f = cv2.GaussianBlur(rect_bgr, (0, 0), 3).astype(np.float32)
    chroma = f / np.maximum(f.sum(axis=2, keepdims=True), 1)
    c_med = np.median(chroma[inner], axis=0)
    dev = np.linalg.norm(chroma - c_med, axis=2)
    cand = (dev > QC["stain_chroma_dev"]) & inner
    if shadow_frac <= QC["shadow_area_max"]:
        norm = gray.astype(np.float32) / np.maximum(bg, 1)
        cand |= (norm < QC["stain_dark_ratio"]) & inner
    dark = np.uint8(cand) * 255
    # yupqa chiziqlar (matn, jadval) va zich harflar guruhi opening bilan yo'qoladi
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (QC["stain_open"], QC["stain_open"])))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    stain_mask = np.zeros_like(dark)
    max_area = 0
    for k in range(1, n):
        a = stats[k, cv2.CC_STAT_AREA]
        if a >= QC["stain_area_min"]:
            stain_mask[lab == k] = 255
            max_area = max(max_area, a)
    reasons = []
    if shadow_frac > QC["shadow_area_max"]:
        reasons.append(("soya", f"soya aniqlandi (varaq maydonining {100 * shadow_frac:.0f}%)"))
    if max_area > 0:
        reasons.append(("dog'", f"dog' aniqlandi (maydon={int(max_area)} px)"))
    return dict(shadow_frac=shadow_frac, stain_area=float(max_area)), reasons, (shadow.astype(np.uint8) * 255), stain_mask


def quality_control(rect_bgr, mask):
    """To'liq sifat nazorati. Qaytaradi dict(verdict, accepted, types, reasons, metrics, shadow, stain)."""
    metrics, reasons = {}, []
    m1, r1 = check_corners(mask); metrics.update(m1); reasons += r1
    m2, r2 = check_edges_and_angles(mask); metrics.update(m2); reasons += r2
    m3, r3, shadow, stain = illumination_and_stains(rect_bgr, mask); metrics.update(m3); reasons += r3
    types = sorted({t for t, _ in reasons})
    texts = [s for _, s in reasons]
    if reasons:
        verdict = "Rad etildi: " + "; ".join(texts)
    else:
        verdict = "Qabul qilindi"
    return dict(verdict=verdict, accepted=not reasons, types=types, reasons=texts,
                metrics=metrics, shadow=shadow, stain=stain)
