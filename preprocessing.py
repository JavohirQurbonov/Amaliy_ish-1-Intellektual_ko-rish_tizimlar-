# -*- coding: utf-8 -*-
"""
preprocessing.py - 2-bosqich: dastlabki ishlov berish quvuri (pipeline).

Tartib (3-mavzu):
  1) Kulrang tusga o'tkazish
  2) Shovqinni bostirish: Gauss filtri (chiziqli) + bilateral filtr (nochiziqli, chekkani saqlaydi)
  3) Kontrastni yaxshilash: CLAHE (taqqoslash uchun: oddiy gistogramma tenglashtirish)
  4) Canny chekka aniqlash (bo'sag'alar Otsu usuli bilan avtomatik tanlanadi)
  5) Harris javob xaritasi va Shi-Tomasi burchak nuqtalari
  6) Varaqning 4 burchagini topish: chekka xaritasi -> qavariq qobiq -> 4 ta
     eng uzun tomon -> to'g'ri chiziqlar kesishmasi -> Shi-Tomasi bilan aniqlashtirish
"""
import cv2
import numpy as np


# ---------------------------------------------------------------- 1-3: shovqin va kontrast
def denoise_and_enhance(bgr):
    """Shovqinni bostirish va kontrastni oshirish. Oraliq natijalarni qaytaradi."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (5, 5), 1.4)                    # chiziqli filtr
    bilat = cv2.bilateralFilter(gauss, d=9, sigmaColor=40, sigmaSpace=7)   # nochiziqli filtr
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(bilat)
    heq = cv2.equalizeHist(bilat)                                  # taqqoslash uchun
    return dict(gray=gray, gauss=gauss, bilat=bilat, clahe=clahe, heq=heq)


# ---------------------------------------------------------------- 4: Canny
def canny_auto(gray, valid=None):
    """Canny: yuqori bo'sag'a = Otsu bo'sag'asi, past bo'sag'a = 0.5 * yuqori."""
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hi, lo = float(t), 0.5 * float(t)
    edges = cv2.Canny(gray, lo, hi, L2gradient=True)
    if valid is not None:                          # undistortsiyadan qolgan qora chegarani tashlab yuborish
        edges[valid == 0] = 0
    return edges, (lo, hi)


# ---------------------------------------------------------------- 5: burchaklar
def harris_response(gray, block=3, ksize=3, k=0.04):
    """Harris burchak javobi R = det(M) - k*trace(M)^2 (vizualizatsiya uchun)."""
    r = cv2.cornerHarris(np.float32(gray), block, ksize, k)
    return r


def shi_tomasi(gray, mask=None, max_corners=50, quality=0.01, min_dist=10, block=7):
    """Shi-Tomasi: M matritsaning kichik xos qiymati bo'yicha eng yaxshi burchaklar."""
    pts = cv2.goodFeaturesToTrack(gray, max_corners, quality, min_dist, mask=mask, blockSize=block)
    return np.zeros((0, 2), np.float32) if pts is None else pts.reshape(-1, 2)


# ---------------------------------------------------------------- 6: 4 burchak
def _line_through(p, q):
    return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])


def _intersect(l1, l2):
    x = np.cross(l1, l2)
    if abs(x[2]) < 1e-9:
        return None
    return x[:2] / x[2]


def order_points(pts):
    """Burchaklarni tartiblash: yuqori-chap, yuqori-o'ng, pastki-o'ng, pastki-chap."""
    pts = np.asarray(pts, np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def _fit_side(edges, p, q):
    """p-q kesma atrofidagi chekka piksellariga mustahkam (Huber) to'g'ri chiziq o'tkazish."""
    ys, xs = np.nonzero(edges)
    pts = np.c_[xs, ys].astype(np.float32)
    d = np.asarray(q, float) - np.asarray(p, float)
    L = np.linalg.norm(d)
    u = d / L
    rel = pts - np.asarray(p, np.float32)
    along = rel @ u
    perp = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
    sel = (along > 0.08 * L) & (along < 0.92 * L) & (perp < 6)
    if sel.sum() < 20:
        return _line_through(p, q)
    vx, vy, x0, y0 = cv2.fitLine(pts[sel], cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return _line_through((x0, y0), (x0 + vx, y0 + vy))


def find_document_quad(edges, gray_blur, img_shape):
    """
    Chekka xaritasidan varaqning 4 burchagini topadi.
    Qaytaradi: (burchaklar[4,2], usul_nomi, shitomasi_tuzatilgan_soni)
    """
    h, w = img_shape[:2]
    e = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(cv2.convexHull(c)) > 0.03 * w * h]
    if not cnts:
        return None, "topilmadi", 0
    cnt = max(cnts, key=lambda c: cv2.contourArea(cv2.convexHull(c)))
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    poly = cv2.approxPolyDP(hull, 0.006 * peri, True)[:, 0, :].astype(np.float64)

    method = "tomonlar kesishmasi"
    if len(poly) < 4:
        box = cv2.boxPoints(cv2.minAreaRect(hull)); return order_points(box), "minAreaRect", 0
    n = len(poly)
    lens = np.array([np.linalg.norm(poly[(i + 1) % n] - poly[i]) for i in range(n)])
    idx = sorted(np.argsort(lens)[-4:])                     # 4 ta eng uzun tomon (tsiklik tartibda)
    sides = [(poly[i], poly[(i + 1) % n]) for i in idx]
    lines = [_fit_side(edges, p, q) for p, q in sides]      # har bir tomon uchun aniqlashtirilgan chiziq
    corners = []
    for i in range(4):
        c = _intersect(lines[i], lines[(i + 1) % 4])
        if c is None:
            return order_points(cv2.boxPoints(cv2.minAreaRect(hull))), "minAreaRect", 0
        corners.append(c)
    corners = np.array(corners, np.float32)

    # Shi-Tomasi bilan aniqlashtirish: nomzod burchak atrofida (r=30 px) haqiqiy burchak qidiriladi
    refined = 0
    for i, c in enumerate(corners):
        m = np.zeros((h, w), np.uint8)
        cv2.circle(m, tuple(int(v) for v in c), 30, 255, -1)
        cand = shi_tomasi(gray_blur, m)
        if len(cand):
            j = np.argmin(np.linalg.norm(cand - c, axis=1))
            if np.linalg.norm(cand[j] - c) < 20:
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                pt = cv2.cornerSubPix(gray_blur, cand[j].reshape(1, 1, 2).astype(np.float32), (9, 9), (-1, -1), crit)
                corners[i] = pt.reshape(2)
                refined += 1
    return order_points(corners), method, refined


def draw_corners(bgr, corners, edges=None):
    """Vizualizatsiya: topilgan 4 burchak va konturni chizish."""
    vis = bgr.copy()
    if corners is None:
        return vis
    c = corners.astype(np.int32)
    cv2.polylines(vis, [c.reshape(-1, 1, 2)], True, (0, 255, 0), 3, cv2.LINE_AA)
    for i, p in enumerate(c):
        cv2.circle(vis, tuple(int(v) for v in p), 12, (0, 0, 255), -1)
        cv2.putText(vis, ["TL", "TR", "BR", "BL"][i], (p[0] + 14, p[1] - 14), 0, 1.2, (0, 255, 255), 3)
    return vis
