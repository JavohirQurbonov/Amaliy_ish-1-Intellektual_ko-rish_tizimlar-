# -*- coding: utf-8 -*-
"""
calibration.py - 1-bosqich: kamerani shaxmat taxtasi yordamida kalibrlash.

Nazariy asos (2-mavzu): pinhole model
        s*[u, v, 1]^T = K [R | t] [X, Y, Z, 1]^T,
        K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
va linzaning radial (k1,k2,k3) hamda tangensial (p1,p2) distorsiyasi:
        x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2)
        y_d = y (1 + k1 r^2 + k2 r^4 + k3 r^6) + p1 (r^2 + 2 y^2) + 2 p2 x y
Zhang usuli (cv2.calibrateCamera) K va distorsiya koeffitsientlarini
bir nechta shaxmat ko'rinishidan aniqlaydi.
"""
import glob
import cv2
import numpy as np
from config import *


def find_corners(gray):
    """Shaxmat taxtasining ichki burchaklarini topish va subpiksel aniqlikka keltirish."""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, BOARD_INNER, flags)
    if not ok:                                   # zaxira (SB) detektori
        ok, corners = cv2.findChessboardCornersSB(gray, BOARD_INNER)
    if ok:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    return ok, corners


def calibrate(pattern=None, save_vis=True):
    """
    Barcha kalibrlash tasvirlaridan kamera parametrlarini aniqlaydi.

    Qaytaradi: dict(K, dist, rms, per_image_err, size, n_used, newK, roi)
    """
    pattern = pattern or str(CALIB_DIR / "*.jpg")
    files = sorted(glob.glob(pattern))
    if len(files) < 10:
        raise RuntimeError(f"Kalibrlash uchun kamida 10 ta tasvir kerak, topildi: {len(files)}")

    # 3D nuqtalar: z = 0 tekislikdagi to'r (mm)
    objp = np.zeros((BOARD_INNER[0] * BOARD_INNER[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_INNER[0], 0:BOARD_INNER[1]].T.reshape(-1, 2) * SQUARE_MM

    obj_pts, img_pts, used = [], [], []
    size = None
    for f in files:
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]                          # (w, h)
        ok, corners = find_corners(gray)
        if ok:
            obj_pts.append(objp); img_pts.append(corners); used.append(f)
            if save_vis and len(used) <= 3:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, BOARD_INNER, corners, ok)
                FIG_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(FIG_DIR / f"kalib_burchaklar_{len(used)}.jpg"), vis)
        else:
            print(f"  [!] burchak topilmadi: {f}")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)

    # Har bir tasvir bo'yicha qayta proyeksiya xatosi (RMS, piksel)
    errs = []
    for op, ip, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        errs.append(float(np.sqrt(np.mean(np.sum((ip - proj) ** 2, axis=2)))))

    # alpha=1: barcha piksellar saqlanadi (varaq burchaklari kadrdan chiqib ketmasligi uchun)
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, size, 1.0, size)
    return dict(K=K, dist=dist, rms=float(rms), per_image_err=errs, size=size,
                n_used=len(used), n_total=len(files), newK=newK, roi=roi,
                files=used, rvecs=rvecs, tvecs=tvecs, obj_pts=obj_pts, img_pts=img_pts)


def build_undistort_maps(cal):
    """Distorsiyani bartaraf etish uchun remap xaritalari va yaroqli (qora bo'lmagan) soha niqobi."""
    w, h = cal["size"]
    map1, map2 = cv2.initUndistortRectifyMap(cal["K"], cal["dist"], None, cal["newK"], (w, h), cv2.CV_32FC1)
    valid = cv2.remap(np.full((h, w), 255, np.uint8), map1, map2, cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8))    # chegara artefaktlaridan saqlanish
    return map1, map2, valid


def undistort(img, maps):
    """Distorsiyani bartaraf etish (2-mavzu: radial va tangensial)."""
    map1, map2, _ = maps
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)


if __name__ == "__main__":
    c = calibrate()
    print("RMS:", c["rms"]); print("K=\n", c["K"]); print("dist=", c["dist"].ravel())


def line_straightness(cal):
    """
    Kalibrlash sifatining mustaqil mezoni: shaxmat taxtasidagi qatorlar (to'g'ri chiziqlar)
    distorsiyani bartaraf etishdan oldin va keyin to'g'ri chiziqdan qanchalik og'ishi (RMS, px).
    """
    nx, ny = BOARD_INNER
    before, after = [], []
    for ip in cal["img_pts"]:
        pts = ip.reshape(-1, 2)
        und = cv2.undistortPoints(ip.reshape(-1, 1, 2), cal["K"], cal["dist"], P=cal["newK"]).reshape(-1, 2)
        for src, store in ((pts, before), (und, after)):
            g = src.reshape(ny, nx, 2)
            for row in g:                                  # har bir qator - 9 ta kollinear nuqta
                c = row.mean(0)
                u, s, vt = np.linalg.svd(row - c)          # eng yaxshi to'g'ri chiziq (TLS)
                d = (row - c) @ vt[1]                      # chiziqqa perpendikulyar og'ish
                store.append(float(np.sqrt(np.mean(d ** 2))))
    return float(np.mean(before)), float(np.mean(after))
