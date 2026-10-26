# -*- coding: utf-8 -*-
"""
synth_data.py - sintetik (dasturiy) test to'plamini hosil qilish moduli.

Nima uchun kerak?
  Haqiqiy kamera surati olish imkoni bo'lmagan sharoitda butun quvurni
  (pipeline) ob'ektiv baholash uchun ETALON (ground truth) ma'lum bo'lgan
  tasvirlar zarur: varaq burchaklari, varaq niqobi, matn niqobi va nuqson turi.
  Shu sababli tasvirlar pinhole kamera + radial/tangensial distorsiya
  modeli bo'yicha dasturiy ravishda sintez qilinadi:

     tekislik (stol + varaq)  --(poza R,t)-->  pinhole  --(distorsiya)-->  piksel

  Chiqish:
    data/kalibrlash/kalib_XX.jpg   - 20 ta shaxmat taxtasi (9x6) tasviri
    data/hujjatlar/hujjat_XX.jpg   - 24 ta hujjat tasviri (12 sifatli, 12 nuqsonli)
    data/etalon/hujjat_XX.json     - etalon: poza, varaq ko'pburchagi, nuqson turi
    data/etalon/hujjat_XX_matn.png - etalon matn (siyoh) niqobi
    data/etalon/kamera_haqiqiy.json- kameraning HAQIQIY parametrlari (tekshiruv uchun)

  O'z (haqiqiy) tasvirlaringiz bo'lsa, ularni shu papkalarga qo'ysangiz bas:
  qolgan modullar o'zgarishsiz ishlayveradi (etalon bo'lmasa IoU hisoblanmaydi).
"""
import json
import cv2
import numpy as np
from config import *

# ----------------------------------------------------- Kameraning haqiqiy modeli
IMG_W, IMG_H = 1600, 1200
K_TRUE = np.array([[1400.0, 0, 805.0], [0, 1400.0, 597.0], [0, 0, 1]])
DIST_TRUE = np.array([-0.28, 0.11, 0.0012, -0.0009, 0.0])   # k1,k2,p1,p2,k3

PX_MM = 5                      # tekislik teksturasi: 1 mm = 5 piksel
DOC_W, DOC_H = 1050, 1486      # A4 varaq teksturasi (210 x 297.2 mm)
TAB = 4000                     # stol teksturasi (800 x 800 mm)

WORDS = ("hisobot mahsulot narx miqdor sana raqam shartnoma tasvir kamera "
         "burchak chegara kontrast filtr natija tahlil tizim aniqlik "
         "namuna jadval xulosa ko'rish bosqich parametr").split()


# ------------------------------------------------------------ Geometriya
def rot(rx, ry, rz):
    """Eyler burchaklari (gradus) bo'yicha aylanish matritsasi."""
    Rx = cv2.Rodrigues(np.array([np.deg2rad(rx), 0, 0]))[0]
    Ry = cv2.Rodrigues(np.array([0, np.deg2rad(ry), 0]))[0]
    Rz = cv2.Rodrigues(np.array([0, 0, np.deg2rad(rz)]))[0]
    return Rz @ Ry @ Rx


def project(pts_mm, R, t):
    """Tekislik nuqtalarini (mm, z=0) distorsiyali piksel koordinatalariga o'tkazish."""
    p3 = np.c_[pts_mm, np.zeros(len(pts_mm))].astype(np.float64)
    rvec = cv2.Rodrigues(R)[0]
    px, _ = cv2.projectPoints(p3, rvec, t.astype(np.float64), K_TRUE, DIST_TRUE)
    return px.reshape(-1, 2)


_UND_CACHE = {}


def undistorted_grid():
    """Har bir distorsiyali piksel uchun distorsiyasiz piksel koordinatasi (keshlanadi)."""
    if "g" not in _UND_CACHE:
        xs, ys = np.meshgrid(np.arange(IMG_W), np.arange(IMG_H))
        pts = np.stack([xs, ys], -1).reshape(-1, 1, 2).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-10)
        # und = cv2.undistortPointsIter(pts, K_TRUE, DIST_TRUE, None, K_TRUE, crit)
        und = cv2.undistortPoints(
            pts, K_TRUE, DIST_TRUE, P=K_TRUE
        )
        _UND_CACHE["g"] = und.reshape(-1, 2).astype(np.float64)
    return _UND_CACHE["g"]


def plane_maps(R, t, tex_w, tex_h):
    """Kamera tasvirining har bir pikseli uchun tekstura koordinatasi (remap xaritalari)."""
    Hm = K_TRUE @ np.column_stack([R[:, 0], R[:, 1], t])
    und = undistorted_grid()
    ph = np.c_[und, np.ones(len(und))] @ np.linalg.inv(Hm).T
    X, Y = ph[:, 0] / ph[:, 2], ph[:, 1] / ph[:, 2]
    mx = (X * PX_MM + tex_w / 2).reshape(IMG_H, IMG_W).astype(np.float32)
    my = (Y * PX_MM + tex_h / 2).reshape(IMG_H, IMG_W).astype(np.float32)
    return mx, my


# ------------------------------------------------------------ Teksturalar
def table_texture(rng):
    """Qorong'i, tekis yog'och stol: past kontrastli shovqin va tolalar."""
    base = np.array([62, 74, 92], np.float32) + rng.uniform(-8, 8, 3)
    low = cv2.resize(rng.normal(0, 1, (TAB // 100, TAB // 100)).astype(np.float32),
                     (TAB, TAB), interpolation=cv2.INTER_CUBIC)
    grain = cv2.GaussianBlur(rng.normal(0, 1, (TAB // 2, TAB // 2)).astype(np.float32),
                             (0, 0), sigmaX=40, sigmaY=1.5)
    grain = cv2.resize(grain, (TAB, TAB))
    grain = grain / (grain.std() + 1e-6)
    tex = base[None, None, :] + 4.0 * low[..., None] + 2.0 * grain[..., None]
    return np.clip(tex, 0, 255).astype(np.uint8)


def doc_texture(rng, idx, stain=False, fold=None):
    """A4 varaq: sarlavha, jadval, matn, imzo va muhr. Qaytaradi (rasm, matn_niqobi)."""
    paper = np.full((DOC_H, DOC_W, 3), (238, 240, 243), np.uint8)
    paper = np.clip(paper + rng.normal(0, 1.5, paper.shape), 0, 255).astype(np.uint8)
    ink = np.zeros((DOC_H, DOC_W), np.uint8)
    col = (60, 40, 25)          # to'q ko'k siyoh (BGR)

    def put(txt, org, scale, th, font=cv2.FONT_HERSHEY_SIMPLEX):
        cv2.putText(paper, txt, org, font, scale, col, th, cv2.LINE_AA)
        cv2.putText(ink, txt, org, font, scale, 255, th, cv2.LINE_AA)

    def line(p1, p2, th):
        cv2.line(paper, p1, p2, col, th, cv2.LINE_AA)
        cv2.line(ink, p1, p2, 255, th, cv2.LINE_AA)

    put(f"HISOBOT No {100 + idx}", (170, 250), 2.0, 5, cv2.FONT_HERSHEY_DUPLEX)
    put("Intellektual ko'rish tizimlari", (170, 330), 1.1, 3)
    put("Sana: 19.10.2026", (170, 395), 1.0, 3)
    # jadval: 5 qator x 3 ustun
    x0, x1, y0, rh = 170, 880, 450, 80
    xs = [x0, x0 + 300, x0 + 520, x1]
    for r in range(6):
        line((x0, y0 + r * rh), (x1, y0 + r * rh), 3)
    for x in xs:
        line((x, y0), (x, y0 + 5 * rh), 3)
    for r in range(5):
        put(rng.choice(WORDS), (x0 + 15, y0 + r * rh + 55), 1.0, 3)
        put(str(rng.integers(10, 999)), (xs[1] + 15, y0 + r * rh + 55), 1.0, 3)
        put(f"{rng.integers(1, 99)}.{rng.integers(10, 99)}", (xs[2] + 15, y0 + r * rh + 55), 1.0, 3)
    # matn qatorlari
    for r in range(7):
        txt = " ".join(rng.choice(WORDS, size=int(rng.integers(4, 6))))
        put(txt, (170, 960 + r * 55), 1.0, 3)
    # imzo va muhr
    pts = np.array([[170 + i * 9, 1385 + int(25 * np.sin(i / 2.5) * rng.uniform(0.6, 1))]
                    for i in range(22)], np.int32)
    cv2.polylines(paper, [pts], False, col, 4, cv2.LINE_AA)
    cv2.polylines(ink, [pts], False, 255, 4, cv2.LINE_AA)
    cv2.circle(paper, (850, 1330), 70, col, 4, cv2.LINE_AA)
    cv2.circle(ink, (850, 1330), 70, 255, 4, cv2.LINE_AA)
    put("MUHR", (798, 1345), 1.0, 3)

    if stain:                     # qahva dog'i
        cx = int(rng.integers(330, 700)); cy = int(rng.integers(560, 1150))
        r0 = rng.uniform(55, 85)
        ang = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        rad = r0 * (1 + 0.18 * np.sin(3 * ang + rng.uniform(0, 6)) + rng.uniform(-0.05, 0.05, 60))
        poly = np.c_[cx + rad * np.cos(ang), cy + rad * np.sin(ang)].astype(np.int32)
        m = np.zeros((DOC_H, DOC_W), np.float32)
        cv2.fillPoly(m, [poly], 1.0)
        ring = cv2.GaussianBlur(m, (0, 0), 3) - cv2.GaussianBlur(m, (0, 0), 14)
        m = cv2.GaussianBlur(m, (0, 0), 5)
        brown = np.array([55, 95, 140], np.float32)
        paper = (paper * (1 - 0.82 * m[..., None]) + brown * 0.82 * m[..., None])
        paper = np.clip(paper * (1 - 0.25 * np.clip(ring, 0, 1)[..., None]), 0, 255).astype(np.uint8)

    if fold is not None:          # buklangan burchak: (burchak_nomi, yon uzunligi)
        corner, L = fold
        flap_col = np.array((200, 203, 206), np.float32)
        tri = {"tl": [(0, L), (L, 0), (L, L)], "tr": [(DOC_W - L, 0), (DOC_W, L), (DOC_W - L, L)],
               "br": [(DOC_W, DOC_H - L), (DOC_W - L, DOC_H), (DOC_W - L, DOC_H - L)],
               "bl": [(0, DOC_H - L), (L, DOC_H), (L, DOC_H - L)]}[corner]
        m = np.zeros((DOC_H, DOC_W), np.float32)
        cv2.fillPoly(m, [np.array(tri, np.int32)], 1.0)
        m = cv2.GaussianBlur(m, (0, 0), 1.2)
        paper = (paper * (1 - m[..., None]) + flap_col * m[..., None]).astype(np.uint8)
    return paper, ink


def doc_polygon(rng, defect, L=None):
    """Varaq silueti (varaq teksturasi pikselida). Nuqsonga qarab o'zgaradi."""
    W, H = DOC_W, DOC_H
    if defect == "burchak":      # buklangan burchak -> siluet beshburchak
        corner = rng.choice(["tl", "tr", "br", "bl"])
        poly = {"tl": [(L, 0), (W, 0), (W, H), (0, H), (0, L)],
                "tr": [(0, 0), (W - L, 0), (W, L), (W, H), (0, H)],
                "br": [(0, 0), (W, 0), (W, H - L), (W - L, H), (0, H)],
                "bl": [(0, 0), (W, 0), (W, H), (L, H), (0, H - L)]}[str(corner)]
        return np.array(poly, np.float64), str(corner)
    if defect == "chet":         # yirtilgan notekis chet
        side = str(rng.choice(["top", "right", "bottom", "left"]))
        n = 80
        amp = rng.uniform(25, 40)
        smooth = np.convolve(rng.normal(0, 1, n + 20), np.ones(9) / 9, "same")[10:-10]
        off = np.abs(smooth / (np.abs(smooth).max() + 1e-6)) * amp + rng.uniform(2, 6, n)
        s = np.linspace(0, 1, n)
        if side == "top":
            edge = np.c_[s * W, off]; poly = [(0, H), (0, 0)] + edge.tolist() + [(W, 0), (W, H)]
            poly = [(W, H), (0, H), (0, 0)] + edge.tolist() + [(W, 0)]
        elif side == "bottom":
            edge = np.c_[s * W, H - off]
            poly = [(0, 0), (W, 0), (W, H)] + edge[::-1].tolist() + [(0, H)]
        elif side == "left":
            edge = np.c_[off, s * H]
            poly = [(W, 0), (W, H), (0, H)] + edge[::-1].tolist() + [(0, 0)]
        else:
            edge = np.c_[W - off, s * H]
            poly = [(0, 0), (W, 0)] + edge.tolist() + [(W, H), (0, H)]
        return np.array(poly, np.float64), side
    return np.array([(0, 0), (W, 0), (W, H), (0, H)], np.float64), ""


# ------------------------------------------------------------ Tasvir effektlari
def finish_image(img, rng, light):
    """Yoritilganlik sharoiti, shovqin, xiralik: 'telefon kamerasi' effekti."""
    f = img.astype(np.float32)
    ys, xs = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float32)
    xn, yn = (xs - IMG_W / 2) / (IMG_W / 2), (ys - IMG_H / 2) / (IMG_H / 2)
    vign = 1 - 0.07 * (xn ** 2 + yn ** 2)
    if light == "normal":
        gain = 1.0 * vign
    elif light == "dim":                      # xira yoritilganlik
        gain = 0.52 * vign
    elif light == "sidelight":                # yon tomondan yorug'lik (gradient)
        a = rng.uniform(0, 2 * np.pi)
        gain = (1 + 0.08 * (np.cos(a) * xn + np.sin(a) * yn)) * vign
    else:                                     # iliq (sariq) yorug'lik
        gain = vign
        f = f * np.array([0.82, 0.93, 1.05], np.float32)[None, None, :]
    f = f * gain[..., None]
    f = cv2.GaussianBlur(f, (0, 0), 0.7)
    f = f + rng.normal(0, 4.0, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def pose_ok(corners_px, margin=25):
    return bool(np.all((corners_px[:, 0] > margin) & (corners_px[:, 0] < IMG_W - margin) &
                       (corners_px[:, 1] > margin) & (corners_px[:, 1] < IMG_H - margin)))


# ------------------------------------------------------------ Kalibrlash to'plami
def make_calibration_set(n=20, seed=11):
    rng = np.random.default_rng(seed)
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    bw, bh = 10, 7                                   # katak soni
    S = int(SQUARE_MM * PX_MM)
    page_w, page_h = bw * S + 200, bh * S + 200
    page = np.full((page_h, page_w, 3), 235, np.uint8)
    for i in range(bh):
        for j in range(bw):
            if (i + j) % 2 == 0:
                page[100 + i * S:100 + (i + 1) * S, 100 + j * S:100 + (j + 1) * S] = 25
    # ichki burchaklar va tashqi kontur (mm, markazga nisbatan)
    gx, gy = np.meshgrid(np.arange(1, bw), np.arange(1, bh))
    inner = np.c_[(gx.ravel() - bw / 2) * SQUARE_MM, (gy.ravel() - bh / 2) * SQUARE_MM]
    outer = np.array([[-bw / 2 * SQUARE_MM - 4, -bh / 2 * SQUARE_MM - 4],
                      [bw / 2 * SQUARE_MM + 4, -bh / 2 * SQUARE_MM - 4],
                      [bw / 2 * SQUARE_MM + 4, bh / 2 * SQUARE_MM + 4],
                      [-bw / 2 * SQUARE_MM - 4, bh / 2 * SQUARE_MM + 4]])
    done, tries = 0, 0
    while done < n and tries < 2000:
        tries += 1
        R = rot(rng.uniform(-32, 32), rng.uniform(-32, 32), rng.uniform(-25, 25))
        t = np.array([rng.uniform(-50, 50), rng.uniform(-40, 40), rng.uniform(430, 600)])
        if not (pose_ok(project(inner, R, t), 35) and pose_ok(project(outer, R, t), 20)):
            continue
        tab = table_texture(rng)
        y0, x0 = TAB // 2 - page_h // 2, TAB // 2 - page_w // 2
        tab[y0:y0 + page_h, x0:x0 + page_w] = page
        mx, my = plane_maps(R, t, TAB, TAB)
        img = cv2.remap(cv2.GaussianBlur(tab, (0, 0), 0.6), mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
        img = finish_image(img, rng, rng.choice(["normal", "sidelight", "normal"]))
        cv2.imwrite(str(CALIB_DIR / f"kalib_{done + 1:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        done += 1
    print(f"[synth] kalibrlash tasvirlari: {done} ta")


# ------------------------------------------------------------ Hujjat to'plami
PLAN = (["yo'q"] * 12 + ["burchak"] * 3 + ["chet"] * 3 + ["dog'"] * 3 + ["soya"] * 3)
LIGHTS = ["normal", "dim", "sidelight", "warm"]


def make_document_set(seed=2026):
    rng = np.random.default_rng(seed)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    order = rng.permutation(len(PLAN))               # tartibni aralashtirish
    for k, pi in enumerate(order):
        defect = PLAN[pi]
        light = LIGHTS[k % 4]
        L = int(rng.integers(120, 200)) if defect == "burchak" else None
        poly, extra = doc_polygon(rng, "burchak" if defect == "burchak" else
                                  ("chet" if defect == "chet" else ""), L)
        fold = (extra, L) if defect == "burchak" else None
        paper, ink = doc_texture(rng, k + 1, stain=(defect == "dog'"), fold=fold)

        # stol + varaq (siluet bo'yicha)
        tab = table_texture(rng)
        ox, oy = TAB // 2 - DOC_W // 2, TAB // 2 - DOC_H // 2
        mask = np.zeros((TAB, TAB), np.uint8)
        cv2.fillPoly(mask, [np.round(poly + [ox, oy]).astype(np.int32)], 255)
        sh = cv2.GaussianBlur(np.roll(mask, (10, 10), (0, 1)).astype(np.float32) / 255, (0, 0), 14)
        tab = (tab * (1 - 0.35 * sh[..., None])).astype(np.uint8)       # varaq soyasi
        full = tab.copy()
        full[oy:oy + DOC_H, ox:ox + DOC_W] = np.where(
            (mask[oy:oy + DOC_H, ox:ox + DOC_W] > 0)[..., None], paper, tab[oy:oy + DOC_H, ox:ox + DOC_W])

        # kast soya (varaq va stol ustiga): yarim tekislik, yumshoq chegara
        if defect == "soya":
            ang = rng.uniform(0, 2 * np.pi)
            yy, xx = np.mgrid[0:TAB, 0:TAB].astype(np.float32)
            s = (xx - TAB / 2) * np.cos(ang) + (yy - TAB / 2) * np.sin(ang)
            sd = s[oy:oy + DOC_H, ox:ox + DOC_W]
            c = np.quantile(sd, rng.uniform(0.62, 0.78))
            depth = rng.uniform(0.32, 0.42)
            field = 1 - depth / (1 + np.exp(-(s - c) / 18.0))
            full = (full * field[..., None]).astype(np.uint8)

        # kamera pozasi
        for _ in range(500):
            R = rot(rng.uniform(-26, 26), rng.uniform(-26, 26), rng.uniform(-25, 25))
            t = np.array([rng.uniform(-40, 40), rng.uniform(-40, 40), rng.uniform(430, 520)])
            c_mm = np.array([[-105, -148.6], [105, -148.6], [105, 148.6], [-105, 148.6]])
            cpx = project(c_mm, R, t)
            if pose_ok(cpx, 30):
                break
        else:
            raise RuntimeError("mos poza topilmadi")
        mx, my = plane_maps(R, t, TAB, TAB)
        img = cv2.remap(cv2.GaussianBlur(full, (0, 0), 0.6), mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
        img = finish_image(img, rng, light)
        name = f"hujjat_{k + 1:02d}"
        cv2.imwrite(str(DOC_DIR / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        cv2.imwrite(str(GT_DIR / f"{name}_matn.png"), ink)
        meta = dict(id=name, nuqson=defect, yoritilganlik=light, rvec=cv2.Rodrigues(R)[0].ravel().tolist(),
                    t=t.tolist(),
                    varaq_kopburchak_mm=((poly - [DOC_W / 2, DOC_H / 2]) / PX_MM).tolist(),
                    burchaklar_mm=c_mm.tolist(), burchaklar_px_distorsiyali=cpx.tolist())
        (GT_DIR / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[synth] hujjat tasvirlari: {len(order)} ta")
    (GT_DIR / "kamera_haqiqiy.json").write_text(json.dumps(
        dict(K=K_TRUE.tolist(), dist=DIST_TRUE.tolist(), size=[IMG_W, IMG_H]), indent=1), encoding="utf-8")


def generate_all():
    make_calibration_set()
    make_document_set()


if __name__ == "__main__":
    generate_all()
