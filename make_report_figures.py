# -*- coding: utf-8 -*-
"""make_report_figures.py - hisobot uchun rasmlarni tayyorlash (main.py dan keyin ishga tushiriladi)."""
import json, glob
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
import calibration as cb
import main as mn

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
FIG_DIR.mkdir(parents=True, exist_ok=True)
cal = cb.calibrate(save_vis=False)
maps = cb.build_undistort_maps(cal)
sb, sa = cb.line_straightness(cal)
info = json.loads((OUT_DIR / "kalibrlash_natijasi.json").read_text())
info.update(straight_before=sb, straight_after=sa)
(OUT_DIR / "kalibrlash_natijasi.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
print(f"to'g'ri chiziqlilik RMS: oldin={sb:.3f}px, keyin={sa:.3f}px")

# ---------- 1-rasm: kalibrlash. Eng katta distorsiya ko'rinadigan tasvirni tanlash
best, bi = -1, 0
for i, ip in enumerate(cal["img_pts"]):
    r = np.linalg.norm(ip.reshape(-1, 2) - [cal["K"][0, 2], cal["K"][1, 2]], axis=1).max()
    if r > best: best, bi = r, i
f = cal["files"][bi]
raw = cv2.imread(f); und = cb.undistort(raw, maps)
ip = cal["img_pts"][bi]
und_pts = cv2.undistortPoints(ip.reshape(-1, 1, 2), cal["K"], cal["dist"], P=cal["newK"]).reshape(-1, 2)
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
for a, im, pts, t in ((ax[0], raw, ip.reshape(-1, 2), "а) Kirish + topilgan burchaklar"),
                      (ax[1], und, und_pts, "b) Distorsiyasi bartaraf etilgan")):
    a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); a.axis("off"); a.set_title(t, fontsize=8)
    g = pts.reshape(6, 9, 2)
    a.plot(pts[:, 0], pts[:, 1], "r.", ms=2.5)
    for r in (0, 5):                                   # birinchi/oxirgi qator uchun ideal to'g'ri chiziq
        a.plot([g[r, 0, 0], g[r, -1, 0]], [g[r, 0, 1], g[r, -1, 1]], "c-", lw=0.8)
ax[0].set_title("a) Kirish + topilgan burchaklar", fontsize=8)
errs = cal["per_image_err"]
ax[2].bar(range(1, len(errs) + 1), errs, color="#3b7dd8")
ax[2].axhline(cal["rms"], color="r", ls="--", lw=1, label=f"umumiy RMS={cal['rms']:.3f}")
ax[2].set_xlabel("tasvir raqami"); ax[2].set_ylabel("qayta proyeksiya xatosi, px")
ax[2].set_title("v) Har bir tasvir bo'yicha xato", fontsize=8); ax[2].legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(FIG_DIR / "1_kalibrlash.png", dpi=200); plt.close(fig)

# ---------- 2-rasm: bosqichlar paneli (sifatli namuna: hujjat_05)
o = mn.process_image(DOC_DIR / "hujjat_05.jpg", cal, maps)
c = o["corners"]; mg = 90
x0, y0 = np.maximum(c.min(0) - mg, 0).astype(int); x1, y1 = (c.max(0) + mg).astype(int)
crop = lambda im: im[y0:y1, x0:x1]
edges_vis = cv2.dilate(o["edges"], np.ones((3, 3), np.uint8))          # faqat ko'rinish uchun qalinlashtirilgan
items = [("1. Kirish (distorsiyali)", cv2.cvtColor(crop(o["raw"]), cv2.COLOR_BGR2RGB)),
         ("2. Kalibrlangan", cv2.cvtColor(crop(o["und"]), cv2.COLOR_BGR2RGB)),
         ("3. Gauss + CLAHE", crop(o["pre"]["clahe"])),
         ("4. Canny chekkalari", crop(edges_vis)),
         ("5. 4 burchak (Shi-Tomasi)", cv2.cvtColor(crop(mn.pp.draw_corners(o["und"], o["corners"])), cv2.COLOR_BGR2RGB)),
         ("6. Perspektiv tekislash", cv2.cvtColor(o["rect"], cv2.COLOR_BGR2RGB)),
         ("7. Varaq niqobi", crop(o["sheet"])),
         ("8. Matn niqobi", o["ink"])]
fig, axs = plt.subplots(2, 4, figsize=(7.4, 5.4))
for a, (t, im) in zip(axs.ravel(), items):
    a.imshow(im, cmap="gray" if im.ndim == 2 else None); a.set_title(t, fontsize=7.5); a.axis("off")
fig.tight_layout(); fig.savefig(FIG_DIR / "2_quvur.png", dpi=200); plt.close(fig)

# ---------- 3-rasm: sifat nazorati namunalari
names = [("hujjat_04", "Sifatli"), ("hujjat_02", "Burchak"), ("hujjat_06", "Chet"),
         ("hujjat_01", "Dog'"), ("hujjat_09", "Soya")]
fig, axs = plt.subplots(1, 5, figsize=(7.4, 3.0))
for a, (n, t) in zip(axs, names):
    o = mn.process_image(DOC_DIR / f"{n}.jpg", cal, maps)
    vis = o["rect"].copy()
    for key, col in (("shadow", (255, 80, 0)), ("stain", (0, 0, 255))):
        m = o["qc"][key]
        if m is not None:
            vis[m > 0] = (0.55 * vis[m > 0] + 0.45 * np.array(col)).astype(np.uint8)
    # chet va burchak nuqsonlari: varaq niqobining tashqarisini qizil rangda ko'rsatish
    out = (o["sheet_rect"] == 0)
    vis[out] = (0.4 * vis[out] + 0.6 * np.array((0, 0, 255))).astype(np.uint8)
    a.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); a.axis("off")
    ok = o["qc"]["accepted"]
    a.set_title(f"{n[-2:]} ({t})\n" + ("Qabul qilindi" if ok else "Rad etildi: " + "/".join(o["qc"]["types"])),
                fontsize=7, color="green" if ok else "red")
fig.tight_layout(); fig.savefig(FIG_DIR / "3_sifat_nazorati.png", dpi=200); plt.close(fig)
print("rasmlar tayyor")
