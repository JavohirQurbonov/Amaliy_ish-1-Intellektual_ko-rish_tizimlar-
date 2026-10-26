# -*- coding: utf-8 -*-
"""
main.py - butun quvurni (pipeline) ishga tushiruvchi asosiy skript.

Ishlatish:
    python main.py --generate      # sintetik to'plamni hosil qilib, so'ng quvurni ishga tushiradi
    python main.py                 # data/ papkasidagi (o'z) tasvirlar bilan ishga tushiradi

Bosqichlar: kalibrlash -> dastlabki ishlov -> burchaklar -> perspektiv tekislash
            -> segmentatsiya -> sifat nazorati -> baholash.
Har bir bosqichning oraliq natijasi natijalar/bosqichlar/<id>/ papkasida saqlanadi.
"""
import argparse, csv, glob, json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *
import calibration as cb
import preprocessing as pp
import perspective as ps
import segmentation as sg
import quality_control as qc
import evaluation as ev


def save(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def process_image(path, cal, maps):
    """Bitta hujjat tasviri uchun to'liq quvur. Barcha oraliq natijalarni qaytaradi."""
    name = Path(path).stem
    raw = cv2.imread(str(path))
    valid = maps[2]
    und = cb.undistort(raw, maps)                                  # 1. kalibrlash
    pre = pp.denoise_and_enhance(und)                              # 2. shovqin + CLAHE
    edges, thr = pp.canny_auto(pre["clahe"], valid)                #    Canny
    harris = pp.harris_response(pre["gauss"])                      #    Harris
    corners, method, nref = pp.find_document_quad(edges, pre["gauss"], und.shape)   # 3. burchaklar
    out = dict(name=name, raw=raw, und=und, pre=pre, edges=edges, thr=thr, harris=harris,
               corners=corners, method=method, n_refined=nref)
    if corners is None:
        out["verdict"] = "Rad etildi: varaq konturi topilmadi"; out["qc"] = None
        return out
    rect, M = ps.four_point_transform(und, corners)                # 4. perspektiv
    sheet, otsu_t, otsu_mask = sg.segment_sheet(und, valid)        # 5. segmentatsiya (varaq)
    sheet_rect = ps.warp_mask(sheet, M)
    ink = sg.segment_ink(rect, sheet_rect)                         #    segmentatsiya (matn)
    res = qc.quality_control(rect, sheet_rect)                     # 6. sifat nazorati
    out.update(rect=rect, M=M, sheet=sheet, otsu_t=otsu_t, otsu_mask=otsu_mask,
               sheet_rect=sheet_rect, ink=ink, qc=res, verdict=res["verdict"])
    return out


def save_stages(o):
    d = STAGE_DIR / o["name"]
    save(d / "01_kirish.jpg", o["raw"])
    save(d / "02_kalibrlangan.jpg", o["und"])
    save(d / "03_kontrast_CLAHE.jpg", o["pre"]["clahe"])
    save(d / "04_chekka_Canny.png", o["edges"])
    hv = cv2.normalize(np.clip(o["harris"], 0, None) ** 0.3, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    save(d / "05a_Harris_javob.png", hv)
    save(d / "05b_burchaklar.jpg", pp.draw_corners(o["und"], o["corners"]))
    if o.get("qc") is None:
        return
    save(d / "06_tekislangan.jpg", o["rect"])
    save(d / "07_niqob_varaq.png", o["sheet"])
    save(d / "08_niqob_matn.png", o["ink"])
    vis = o["rect"].copy()
    for key, col in (("shadow", (255, 0, 0)), ("stain", (0, 0, 255))):
        m = o["qc"][key]
        if m is not None:
            vis[m > 0] = (0.5 * vis[m > 0] + 0.5 * np.array(col)).astype(np.uint8)
    txt = "QABUL" if o["qc"]["accepted"] else "RAD ETILDI"
    cv2.putText(vis, txt, (30, 70), 0, 2, (0, 200, 0) if o["qc"]["accepted"] else (0, 0, 255), 5)
    save(d / "09_sifat_nazorati.jpg", vis)


def panel(o, path):
    """Bitta tasvir uchun 8 bosqichli panel (hisobot rasmi)."""
    items = [("1. Kirish (distorsiyali)", cv2.cvtColor(o["raw"], cv2.COLOR_BGR2RGB)),
             ("2. Kalibrlangan", cv2.cvtColor(o["und"], cv2.COLOR_BGR2RGB)),
             ("3. Gauss+CLAHE", o["pre"]["clahe"]),
             ("4. Canny chekkalari", o["edges"]),
             ("5. Burchaklar", cv2.cvtColor(pp.draw_corners(o["und"], o["corners"]), cv2.COLOR_BGR2RGB)),
             ("6. Tekislangan", cv2.cvtColor(o["rect"], cv2.COLOR_BGR2RGB)),
             ("7. Varaq niqobi", o["sheet"]),
             ("8. Matn niqobi", o["ink"])]
    fig, ax = plt.subplots(2, 4, figsize=(13, 8.2))
    for a, (t, im) in zip(ax.ravel(), items):
        a.imshow(im, cmap="gray" if im.ndim == 2 else None); a.set_title(t, fontsize=10); a.axis("off")
    fig.suptitle(f"{o['name']}: {o['verdict'][:110]}", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="sintetik ma'lumotlarni hosil qilish")
    args = ap.parse_args()
    if args.generate:
        import synth_data; synth_data.generate_all()
    for d in (OUT_DIR, STAGE_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---- 1. kalibrlash
    cal = cb.calibrate()
    maps = cb.build_undistort_maps(cal)
    gtc = GT_DIR / "kamera_haqiqiy.json"
    cal_info = dict(rms=cal["rms"], n_used=cal["n_used"], n_total=cal["n_total"],
                    K=cal["K"].tolist(), dist=cal["dist"].ravel().tolist(),
                    per_image_err=cal["per_image_err"])
    if gtc.exists():
        g = json.loads(gtc.read_text())
        cal_info["haqiqiy_K"] = g["K"]; cal_info["haqiqiy_dist"] = g["dist"]
    (OUT_DIR / "kalibrlash_natijasi.json").write_text(json.dumps(cal_info, indent=1, ensure_ascii=False))
    print(f"[kalibrlash] RMS={cal['rms']:.4f}px, fx={cal['K'][0,0]:.1f}, fy={cal['K'][1,1]:.1f}")

    # ---- 2-6. har bir hujjat
    rows, results = [], []
    for f in sorted(glob.glob(str(DOC_DIR / "*.jpg"))):
        o = process_image(f, cal, maps)
        save_stages(o)
        meta = ev.load_meta(o["name"])
        row = dict(id=o["name"], verdikt=o["verdict"])
        if o.get("qc") is not None:
            m = o["qc"]["metrics"]
            row.update(burchak_toldirish_min=min(m["corner_fill"]), chet_rms_maks=max(m["edge_rms"].values()),
                       chet_oraliq_maks=max(m["edge_range"].values()),
                       burchak_ogish_maks=max(abs(90 - a) for a in m["angles"].values()),
                       soya_ulushi=m["shadow_frac"], dog_maydoni=m["stain_area"],
                       aniqlangan_turlar=",".join(o["qc"]["types"]))
        if meta is not None and o.get("qc") is not None:
            gt_c = ev.gt_corners_undist(meta, cal)
            row["burchak_xatosi_px"] = float(np.mean(np.linalg.norm(gt_c - o["corners"], axis=1)))
            gt_sheet = ev.gt_sheet_mask_undist(meta, cal, o["und"].shape)
            row["IoU_varaq"] = ev.iou(o["sheet"], gt_sheet)
            gt_ink = ev.gt_ink_rectified(meta, cal, o["M"], o["name"])
            row["IoU_matn"] = ev.iou(o["ink"], gt_ink)
            row["nuqson_haqiqiy"] = meta["nuqson"]; row["yoritilganlik"] = meta["yoritilganlik"]
            row["rad_etildi_haqiqiy"] = meta["nuqson"] != "yo'q"
            row["rad_etildi_bashorat"] = not o["qc"]["accepted"]
            row["xulosa_togri"] = row["rad_etildi_haqiqiy"] == row["rad_etildi_bashorat"]
            row["tur_togri"] = (meta["nuqson"] in o["qc"]["types"]) if row["rad_etildi_haqiqiy"] else row["xulosa_togri"]
        rows.append(row); results.append(o)
        print(f"  {o['name']}: {o['verdict'][:100]}")
    for o in results:
        if o.get("qc") is not None and o["name"] in ("hujjat_01", "hujjat_02", "hujjat_06", "hujjat_09"):
            panel(o, FIG_DIR / f"panel_{o['name']}.png")

    # ---- CSV va umumiy ko'rsatkichlar
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "id", k != "verdikt", k))
    with open(OUT_DIR / "natijalar.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    lab = [r for r in rows if "xulosa_togri" in r]
    if lab:
        tp, tn, fp, fn = ev.confusion(lab)
        summ = dict(n=len(lab), qc_togri=sum(r["xulosa_togri"] for r in lab),
                    qc_aniqlik=sum(r["xulosa_togri"] for r in lab) / len(lab),
                    tur_aniqlik=sum(r["tur_togri"] for r in lab) / len(lab),
                    TP=tp, TN=tn, FP=fp, FN=fn,
                    IoU_varaq_mean=float(np.mean([r["IoU_varaq"] for r in lab])),
                    IoU_varaq_min=float(np.min([r["IoU_varaq"] for r in lab])),
                    IoU_matn_mean=float(np.mean([r["IoU_matn"] for r in lab])),
                    IoU_matn_min=float(np.min([r["IoU_matn"] for r in lab])),
                    burchak_xato_mean=float(np.mean([r["burchak_xatosi_px"] for r in lab if r["nuqson_haqiqiy"] in ("yo'q", "dog'", "soya")])),
                    burchak_xato_max=float(np.max([r["burchak_xatosi_px"] for r in lab if r["nuqson_haqiqiy"] in ("yo'q", "dog'", "soya")])))
        (OUT_DIR / "umumiy_korsatkichlar.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
        print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
