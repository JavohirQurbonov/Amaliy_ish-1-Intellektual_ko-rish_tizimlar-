# -*- coding: utf-8 -*-
"""
config.py - loyihaning markaziy sozlamalari.

Barcha yo'llar, kalibrlash parametrlari, chiqish tasvir o'lchami va
sifat nazorati (QC) bo'sag'alari shu yerda jamlangan. Bo'sag'alar
tajriba yo'li bilan (sintetik tasvirlarning dastlabki 'sozlash' qismida)
tanlangan va hisobotda asoslab berilgan.
"""
from pathlib import Path

# ---------------------------------------------------------------- Yo'llar
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CALIB_DIR = DATA_DIR / "kalibrlash"        # shaxmat taxtasi tasvirlari
DOC_DIR = DATA_DIR / "hujjatlar"           # test hujjat tasvirlari
GT_DIR = DATA_DIR / "etalon"               # etalon (ground truth) ma'lumotlar
OUT_DIR = ROOT / "natijalar"               # barcha natijalar
STAGE_DIR = OUT_DIR / "bosqichlar"         # har bir tasvir uchun oraliq natijalar
FIG_DIR = OUT_DIR / "rasmlar"              # hisobot uchun rasmlar

# -------------------------------------------------- Kalibrlash parametrlari
BOARD_INNER = (9, 6)       # shaxmat taxtasining ichki burchaklari soni (9x6)
SQUARE_MM = 25.0           # bitta katak o'lchami, mm

# -------------------------------------------------- Tekislangan tasvir
OUT_W, OUT_H = 1000, 1414  # A4 (1:1.414) nisbatidagi chiqish o'lchami, piksel

# ----------------------------------------------- Sifat nazorati bo'sag'alari
QC = dict(
    corner_patch=100,        # burchak yaxlitligi tekshiriladigan oyna (px)
    corner_fill_min=0.90,    # oynaning kamida 90% i varaq bilan to'lishi shart
    edge_margin=150,         # chet profili burchaklardan shuncha px uzoqda olinadi
    edge_band=120,           # chetdan ichkariga qaraladigan polosa kengligi (px)
    edge_rms_max=3.5,        # chet profilining to'g'ri chiziqdan RMS og'ishi (px)
    edge_range_max=10.0,     # chet profili og'ishining (p95-p5) oralig'i (px)
    angle_dev_max=2.0,       # to'g'ri burchakdan eng katta og'ish (daraja)
    shadow_ratio=0.78,       # yoritilganlik sathi p90 ning shu ulushidan past => soya
    shadow_area_max=0.05,    # soya egallagan maydon ulushi chegarasi
    stain_dark_ratio=0.80,   # fon yorqinligining shu ulushidan qoraroq => dog' nomzodi
    stain_chroma_dev=0.035,  # xromatiklikning varaq median rangidan og'ishi (dog' belgisi)
    stain_open=33,           # dog'ni matndan ajratuvchi morfologik 'opening' oynasi (px)
    stain_area_min=2500,     # dog' deb hisoblanadigan eng kichik maydon (px^2)
)
