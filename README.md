# Mini-loyiha: hujjat tasviri uchun kalibrlash, tekislash, segmentatsiya va sifat nazorati

Fan: **Intellektual ko'rish tizimlari**. Faqat klassik kompyuter ko'rish usullari (chuqur o'rganishsiz).

## Ishga tushirish
```bash
pip install -r requirements.txt
python main.py --generate     # sintetik to'plamni hosil qiladi va butun quvurni ishga tushiradi
python main.py                # data/ ichidagi tasvirlar bilan (o'z suratlaringiz bilan ham)
python make_report_figures.py # hisobot rasmlarini qayta yaratadi
```

## O'z (haqiqiy) suratlaringiz bilan ishlatish
1. 9x6 ichki burchakli shaxmat taxtasini (katak 25 mm; boshqa bo'lsa `config.py` da `SQUARE_MM`) turli burchakdan
   15-20 marta suratga oling -> `data/kalibrlash/*.jpg`
2. Xuddi shu kamera bilan hujjat suratlarini oling -> `data/hujjatlar/*.jpg`
3. `python main.py` (generator ishlatilmaydi). `data/etalon/` bo'lmasa IoU va aniqlik hisoblanmaydi,
   lekin barcha oraliq natijalar va sifat nazorati xulosasi baribir chiqadi.
4. QC bo'sag'alarini `config.py` dagi `QC` lug'atida o'z suratlaringizga moslang.

## Tuzilma
| Fayl | Vazifa |
|---|---|
| `config.py` | yo'llar, parametrlar, QC bo'sag'alari |
| `synth_data.py` | etalonli sintetik to'plam generatori |
| `calibration.py` | 1-bosqich: shaxmat kalibrlash, distorsiyani bartaraf etish |
| `preprocessing.py` | 2-bosqich: Gauss/bilateral, CLAHE, Canny, Harris, Shi-Tomasi, 4 burchak |
| `perspective.py` | 3-bosqich: to'rt nuqtali perspektiv almashtirish |
| `segmentation.py` | 4-bosqich: Otsu+watershed, adaptiv bo'sag'a, region growing |
| `quality_control.py` | 5-bosqich: qoidaviy sifat nazorati |
| `evaluation.py` | IoU, burchak xatosi, aniqlik |
| `main.py` | butun quvur, oraliq natijalarni saqlash, CSV/JSON hisobot |

## Natijalar (`natijalar/`)
* `bosqichlar/<id>/` - har bir tasvir uchun 01_kirish ... 09_sifat_nazorati (oraliq natijalar)
* `rasmlar/` - hisobot rasmlari; `natijalar.csv` - har bir tasvir bo'yicha jadval
* `umumiy_korsatkichlar.json`, `kalibrlash_natijasi.json`
