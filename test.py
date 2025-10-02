from lightkurve import search_lightcurvefile
from astroquery.ipac.nexsci import NasaExoplanetArchive
import pandas as pd

# ดึงข้อมูลดาวเคราะห์ตัวอย่าง (K2-18 b)
planet = "K2-18 b"
ps_table = NasaExoplanetArchive.query_object(planet, table="ps")
print(ps_table[["pl_name","pl_orbper","pl_tranmid"]])

# ดึง Light Curve ของดาวแม่ (TIC)
target = "TIC 25155310"
lc = search_lightcurvefile(target, mission="TESS").download().PDCSAP_FLUX.normalize()