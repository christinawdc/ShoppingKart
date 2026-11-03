"""Write checkout/qr_codes.html: one QR code per product, to print or show on a second screen.

    python checkout/make_qr.py
"""

import base64
import io
import json
from pathlib import Path

import qrcode

here = Path(__file__).parent
products = json.loads((here / "products.json").read_text())

cards = []
for p in products:
    buf = io.BytesIO()
    qrcode.make(f"shoppingkart:{p['id']}", box_size=8, border=2).save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    cards.append(f'<figure><img src="data:image/png;base64,{b64}"><figcaption>'
                 f'{p["emoji"]} {p["name"]}<br>₹{p["price"]}</figcaption></figure>')

(here / "qr_codes.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>ShoppingKart QR codes</title>
<style>body{{font-family:system-ui;margin:24px}}main{{display:grid;grid-template-columns:repeat(4,1fr);gap:20px}}
figure{{margin:0;text-align:center;border:1px solid #ccc;border-radius:8px;padding:10px}}img{{width:100%}}
@media print{{figure{{break-inside:avoid}}}}</style><h1>ShoppingKart QR codes</h1><main>{''.join(cards)}</main>""")
print(f"wrote {here / 'qr_codes.html'} ({len(products)} codes)")
