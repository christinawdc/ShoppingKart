"""Build the PDF bill for an order."""

from fpdf import FPDF


def make_bill(order, when, lines, total):
    """lines: [(product dict, qty)]. Returns the PDF as bytes."""
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 10, "ShoppingKart", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(110)
    pdf.cell(0, 6, "Bill", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)
    pdf.ln(4)
    pdf.cell(0, 6, f"Order: {order}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Date: {when:%d %b %Y, %H:%M}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Helvetica has no rupee sign, so amounts are written "Rs."
    widths = (95, 20, 30, 35)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(235)
    for w, h, a in zip(widths, ("Item", "Qty", "Price", "Amount"), "LRRR"):
        pdf.cell(w, 8, h, border=0, align=a, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    for p, q in lines:
        pdf.cell(widths[0], 8, p["name"])
        pdf.cell(widths[1], 8, str(q), align="R")
        pdf.cell(widths[2], 8, f"Rs. {p['price']:,.2f}", align="R")
        pdf.cell(widths[3], 8, f"Rs. {p['price'] * q:,.2f}", align="R")
        pdf.ln()

    pdf.ln(2)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + sum(widths), pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(sum(widths[:3]), 9, "Total", align="R")
    pdf.cell(widths[3], 9, f"Rs. {total:,.2f}", align="R")
    pdf.ln(16)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(110)
    pdf.cell(0, 6, "Thank you for shopping with ShoppingKart!", align="C")
    return bytes(pdf.output())
