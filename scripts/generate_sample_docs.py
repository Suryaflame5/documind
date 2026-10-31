"""
Sample Document Generator
Generates realistic, professional enterprise PDF documents using ReportLab,
including an exact replica of the reference invoice (INV-2026-00432.pdf),
a retail receipt, a tax form, and an enterprise service agreement.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from documind.core.config import settings

SAMPLES_DIR = settings.SAMPLES_DIR
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def generate_reference_invoice(output_path: Path):
    """
    Synthesizes the exact reference invoice seen in the problem specification:
    INV-2026-00432.pdf from Summit Technologies LLC to Acme Co.
    """
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    # Header Logo / Title
    c.setFont("Helvetica-Bold", 22)
    c.setFillColor(colors.HexColor("#1e3a8a"))
    c.drawString(54, height - 70, "▲ SUMMIT")
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(54, height - 86, "TECHNOLOGIES")

    c.setFont("Helvetica-Bold", 24)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawRightString(width - 54, height - 75, "INVOICE")

    # Vendor Details (Left)
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.HexColor("#1e293b"))
    c.drawString(54, height - 130, "Summit Technologies LLC")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#475569"))
    c.drawString(54, height - 145, "345 Innovation Way, Suite 200")
    c.drawString(54, height - 158, "Denver, CO 80202")
    c.drawString(54, height - 171, "United States")
    c.drawString(54, height - 184, "info@summittechnologies.com")

    # Invoice Details Box (Right)
    box_x = width - 220
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.setLineWidth(1)
    c.roundRect(box_x, height - 195, 166, 75, 4, stroke=1, fill=0)

    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(box_x + 10, height - 135, "Invoice #")
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(box_x + 10, height - 148, "INV-2026-00432")

    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(box_x + 10, height - 165, "Invoice Date")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(box_x + 10, height - 176, "May 20, 2026")

    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(box_x + 95, height - 165, "Due Date")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(box_x + 95, height - 176, "Jun 19, 2026")

    # Bill To Box
    c.roundRect(54, height - 280, 220, 70, 4, stroke=1, fill=0)
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(64, height - 225, "Bill To")
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(64, height - 240, "Acme Co.")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#475569"))
    c.drawString(64, height - 253, "123 Market Street, Floor 4")
    c.drawString(64, height - 266, "San Francisco, CA 94105, United States")

    # Table Header
    y_table = height - 320
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.rect(54, y_table - 18, width - 108, 22, stroke=0, fill=1)
    c.setStrokeColor(colors.HexColor("#e2e8f0"))
    c.line(54, y_table - 18, width - 54, y_table - 18)

    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.HexColor("#475569"))
    c.drawString(64, y_table - 12, "DESCRIPTION")
    c.drawRightString(340, y_table - 12, "QTY")
    c.drawRightString(430, y_table - 12, "UNIT PRICE")
    c.drawRightString(width - 64, y_table - 12, "AMOUNT")

    # Line Items
    items = [
        ("Cloud Migration Services", "40", "$125.00", "$5,000.00"),
        ("Security Assessment", "1", "$2,500.00", "$2,500.00"),
        ("Data Integration & ETL", "1", "$1,750.00", "$1,750.00"),
    ]

    curr_y = y_table - 42
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#1e293b"))
    for desc, qty, unit, amt in items:
        c.drawString(64, curr_y, desc)
        c.drawRightString(340, curr_y, qty)
        c.drawRightString(430, curr_y, unit)
        c.drawRightString(width - 64, curr_y, amt)
        c.setStrokeColor(colors.HexColor("#f1f5f9"))
        c.line(54, curr_y - 8, width - 54, curr_y - 8)
        curr_y -= 26

    # Totals Summary
    tot_y = curr_y - 15
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(350, tot_y, "Subtotal")
    c.drawRightString(width - 64, tot_y, "$9,250.00")

    tot_y -= 18
    c.drawString(350, tot_y, "Tax (8.25%)")
    c.drawRightString(width - 64, tot_y, "$763.13")

    tot_y -= 24
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.line(340, tot_y + 16, width - 54, tot_y + 16)
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.roundRect(335, tot_y - 8, width - 389, 26, 3, stroke=1, fill=1)
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(345, tot_y, "TOTAL DUE")
    c.drawRightString(width - 64, tot_y, "USD $10,013.13")

    # Footer note
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(54, 80, "Thank you for your business!")

    c.save()
    print(f"[+] Generated Reference Invoice: {output_path}")


def generate_receipt_pdf(output_path: Path):
    c = canvas.Canvas(str(output_path), pagesize=(300, 520))
    w, h = 300, 520

    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(w / 2, h - 35, "STAR COFFEE & BAKERY")
    c.setFont("Helvetica", 8)
    c.drawCentredString(w / 2, h - 50, "452 Lexington Ave, New York, NY 10017")
    c.drawCentredString(w / 2, h - 62, "Tel: (212) 555-0199")

    c.line(20, h - 75, w - 20, h - 75)
    c.setFont("Helvetica", 8)
    c.drawString(20, h - 90, "RECEIPT #: RCPT-84920")
    c.drawString(20, h - 104, "DATE: 2026-06-14  11:24 AM")
    c.drawString(20, h - 118, "CASHIER: Sarah (POS-02)")
    c.line(20, h - 130, w - 20, h - 130)

    items = [
        ("Espresso Roast - 1lb Bag", "$14.50"),
        ("Cold Brew 16oz", "$4.75"),
        ("Almond Croissant", "$3.95"),
        ("Artisan Sourdough Loaf", "$6.25"),
    ]
    y = h - 150
    for desc, amt in items:
        c.drawString(20, y, desc)
        c.drawRightString(w - 20, y, amt)
        y -= 20

    c.line(20, y - 5, w - 20, y - 5)
    y -= 25
    c.drawString(20, y, "SUBTOTAL")
    c.drawRightString(w - 20, y, "$29.45")
    y -= 16
    c.drawString(20, y, "SALES TAX (8.00%)")
    c.drawRightString(w - 20, y, "$2.36")
    y -= 20
    c.setFont("Helvetica-Bold", 10)
    c.drawString(20, y, "TOTAL AMOUNT")
    c.drawRightString(w - 20, y, "$31.81")

    y -= 30
    c.setFont("Helvetica", 8)
    c.drawString(20, y, "PAYMENT METHOD: VISA **** 4092")
    c.drawString(20, y - 14, "AUTH CODE: 871294")
    c.drawCentredString(w / 2, y - 50, "THANK YOU FOR YOUR VISIT!")
    c.save()
    print(f"[+] Generated Receipt PDF: {output_path}")


def generate_form_pdf(output_path: Path):
    c = canvas.Canvas(str(output_path), pagesize=letter)
    w, h = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(54, h - 60, "Form W-9")
    c.setFont("Helvetica-Bold", 11)
    c.drawString(140, h - 55, "Request for Taxpayer Identification Number and Certification")
    c.setFont("Helvetica", 8)
    c.drawString(140, h - 68, "Department of the Treasury - Internal Revenue Service")
    c.line(54, h - 75, w - 54, h - 75)

    c.setFont("Helvetica", 9)
    c.drawString(54, h - 100, "1. Name of entity: Acme Co.")
    c.drawString(54, h - 120, "2. Business name: Acme Cloud Enterprise Division")
    c.drawString(54, h - 140, "3. Federal tax classification: [X] Corporation")
    c.drawString(54, h - 160, "4. Address: 123 Market Street, Floor 4, San Francisco, CA 94105")
    c.drawString(54, h - 180, "5. Requester's name: State Compliance Board")

    c.line(54, h - 195, w - 54, h - 195)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(54, h - 215, "Part I: Taxpayer Identification Number (TIN)")
    c.setFont("Helvetica", 9)
    c.drawString(54, h - 235, "Employer Identification Number (EIN): 94-2849102")
    c.drawString(54, h - 255, "Form Reference ID: FRM-2026-5819")
    c.drawString(54, h - 275, "Certification Date: 2026-04-15")

    c.line(54, h - 290, w - 54, h - 290)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(54, h - 310, "Part II: Certification")
    c.setFont("Helvetica", 8)
    c.drawString(54, h - 330, "Under penalties of perjury, I certify that the number shown on this form is my correct taxpayer identification number.")
    c.drawString(54, h - 350, "Authorized Signature: /s/ John Doe, Chief Financial Officer")
    c.drawString(54, h - 370, "Date Signed: 2026-04-15")
    c.save()
    print(f"[+] Generated Tax Form PDF: {output_path}")


def generate_contract_pdf(output_path: Path):
    c = canvas.Canvas(str(output_path), pagesize=letter)
    w, h = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(w / 2, h - 60, "MASTER SERVICES AGREEMENT")
    c.setFont("Helvetica", 9)
    c.drawCentredString(w / 2, h - 75, "Agreement Reference: AGR-2026-9021")
    c.line(54, h - 85, w - 54, h - 85)

    c.setFont("Helvetica", 9)
    c.drawString(54, h - 110, "This Master Services Agreement is entered into on March 15, 2026, by and between:")
    c.setFont("Helvetica-Bold", 9)
    c.drawString(54, h - 130, "PARTY A (Service Provider): Summit Technologies LLC")
    c.drawString(54, h - 150, "PARTY B (Client): Acme Co.")
    c.setFont("Helvetica", 9)
    c.drawString(54, h - 180, "1. PURPOSE & SCOPE OF ENGAGEMENT")
    c.drawString(54, h - 195, "Service Provider shall deliver cloud infrastructure migration, automated document intelligence pipelines,")
    c.drawString(54, h - 208, "and security compliance auditing according to mutually agreed Statements of Work.")

    c.drawString(54, h - 235, "2. CONFIDENTIALITY & DATA PROTECTION")
    c.drawString(54, h - 248, "Both parties maintain strict covenants of confidentiality concerning proprietary software, source algorithms,")
    c.drawString(54, h - 261, "customer data records, and security telemetry.")

    c.drawString(54, h - 290, "3. GOVERNING LAW & JURISDICTION")
    c.drawString(54, h - 303, "This agreement is governed by the laws of the State of Delaware.")
    c.drawString(54, h - 340, "Effective Date: March 15, 2026")
    c.drawString(54, h - 360, "Total Contract Baseline Commitment: USD $0.00 (Governed by individual SOWs)")
    c.save()
    print(f"[+] Generated Contract PDF: {output_path}")


def generate_all_samples():
    generate_reference_invoice(SAMPLES_DIR / "sample_invoice.pdf")
    generate_receipt_pdf(SAMPLES_DIR / "sample_receipt.pdf")
    generate_form_pdf(SAMPLES_DIR / "sample_form.pdf")
    generate_contract_pdf(SAMPLES_DIR / "sample_contract.pdf")


if __name__ == "__main__":
    generate_all_samples()
