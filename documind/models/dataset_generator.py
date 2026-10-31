"""
Enterprise Document Dataset Generator
Synthesizes realistic, diverse, multi-class enterprise documents with ground-truth labels
for invoices, receipts, forms, and contracts/other documents.
"""

import random
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path


VENDORS = [
    ("Summit Technologies LLC", "345 Innovation Way, Suite 200, Denver, CO 80202"),
    ("Acme Cloud Solutions Inc.", "742 Evergreen Terrace, Springfield, OR 97477"),
    ("Apex Global Logistics", "1200 Harbor Boulevard, Long Beach, CA 90802"),
    ("Nexus Data Systems", "500 Tech Parkway, Austin, TX 78701"),
    ("Vanguard Enterprise Software", "101 Federal Street, Boston, MA 02110"),
    ("Pacific Retailers Group", "88 Market St, San Francisco, CA 94105"),
    ("Star Coffee & Bakery", "452 Lexington Ave, New York, NY 10017"),
    ("Metro Office Supplies Co.", "890 Industrial Rd, Chicago, IL 60607"),
    ("Blue Horizon Medical Labs", "320 Health Sciences Dr, Raleigh, NC 27601"),
    ("OmniCorp Facilities Management", "650 Corporate Woods, Atlanta, GA 30339"),
]

CUSTOMERS = [
    ("Acme Co.", "123 Market Street, Floor 4, San Francisco, CA 94105"),
    ("Globex Corporation", "555 Industrial Pkwy, Seattle, WA 98101"),
    ("Initech Enterprises", "4120 Freidrich Ln, Austin, TX 78744"),
    ("Hooli Technology Group", "3200 Innovation Blvd, Palo Alto, CA 94304"),
    ("Massive Dynamic Corp.", "700 Fifth Avenue, New York, NY 10019"),
    ("Wayne Enterprises LLC", "1007 Mountain Drive, Gotham, NJ 07001"),
    ("Stark Industries Ltd.", "10880 Wilshire Blvd, Los Angeles, CA 90024"),
    ("Umbrella Pharmaceuticals", "440 Raccoon Way, Raccoon City, MO 63011"),
]

INVOICE_ITEMS = [
    ("Cloud Migration Services", 125.0, 40, 5000.0),
    ("Security Architecture Assessment", 2500.0, 1, 2500.0),
    ("Data Integration & ETL Pipeline", 1750.0, 1, 1750.0),
    ("Kubernetes Cluster Maintenance", 800.0, 2, 1600.0),
    ("PostgreSQL Database Optimization", 1200.0, 1, 1200.0),
    ("Frontend React Component Library", 3500.0, 1, 3500.0),
    ("Enterprise Support Retainer - Tier 3", 4500.0, 1, 4500.0),
    ("Network Security Penetration Test", 3200.0, 1, 3200.0),
    ("API Gateway Rate Limiting Setup", 950.0, 2, 1900.0),
    ("CI/CD Deployment Pipeline Automation", 1400.0, 2, 2800.0),
]

RECEIPT_ITEMS = [
    ("Espresso Roast - 1lb Bag", 14.50),
    ("Artisan Sourdough Loaf", 6.25),
    ("Cold Brew 16oz", 4.75),
    ("Almond Croissant", 3.95),
    ("Premium Gasoline 87 Octane", 45.20),
    ("Car Wash - Deluxe Package", 15.00),
    ("Office Supplies Pack (Pens/Pads)", 22.80),
    ("HDMI DisplayPort Cable 6ft", 18.99),
    ("USB-C Fast Charging Hub", 34.50),
    ("Wireless Bluetooth Mouse", 28.00),
]

FORM_TITLES = [
    ("W-9 Request for Taxpayer Identification Number and Certification", "Department of the Treasury - Internal Revenue Service"),
    ("Form W-4 Employee's Withholding Certificate", "Department of the Treasury - Internal Revenue Service"),
    ("Form I-9 Employment Eligibility Verification", "U.S. Citizenship and Immigration Services"),
    ("Vendor Onboarding & ACH Direct Deposit Authorization Form", "Corporate Finance & Treasury Services"),
    ("Confidential Incident Disclosure and Compliance Audit Form", "Office of Ethics and Regulatory Compliance"),
    ("Corporate Supplier Diversity Registration Questionnaire", "Enterprise Procurement Operations"),
]

OTHER_DOCUMENTS = [
    (
        "MASTER SERVICES AGREEMENT",
        "This Master Services Agreement ('Agreement') is entered into as of the Effective Date by and between the parties. "
        "The Service Provider agrees to perform technical consulting, software engineering, and systems administration as described in subsequent Statements of Work (SOW). "
        "CONFIDENTIALITY: Each party agrees to protect proprietary information with at least the same degree of care it uses for its own confidential assets. "
        "TERMINATION: Either party may terminate this agreement upon thirty (30) days written notice of material breach. "
        "GOVERNING LAW: This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware."
    ),
    (
        "NON-DISCLOSURE AND MUTUAL CONFIDENTIALITY COVENANT",
        "This Mutual Non-Disclosure Agreement ('NDA') protects confidential enterprise intellectual property disclosed between the undersigned parties. "
        "Definition of Confidential Information: Includes technical specifications, source code, financial projections, customer rosters, and product roadmaps. "
        "Term: The obligations hereunder shall continue in effect for a period of five (5) years following receipt of disclosure. "
        "Injunctive Relief: Breach of confidentiality covenants will cause irreparable harm for which monetary damages alone would be inadequate."
    ),
    (
        "Q3 ENTERPRISE SYSTEMS AUDIT & ARCHITECTURE REVIEW",
        "EXECUTIVE SUMMARY: During the third quarter audit, enterprise infrastructure reliability exceeded 99.98% uptime SLA across all regions. "
        "Key vulnerability assessments identified zero critical zero-day exploits. Recommendations include hardening IAM policy roles, rotating service account credentials, "
        "and migrating deprecated REST endpoints to gRPC streaming microservices. Next quarterly review scheduled for December 2026."
    ),
    (
        "SOFTWARE LICENSE AGREEMENT & TERMS OF SERVICE",
        "IMPORTANT - READ CAREFULLY: This End User License Agreement is a legal agreement between Licensee and Licensor. "
        "GRANT OF LICENSE: Licensor grants Licensee a non-exclusive, non-transferable, revocable license to execute the Software strictly in accordance with documentation. "
        "LIMITATION OF LIABILITY: IN NO EVENT SHALL LICENSOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES WHATSOEVER."
    ),
]


class EnterpriseDatasetGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate_invoice_sample(self, doc_idx: int) -> Dict[str, Any]:
        vendor_name, vendor_addr = random.choice(VENDORS[:5])
        cust_name, cust_addr = random.choice(CUSTOMERS)
        inv_num = f"INV-2026-{10000 + doc_idx}"
        day = random.randint(1, 28)
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
        month = random.choice(month_names)
        inv_date = f"{month} {day:02d}, 2026"
        due_day = min(28, day + 15)
        due_date = f"{month} {due_day:02d}, 2026"
        
        # Pick 2-4 items
        selected_items = random.sample(INVOICE_ITEMS, k=random.randint(2, 4))
        subtotal = 0.0
        line_items_text = []
        parsed_items = []
        for desc, unit_price, qty, _ in selected_items:
            amt = unit_price * qty
            subtotal += amt
            line_items_text.append(f"{desc:<36} {qty:>3}    ${unit_price:>8.2f}    ${amt:>10.2f}")
            parsed_items.append({
                "description": desc,
                "quantity": float(qty),
                "unit_price": float(unit_price),
                "amount": float(amt)
            })
            
        tax_rate = 0.0825
        tax = round(subtotal * tax_rate, 2)
        total = round(subtotal + tax, 2)
        
        raw_text = f"""
================================================================================
                                INVOICE
================================================================================
Vendor:
{vendor_name}
{vendor_addr}
info@{vendor_name.lower().replace(' ', '').replace('.', '').replace(',', '')}.com

Invoice #: {inv_num}
Invoice Date: {inv_date}
Due Date: {due_date}

Bill To:
{cust_name}
{cust_addr}

--------------------------------------------------------------------------------
DESCRIPTION                          QTY    UNIT PRICE        AMOUNT
--------------------------------------------------------------------------------
""" + "\n".join(line_items_text) + f"""
--------------------------------------------------------------------------------
                                      Subtotal:        ${subtotal:>10.2f}
                                      Tax (8.25%):     ${tax:>10.2f}
                                      TOTAL DUE:       ${total:>10.2f}
                                      Currency:        USD

Payment Terms: Net 30. Please remit electronic ACH payment to First National Bank.
Thank you for your business!
"""
        return {
            "text": raw_text.strip(),
            "category": "invoice",
            "metadata": {
                "document_id": inv_num,
                "date": inv_date,
                "due_date": due_date,
                "total_amount": f"{total:.2f}",
                "currency": "USD",
                "vendor_name": vendor_name,
                "customer_name": cust_name,
                "subtotal": f"{subtotal:.2f}",
                "tax": f"{tax:.2f}",
                "line_items": parsed_items
            }
        }

    def generate_receipt_sample(self, doc_idx: int) -> Dict[str, Any]:
        vendor_name, vendor_addr = random.choice(VENDORS[5:])
        rcpt_num = f"RCPT-{80000 + doc_idx}"
        day = random.randint(1, 28)
        month = random.randint(1, 10)
        rcpt_date = f"2026-{month:02d}-{day:02d}"
        
        selected_items = random.sample(RECEIPT_ITEMS, k=random.randint(2, 5))
        subtotal = sum(price for _, price in selected_items)
        tax = round(subtotal * 0.08, 2)
        total = round(subtotal + tax, 2)
        
        items_lines = [f"{desc:<28} ${amt:>6.2f}" for desc, amt in selected_items]
        
        raw_text = f"""
               {vendor_name.upper()}
            {vendor_addr}
                 TEL: (555) 019-2831
            
RECEIPT #: {rcpt_num}
DATE: {rcpt_date} 14:32:10
CASHIER: EMP-042
REGISTER: POS-03

----------------------------------------
ITEM                             AMOUNT
----------------------------------------
""" + "\n".join(items_lines) + f"""
----------------------------------------
SUBTOTAL:                        ${subtotal:>6.2f}
SALES TAX (8.00%):               ${tax:>6.2f}
TOTAL AMOUNT:                    ${total:>6.2f}
PAID BY: VISA ENDING IN 4092
AUTH CODE: 981249
REF: {rcpt_num}

THANK YOU FOR VISITING!
PLEASE RETAIN FOR YOUR RECORDS.
"""
        return {
            "text": raw_text.strip(),
            "category": "receipt",
            "metadata": {
                "document_id": rcpt_num,
                "date": rcpt_date,
                "due_date": rcpt_date,
                "total_amount": f"{total:.2f}",
                "currency": "USD",
                "vendor_name": vendor_name,
                "customer_name": "Walk-in Customer",
                "subtotal": f"{subtotal:.2f}",
                "tax": f"{tax:.2f}"
            }
        }

    def generate_form_sample(self, doc_idx: int) -> Dict[str, Any]:
        title, agency = random.choice(FORM_TITLES)
        cust_name, cust_addr = random.choice(CUSTOMERS)
        form_id = f"FRM-2026-{5000 + doc_idx}"
        day = random.randint(1, 28)
        form_date = f"2026-06-{day:02d}"
        ein = f"12-{random.randint(1000000, 9999999)}"
        ssn_mask = f"***-**-{random.randint(1000, 9999)}"
        
        raw_text = f"""
================================================================================
FORM {form_id} | {title}
{agency}
================================================================================

PART I: IDENTIFYING INFORMATION
1. Name of Entity or Individual: {cust_name}
2. Business Name / Disregarded Entity Name: {cust_name} Division of Americas
3. Federal Tax Classification: [X] Corporation  [ ] Partnership  [ ] Individual
4. Address (number, street, and apt. or suite no.): {cust_addr}
5. Requester's Name and Address: State Revenue Compliance Board, Austin, TX

PART II: TAXPAYER IDENTIFICATION NUMBER (TIN)
Enter your TIN in the appropriate box.
Employer Identification Number (EIN): {ein}
Social Security Number (SSN): {ssn_mask}

PART III: CERTIFICATION
Under penalties of perjury, I certify that:
1. The number shown on this form is my correct taxpayer identification number;
2. I am not subject to backup withholding because I am exempt;
3. I am a U.S. citizen or other U.S. person.

Authorized Signature: /s/ Document Custodian
Date: {form_date}
Official Document Reference: {form_id}
Status: Completed and Filed
"""
        return {
            "text": raw_text.strip(),
            "category": "form",
            "metadata": {
                "document_id": form_id,
                "date": form_date,
                "due_date": form_date,
                "total_amount": "0.00",
                "currency": "USD",
                "vendor_name": agency,
                "customer_name": cust_name
            }
        }

    def generate_other_sample(self, doc_idx: int) -> Dict[str, Any]:
        title, content = random.choice(OTHER_DOCUMENTS)
        cust_name, cust_addr = random.choice(CUSTOMERS)
        vendor_name, _ = random.choice(VENDORS[:4])
        doc_ref = f"AGR-2026-{3000 + doc_idx}"
        day = random.randint(1, 28)
        doc_date = f"March {day:02d}, 2026"
        
        raw_text = f"""
================================================================================
                       {title}
                    Document Reference: {doc_ref}
================================================================================

THIS AGREEMENT is made effective as of {doc_date}, by and between:
PARTY A: {vendor_name}
PARTY B: {cust_name}

RECITALS:
WHEREAS, Party A and Party B desire to engage in confidential strategic deliberations;
NOW, THEREFORE, the parties mutually covenant and agree as follows:

1. SCOPE AND OBLIGATIONS
{content}

2. REPRESENTATIONS AND WARRANTIES
Each party represents that it has the full corporate authority to enter into this legal agreement.
Neither party shall assign its rights under this document without prior written consent.

3. MISCELLANEOUS
This document constitutes the entire understanding between the parties with respect to its subject matter.
No amendment shall be binding unless executed in writing.

IN WITNESS WHEREOF, the duly authorized representatives have executed this instrument.
Executed on: {doc_date}
Agreement ID: {doc_ref}
"""
        return {
            "text": raw_text.strip(),
            "category": "other",
            "metadata": {
                "document_id": doc_ref,
                "date": doc_date,
                "due_date": doc_date,
                "total_amount": "0.00",
                "currency": "USD",
                "vendor_name": vendor_name,
                "customer_name": cust_name
            }
        }

    def generate_dataset(self, samples_per_class: int = 100) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Generates balanced dataset for all 4 classes and splits into 80% train / 20% test.
        """
        all_samples = []
        for i in range(samples_per_class):
            all_samples.append(self.generate_invoice_sample(i))
            all_samples.append(self.generate_receipt_sample(i))
            all_samples.append(self.generate_form_sample(i))
            all_samples.append(self.generate_other_sample(i))
            
        random.shuffle(all_samples)
        
        split_idx = int(len(all_samples) * 0.8)
        train_data = all_samples[:split_idx]
        test_data = all_samples[split_idx:]
        
        return train_data, test_data


if __name__ == "__main__":
    generator = EnterpriseDatasetGenerator()
    train_set, test_set = generator.generate_dataset(samples_per_class=50)
    print(f"Generated {len(train_set)} train samples and {len(test_set)} test samples.")
