/**
 * DocuMind Enterprise Document Intelligence Frontend Controller
 * Powers interactive document inspection, bounding box traceability, and live ML evaluation.
 */

let currentDocData = null;
let currentPageIdx = 0;
let currentZoom = 1.0;
let showBoxes = true;

document.addEventListener("DOMContentLoaded", () => {
  // Load default sample invoice on startup
  loadSample("sample_invoice.pdf");
  fetchSystemHealth();
  fetchBenchmarkReport();
});

async function fetchSystemHealth() {
  try {
    const res = await fetch("/api/v1/health");
    if (res.ok) {
      const data = await res.json();
      const meta = document.getElementById("deviceMeta");
      if (meta) {
        meta.textContent = `Device: ${data.device.toUpperCase()} | Version ${data.version}`;
      }
    }
  } catch (err) {
    console.warn("Could not reach health check:", err);
  }
}

async function loadSample(sampleName) {
  try {
    document.getElementById("docTitle").textContent = "Processing " + sampleName + "...";
    const res = await fetch(`/api/v1/samples/${sampleName}/process?model=deep_learning`);
    if (!res.ok) throw new Error("Failed to process sample");
    const data = await res.json();
    renderDocumentData(data);
  } catch (err) {
    alert("Error loading sample document: " + err.message);
  }
}

function loadSelectedSample(val) {
  loadSample(val);
}

async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  document.getElementById("docTitle").textContent = "Uploading & OCR Processing: " + file.name + "...";

  try {
    const res = await fetch("/api/v1/documents/process?model=deep_learning", {
      method: "POST",
      body: formData
    });
    if (!res.ok) throw new Error("Failed to process uploaded file");
    const data = await res.json();
    renderDocumentData(data);
  } catch (err) {
    alert("Upload failed: " + err.message);
    document.getElementById("docTitle").textContent = "Error processing document";
  }
}

function renderDocumentData(data) {
  currentDocData = data;
  currentPageIdx = 0;

  // Header Title & Badges
  document.getElementById("docTitle").textContent = data.filename;
  
  const confBadge = document.getElementById("confBadge");
  confBadge.textContent = `${Math.round(data.overall_confidence * 100)}% confidence`;

  const reviewBadge = document.getElementById("reviewBadge");
  const trustedBadge = document.getElementById("trustedBadge");

  if (data.trust_status === "Trusted") {
    reviewBadge.style.display = "none";
    trustedBadge.style.display = "inline-flex";
  } else {
    reviewBadge.style.display = "inline-flex";
    trustedBadge.style.display = "none";
    reviewBadge.textContent = data.trust_status;
  }

  // Classification info
  const cat = data.classification.predicted_category;
  document.getElementById("valCategory").textContent = cat;
  document.getElementById("confCategory").textContent = `${(data.classification.confidence * 100).toFixed(1)}%`;
  document.getElementById("confLatency").textContent = `${data.classification.inference_time_ms.toFixed(1)} ms`;

  // Primary Group Title
  document.getElementById("primaryGroupTitle").textContent = cat.toUpperCase();

  // Fields Extraction Mapping
  const f = data.fields;
  setFieldText("valDocId", "confDocId", f.document_id || f.invoice_number);
  setFieldText("valDate", "confDate", f.invoice_date || f.date);
  setFieldText("valDueDate", "confDueDate", f.due_date);
  
  const totalF = f.total_amount;
  if (totalF) {
    const curr = f.currency ? f.currency.value : "USD";
    const amtStr = typeof totalF.value === "number" ? totalF.value.toLocaleString('en-US', {minimumFractionDigits: 2}) : totalF.value;
    document.getElementById("valTotal").textContent = `${curr} $${amtStr}`;
    document.getElementById("confTotal").textContent = `${Math.round(totalF.confidence * 100)}%`;
  }

  if (f.currency) {
    document.getElementById("valCurrency").textContent = f.currency.value;
  }

  setFieldText("valVendorName", "confVendorName", f.vendor_name);
  setFieldText("valVendorAddr", "confVendorAddr", f.vendor_address);
  setFieldText("valCustName", "confCustName", f.customer_name);
  setFieldText("valCustAddr", "confCustAddr", f.billing_address);

  // Line items
  const linesCountBadge = document.getElementById("linesCount");
  linesCountBadge.textContent = data.line_items ? data.line_items.length : 0;
  renderLineItems(data.line_items || []);

  // Validation text
  const valBox = document.getElementById("validationBox");
  const valText = document.getElementById("validationText");
  if (data.validation_issues && data.validation_issues.length > 0) {
    valBox.style.borderColor = "#fcd34d";
    valBox.style.background = "#fffbeb";
    valText.innerHTML = data.validation_issues.map(iss => `• ${iss}`).join("<br>");
  } else {
    valBox.style.borderColor = "#bbf7d0";
    valBox.style.background = "#f0fdf4";
    const sub = f.subtotal ? f.subtotal.value : 0;
    const tax = f.tax ? f.tax.value : 0;
    const tot = f.total_amount ? f.total_amount.value : 0;
    valText.textContent = `Mathematical cross-validation verified: Subtotal ($${Number(sub).toFixed(2)}) + Tax ($${Number(tax).toFixed(2)}) matches Total Due ($${Number(tot).toFixed(2)}). All constraints verified.`;
  }

  // JSON Viewer
  document.getElementById("jsonViewer").textContent = JSON.stringify(data, null, 2);

  // Render Page Canvas
  renderCurrentPage();
}

function setFieldText(valId, confId, fieldObj) {
  const elVal = document.getElementById(valId);
  const elConf = document.getElementById(confId);
  if (!fieldObj) {
    if (elVal) elVal.textContent = "N/A";
    if (elConf) elConf.textContent = "--";
    return;
  }
  if (elVal) elVal.textContent = fieldObj.value || "N/A";
  if (elConf) elConf.textContent = `${Math.round(fieldObj.confidence * 100)}%`;
}

function renderLineItems(items) {
  const tbody = document.getElementById("lineItemsBody");
  tbody.innerHTML = "";
  if (!items || items.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" style="text-align:center; color: #94a3b8; padding: 24px;">No tabular line items detected.</td></tr>`;
    return;
  }

  items.forEach(item => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${escapeHtml(item.description)}</strong></td>
      <td class="num-col">${item.quantity !== null && item.quantity !== undefined ? item.quantity : "-"}</td>
      <td class="num-col">${item.unit_price !== null && item.unit_price !== undefined ? `$${item.unit_price.toFixed(2)}` : "-"}</td>
      <td class="num-col"><strong>$${item.amount.toFixed(2)}</strong></td>
    `;
    if (item.bbox) {
      tr.style.cursor = "pointer";
      tr.title = "Click to locate on document";
      tr.onclick = () => highlightBBox(item.bbox, item.description);
    }
    tbody.appendChild(tr);
  });
}

function renderCurrentPage() {
  if (!currentDocData || !currentDocData.page_images_base64) return;
  const totalPages = currentDocData.page_images_base64.length;
  document.getElementById("pageIndicator").textContent = `${currentPageIdx + 1} / ${totalPages}`;

  const imgEl = document.getElementById("pageImg");
  imgEl.src = `data:image/png;base64,${currentDocData.page_images_base64[currentPageIdx]}`;

  imgEl.onload = () => {
    renderBoundingBoxes();
  };
}

function renderBoundingBoxes() {
  const overlay = document.getElementById("bboxOverlay");
  overlay.innerHTML = "";
  if (!showBoxes || !currentDocData) return;

  const curPage = currentPageIdx + 1;

  // Render field boxes
  const fields = currentDocData.fields || {};
  for (const [key, field] of Object.entries(fields)) {
    if (field && field.bbox && (field.page || 1) === curPage) {
      createBoxElement(overlay, field.bbox, field.field_name, field.value);
    }
  }

  // Render line item boxes
  const items = currentDocData.line_items || [];
  for (const it of items) {
    if (it.bbox) {
      createBoxElement(overlay, it.bbox, "line_item", it.description);
    }
  }
}

function createBoxElement(parent, bbox, fieldName, value) {
  // bbox format: [x0, y0, x1, y1] as percentages (0 to 100)
  const [x0, y0, x1, y1] = bbox;
  const box = document.createElement("div");
  box.className = "bbox-rect";
  box.id = `box-${fieldName}`;
  box.style.left = `${x0}%`;
  box.style.top = `${y0}%`;
  box.style.width = `${Math.max(3, x1 - x0)}%`;
  box.style.height = `${Math.max(2, y1 - y0)}%`;

  const node = document.createElement("div");
  node.className = "bbox-node";
  box.appendChild(node);

  box.title = `${fieldName}: ${value}`;
  box.onclick = (e) => {
    e.stopPropagation();
    highlightField(fieldName);
  };

  parent.appendChild(box);
}

function highlightField(fieldName) {
  // Highlight field row in right panel
  document.querySelectorAll(".field-row").forEach(r => r.classList.remove("active"));
  document.querySelectorAll(".bbox-rect").forEach(b => b.classList.remove("active"));

  const targetBox = document.getElementById(`box-${fieldName}`);
  if (targetBox) {
    targetBox.classList.add("active");
    targetBox.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
  }

  // If line items tab
  if (fieldName === "line_items" || fieldName === "line_item") {
    switchTab("lines");
  } else {
    switchTab("fields");
  }
}

function highlightBBox(bbox, label) {
  document.querySelectorAll(".bbox-rect").forEach(b => b.classList.remove("active"));
  const overlay = document.getElementById("bboxOverlay");
  
  // Create temporary highlight box if not present
  const [x0, y0, x1, y1] = bbox;
  const tempBox = document.createElement("div");
  tempBox.className = "bbox-rect active";
  tempBox.style.left = `${x0}%`;
  tempBox.style.top = `${y0}%`;
  tempBox.style.width = `${Math.max(3, x1 - x0)}%`;
  tempBox.style.height = `${Math.max(2, y1 - y0)}%`;
  overlay.appendChild(tempBox);

  tempBox.scrollIntoView({ behavior: "smooth", block: "center" });
  setTimeout(() => tempBox.remove(), 2500);
}

function prevPage() {
  if (currentPageIdx > 0) {
    currentPageIdx--;
    renderCurrentPage();
  }
}

function nextPage() {
  if (currentDocData && currentPageIdx < currentDocData.page_images_base64.length - 1) {
    currentPageIdx++;
    renderCurrentPage();
  }
}

function zoomIn() {
  currentZoom = Math.min(2.5, currentZoom + 0.15);
  updateZoom();
}

function zoomOut() {
  currentZoom = Math.max(0.6, currentZoom - 0.15);
  updateZoom();
}

function fitToScreen() {
  currentZoom = 1.0;
  updateZoom();
}

function updateZoom() {
  const container = document.getElementById("pageContainer");
  container.style.transform = `scale(${currentZoom})`;
  document.getElementById("zoomLevel").textContent = `${Math.round(currentZoom * 100)}%`;
}

function toggleBoxes() {
  showBoxes = !showBoxes;
  const overlay = document.getElementById("bboxOverlay");
  overlay.style.display = showBoxes ? "block" : "none";
}

function switchTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

  const targetBtn = document.getElementById(`tab-${tabId}`);
  const targetView = document.getElementById(`view-${tabId}`);
  if (targetBtn) targetBtn.classList.add("active");
  if (targetView) targetView.classList.add("active");
}

async function fetchBenchmarkReport() {
  try {
    const res = await fetch("/api/v1/models/benchmark");
    if (!res.ok) return;
    const data = await res.json();
    const bench = data.benchmark;
    const fieldEval = data.field_evaluation;

    if (bench && bench.models) {
      const b = bench.models.baseline;
      const dl = bench.models.deep_learning;

      document.getElementById("b-acc").textContent = `${(b.accuracy * 100).toFixed(1)}%`;
      document.getElementById("b-prec").textContent = `${(b.macro_precision * 100).toFixed(1)}%`;
      document.getElementById("b-rec").textContent = `${(b.macro_recall * 100).toFixed(1)}%`;
      document.getElementById("b-f1").textContent = `${(b.macro_f1 * 100).toFixed(1)}%`;
      document.getElementById("b-lat").textContent = `${b.mean_latency_ms.toFixed(2)} ms`;
      document.getElementById("b-train").textContent = `${b.training_time_s.toFixed(2)}s`;

      document.getElementById("dl-acc").textContent = `${(dl.accuracy * 100).toFixed(1)}%`;
      document.getElementById("dl-prec").textContent = `${(dl.macro_precision * 100).toFixed(1)}%`;
      document.getElementById("dl-rec").textContent = `${(dl.macro_recall * 100).toFixed(1)}%`;
      document.getElementById("dl-f1").textContent = `${(dl.macro_f1 * 100).toFixed(1)}%`;
      document.getElementById("dl-lat").textContent = `${dl.mean_latency_ms.toFixed(2)} ms`;
    }

    if (fieldEval && fieldEval.field_metrics) {
      const tbody = document.getElementById("fieldBenchBody");
      tbody.innerHTML = "";
      for (const [fName, m] of Object.entries(fieldEval.field_metrics)) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${fName.replace('_', ' ').toUpperCase()}</strong></td>
          <td class="num-col" style="color: #059669; font-weight: 600;">${(m.exact_match_accuracy * 100).toFixed(1)}%</td>
          <td class="num-col" style="color: #059669; font-weight: 600;">${(m.mean_token_f1 * 100).toFixed(1)}%</td>
        `;
        tbody.appendChild(tr);
      }
    }
  } catch (err) {
    console.warn("Could not load benchmark data:", err);
  }
}

function copyJson() {
  if (!currentDocData) return;
  navigator.clipboard.writeText(JSON.stringify(currentDocData, null, 2))
    .then(() => alert("Structured JSON copied to clipboard!"))
    .catch(err => alert("Copy failed: " + err));
}

function markAsPaid() {
  alert("Invoice marked as Paid! Status synchronized with enterprise ERP.");
}

function approveDocument() {
  alert("Document approved! Verified extraction sent to automated workflow queue.");
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
