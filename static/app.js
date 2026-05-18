/* ===========================================================
   Knowledge Editing Web App -- Frontend Logic
   =========================================================== */

const API = "";  // same origin

// -- Helpers --
async function post(url, body) {
  const res = await fetch(API + url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

async function get(url) {
  const res = await fetch(API + url);
  return res.json();
}

function show(id)  { document.getElementById(id).classList.remove("hidden"); }
function hide(id)  { document.getElementById(id).classList.add("hidden"); }
function $(id)     { return document.getElementById(id); }

// -- Probability bars renderer --
function renderProbBars(containerId, tokens, probs) {
  const el = $(containerId);
  if (!tokens || !tokens.length) { el.innerHTML = ""; return; }
  let html = "";
  for (let i = 0; i < tokens.length; i++) {
    const pct = Math.min(probs[i] * 100, 100);
    html += `
      <div class="prob-bar-row">
        <span class="prob-bar-token">${escHtml(tokens[i])}</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${pct}%"></div>
        </div>
        <span class="prob-bar-value">${(probs[i] * 100).toFixed(2)}%</span>
      </div>`;
  }
  el.innerHTML = html;
}

// -- Top predictions renderer --
function renderTopPreds(containerId, predictions) {
  const el = $(containerId);
  if (!predictions || !predictions.length) { el.innerHTML = ""; return; }
  let html = "";
  for (const p of predictions) {
    const pct = Math.min(p.prob * 100, 100);
    html += `
      <div class="prob-bar-row">
        <span class="prob-bar-token">${escHtml(p.token)}</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${pct}%"></div>
        </div>
        <span class="prob-bar-value">${(p.prob * 100).toFixed(2)}%</span>
      </div>`;
  }
  el.innerHTML = html;
}

function escHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// -- Status check --
async function checkStatus() {
  try {
    const d = await get("/api/status");
    const dot = $("statusDot");
    const txt = $("statusText");
    $("editCount").textContent = d.edit_count;
    if (d.loaded) {
      dot.className = "status-dot green";
      let statusMsg = d.is_edited ? "Model đã chỉnh sửa" : "Model sẵn sàng";
      if (d.checkpoint_saved) {
        statusMsg += ` | 💾 Checkpoint: ${d.checkpoint_size_mb} MB`;
      }
      txt.textContent = statusMsg;
    } else {
      dot.className = "status-dot red";
      txt.textContent = "Model chưa tải";
    }
    // Hiện/ẩn nút suggest dựa trên API sẵn có
    const suggestBtn = $("btnSuggest");
    if (suggestBtn) {
      suggestBtn.title = d.gemini_ready
        ? "Gợi ý câu trả lời bằng Gemini AI"
        : "Cần đặt GEMINI_API_KEY trong file .env";
      suggestBtn.disabled = !d.gemini_ready && !d.openai_ready;
    }
  } catch {
    $("statusDot").className = "status-dot red";
    $("statusText").textContent = "Server không phản hồi";
  }
}

// -- AI Suggest target_new --
async function doSuggest() {
  const prompt  = $("editPrompt").value.trim();
  const subject = $("editSubject").value.trim();
  if (!prompt || !subject) {
    return alert("Vui lòng nhập Prompt và Subject trước khi gợi ý.");
  }
  const question = `Trả lời ngắn gọn (tối đa 10 từ): ${prompt}`;
  const btn = $("btnSuggest");
  btn.disabled = true;
  btn.textContent = "⏳ Đang hỏi AI...";
  try {
    const d = await post("/api/suggest", { question, provider: "gemini" });
    if (d.error) throw new Error(d.error);
    $("editTargetNew").value = d.answer;
    btn.textContent = "✅ Đã gợi ý!";
    setTimeout(() => { btn.textContent = "🤖 Gợi ý từ AI"; }, 2000);
  } catch (e) {
    alert("Lỗi gợi ý: " + e.message);
    btn.textContent = "🤖 Gợi ý từ AI";
  } finally {
    btn.disabled = false;
  }
}

// -- Explain / Tham khảo thực tế từ Gemini hoặc Tavily --
// promptInputId: id của ô input chứa câu hỏi
// answerElId:    id của phần từ hiển thị answer
// sourceElId:    id của phần từ hiển thị source
// btnId:         id của nút đã nhấn
async function doExplain(promptInputId, answerElId, sourceElId, btnId) {
  const question = $(promptInputId).value.trim();
  if (!question) return alert("Vui lòng nhập prompt trước.");

  const btn = $(btnId);
  btn.disabled = true;
  btn.textContent = "⏳ Đang hỏi AI...";

  // Xác định body id tương ứng
  // btnId pattern: btnGenExplain -> genAiRefBody | btnVerifyExplain -> verifyAiRefBody
  const prefix    = btnId.replace("btn","").replace("Explain","").toLowerCase(); // "gen" | "verify"
  const bodyEl    = $(prefix + "AiRefBody");

  try {
    const d = await post("/api/explain", { question, provider: "auto" });
    if (d.error) throw new Error(d.error);

    $(sourceElId).textContent = d.source ? `🔗 Nguồn: ${d.source}` : "";
    $(answerElId).textContent = d.answer || "(Không có kết quả)";
    bodyEl.classList.remove("hidden");
    btn.textContent = "🔄 Cập nhật";
  } catch (e) {
    $(sourceElId).textContent = "";
    $(answerElId).textContent = "⚠️ Lỗi: " + e.message;
    bodyEl.classList.remove("hidden");
    btn.textContent = "🌐 Hỏi Gemini / Tavily";
  } finally {
    btn.disabled = false;
  }
}


// -- Step 1: Generate --
async function doGenerate() {
  const prompt = $("genPrompt").value.trim();
  if (!prompt) return alert("Vui lòng nhập prompt.");
  const max_new_tokens = parseInt($("genMaxTokens")?.value || "20");

  $("btnGenerate").disabled = true;
  show("genLoading");
  hide("genResult");

  try {
    const d = await post("/api/generate", { prompt, max_new_tokens });
    if (d.error) throw new Error(d.error);
    $("genOutput").textContent = d.generated;
    $("genOutput").className = "result-box";
    renderTopPreds("genTopPreds", d.top_predictions);
    show("genResult");
  } catch (e) {
    $("genOutput").textContent = "Lỗi: " + e.message;
    $("genOutput").className = "result-box error";
    show("genResult");
  } finally {
    hide("genLoading");
    $("btnGenerate").disabled = false;
  }
}

// -- Step 2: Edit --
async function doEdit() {
  const prompt       = $("editPrompt").value.trim();
  const subject      = $("editSubject").value.trim();
  const ground_truth = $("editGroundTruth").value.trim();
  const target_new   = $("editTargetNew").value.trim();

  if (!prompt || !subject || !target_new) {
    return alert("Vui long nhap Prompt, Subject va Target New.");
  }

  $("btnEdit").disabled = true;
  show("editLoading");
  hide("editResult");

  try {
    const d = await post("/api/edit", { prompt, subject, ground_truth, target_new });
    if (d.error) throw new Error(d.error);

    // Before / After text
    $("editBefore").textContent = d.before.generated;
    $("editAfter").textContent  = d.after.generated;

    // Top predictions
    renderTopPreds("topBefore", d.before.top_predictions);
    renderTopPreds("topAfter",  d.after.top_predictions);

    // Target token probability bars
    renderProbBars("probsBefore", d.before.target_tokens, d.before.target_probs);
    renderProbBars("probsAfter",  d.after.target_tokens,  d.after.target_probs);

    // Metrics
    const accBefore = d.metrics.rewrite_acc_before;
    const accAfter  = d.metrics.rewrite_acc_after;
    $("metricAccBefore").textContent = (accBefore * 100).toFixed(0) + "%";
    $("metricAccBefore").className   = "metric-value " + (accBefore >= 0.5 ? "green" : "red");

    $("metricAccAfter").textContent  = (accAfter * 100).toFixed(0) + "%";
    $("metricAccAfter").className    = "metric-value " + (accAfter >= 0.5 ? "green" : "red");

    const avgProb = d.after.target_probs.length
      ? d.after.target_probs.reduce((a,b) => a+b, 0) / d.after.target_probs.length
      : 0;
    $("metricAvgProb").textContent = (avgProb * 100).toFixed(1) + "%";
    $("metricAvgProb").className   = "metric-value cyan";

    show("editResult");

    // Auto-fill the verify prompt
    $("verifyPrompt").value = prompt;
    $("verifyTarget").value = target_new;

    // Update history
    renderHistory();
    checkStatus();

  } catch (e) {
    alert("Edit that bai: " + e.message);
  } finally {
    hide("editLoading");
    $("btnEdit").disabled = false;
  }
}

// -- Step 3: Verify --
async function doVerify() {
  const prompt = $("verifyPrompt").value.trim();
  const target = $("verifyTarget").value.trim();
  if (!prompt) return alert("Vui long nhap prompt.");

  $("btnVerify").disabled = true;
  show("verifyLoading");
  hide("verifyResult");

  try {
    const d = await post("/api/generate", { prompt });
    if (d.error) throw new Error(d.error);

    $("verifyOutput").textContent = d.generated;
    $("verifyOutput").className = "result-box success";
    renderTopPreds("verifyTopPreds", d.top_predictions);

    // If target is provided, also score it
    if (target) {
      const loc = await post("/api/locality", { prompt, expected: target });
      if (loc.target_tokens && loc.target_tokens.length) {
        renderProbBars("verifyProbs", loc.target_tokens, loc.target_probs);
        show("verifyProbSection");
      } else {
        hide("verifyProbSection");
      }
    } else {
      hide("verifyProbSection");
    }

    show("verifyResult");
  } catch (e) {
    $("verifyOutput").textContent = "Loi: " + e.message;
    $("verifyOutput").className = "result-box error";
    show("verifyResult");
  } finally {
    hide("verifyLoading");
    $("btnVerify").disabled = false;
  }
}

// -- Reset model --
async function doReset() {
  if (!confirm("Reset model về gốc?\n⚠️ Tất cả edit và checkpoint sẽ bị XOÁ.")) return;

  $("btnReset").disabled = true;
  try {
    await post("/api/reset", {});
    // Clear UI
    hide("editResult");
    hide("genResult");
    hide("localResult");
    hide("verifyResult");
    $("metricAccBefore").textContent = "--";
    $("metricAccBefore").className = "metric-value";
    $("metricAccAfter").textContent = "--";
    $("metricAccAfter").className = "metric-value";
    $("metricAvgProb").textContent = "--";
    $("metricAvgProb").className = "metric-value";
    $("historyContent").innerHTML = '<p style="color: var(--text-muted); font-size: .88rem;">Chua co lan chinh sua nao.</p>';
    checkStatus();
  } catch (e) {
    alert("Reset that bai: " + e.message);
  } finally {
    $("btnReset").disabled = false;
  }
}

// -- Step 5: Locality --
async function doLocality() {
  const prompt   = $("localPrompt").value.trim();
  const expected = $("localExpected").value.trim();
  if (!prompt) return alert("Vui long nhap prompt.");

  $("btnLocality").disabled = true;
  show("localLoading");
  hide("localResult");

  try {
    const d = await post("/api/locality", { prompt, expected });
    if (d.error) throw new Error(d.error);

    let output = "Generated: " + d.generated;
    if (expected) {
      output += "\nExpected:  " + d.expected;
      output += "\nAvg Prob:  " + (d.avg_prob * 100).toFixed(2) + "%";
    }
    $("localOutput").textContent = output;
    $("localOutput").className = "result-box" + (d.avg_prob > 0.1 ? " success" : "");

    renderTopPreds("localTopPreds", d.top_predictions);

    if (d.target_tokens && d.target_tokens.length) {
      renderProbBars("probsLocality", d.target_tokens, d.target_probs);
    }

    show("localResult");
  } catch (e) {
    $("localOutput").textContent = "Loi: " + e.message;
    $("localOutput").className = "result-box error";
    show("localResult");
  } finally {
    hide("localLoading");
    $("btnLocality").disabled = false;
  }
}

// -- History --
async function renderHistory() {
  try {
    const list = await get("/api/history");
    if (!list.length) return;

    let html = '<table class="history-table">' +
      "<thead><tr>" +
      "<th>#</th><th>Prompt</th><th>Target New</th><th>Acc Before</th><th>Acc After</th><th>Time</th>" +
      "</tr></thead><tbody>";

    list.forEach(function(item, i) {
      const t = item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : "--";
      html += "<tr>" +
        "<td>" + (i + 1) + "</td>" +
        '<td style="font-family:var(--font-mono);font-size:.8rem">' + escHtml(item.prompt) + "</td>" +
        '<td style="color:var(--neon-green)">' + escHtml(item.target_new) + "</td>" +
        "<td>" + (item.metrics.rewrite_acc_before * 100).toFixed(0) + "%</td>" +
        '<td style="color:var(--neon-green);font-weight:700">' + (item.metrics.rewrite_acc_after * 100).toFixed(0) + "%</td>" +
        "<td>" + t + "</td>" +
        "</tr>";
    });

    html += "</tbody></table>";
    $("historyContent").innerHTML = html;
  } catch (e) { /* ignore */ }
}

// -- Init --
document.addEventListener("DOMContentLoaded", function() {
  checkStatus();
});
