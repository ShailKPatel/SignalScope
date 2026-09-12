// SignalScope App Client JS

let currentResultData = null;
let currentBatchData = null;

document.addEventListener('DOMContentLoaded', () => {
  setupDragAndDrop();
});

function setupDragAndDrop() {
  const dropZone = document.getElementById('dropZone');
  
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('drag-over');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-over');
    }, false);
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      processFiles(files);
    }
  });
}

function handleFileSelect(event) {
  const files = event.target.files;
  if (files && files.length > 0) {
    processFiles(files);
  }
}

async function processFiles(fileList) {
  const files = Array.from(fileList).filter(f => f.type.startsWith('image/'));
  if (files.length === 0) {
    alert('Please select valid image files (JPEG, PNG, WebP).');
    return;
  }

  const captionText = document.getElementById('captionInput').value;

  if (files.length === 1) {
    // Single file upload
    await uploadSingleFile(files[0], captionText);
    document.getElementById('batchSection').style.display = 'none';
  } else {
    // Bulk / Folder upload
    await uploadBatchFiles(files);
  }
}

async function uploadSingleFile(file, caption) {
  const formData = new FormData();
  formData.append('file', file);
  if (caption) {
    formData.append('caption', caption);
  }

  try {
    // Show local preview immediately
    const reader = new FileReader();
    reader.onload = (e) => {
      document.getElementById('baseImage').src = e.target.result;
    };
    reader.readAsDataURL(file);

    const resp = await fetch('/api/predict', {
      method: 'POST',
      body: formData
    });
    
    if (!resp.ok) throw new Error('Prediction API failed');

    const data = await resp.json();
    currentResultData = data;
    renderSingleResult(data);
  } catch (err) {
    console.error('Error analyzing image:', err);
    alert('Error connecting to SignalScope backend service: ' + err.message);
  }
}

async function uploadBatchFiles(files) {
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));

  try {
    const resp = await fetch('/api/predict-batch', {
      method: 'POST',
      body: formData
    });

    if (!resp.ok) throw new Error('Batch API failed');

    const batchData = await resp.json();
    currentBatchData = batchData;
    renderBatchResult(batchData);

    // Show first image in detail view
    if (batchData.results && batchData.results.length > 0) {
      currentResultData = batchData.results[0];
      // Load image source into viewer
      const firstFile = files[0];
      const reader = new FileReader();
      reader.onload = (e) => {
        document.getElementById('baseImage').src = e.target.result;
      };
      reader.readAsDataURL(firstFile);
      renderSingleResult(currentResultData);
    }
  } catch (err) {
    console.error('Error processing batch:', err);
    alert('Error processing batch upload: ' + err.message);
  }
}

function renderSingleResult(data) {
  document.getElementById('resultsGrid').style.display = 'grid';

  const verdict = data.verdict;
  const modules = data.modules;

  // Banner & Score
  const banner = document.getElementById('verdictBanner');
  const label = document.getElementById('verdictLabel');
  const score = document.getElementById('verdictScore');

  label.innerText = verdict.label.toUpperCase();
  score.innerText = `${(verdict.confidence_score * 100).toFixed(1)}%`;
  document.getElementById('imageResolution').innerText = data.resolution;

  if (verdict.is_ai_generated) {
    banner.className = 'verdict-banner ai';
  } else {
    banner.className = 'verdict-banner real';
  }

  // Module A: Heatmap & Cues
  const modA = modules.module_a_explainability;
  document.getElementById('heatmapImage').src = modA.heatmap_base64;
  document.getElementById('explanationSummary').innerText = modA.summary_text;

  const cuesContainer = document.getElementById('visualCuesContainer');
  cuesContainer.innerHTML = modA.cues.map(c => `
    <div class="cue-item">
      <div class="cue-header">
        <span>${c.type}</span>
        <span>Confidence: ${Math.round(c.confidence * 100)}%</span>
      </div>
      <div class="cue-desc">${c.detail}</div>
    </div>
  `).join('');

  // Module B: Attribution
  const modB = modules.module_b_attribution;
  document.getElementById('attrFamily').innerText = modB.family;
  document.getElementById('attrModel').innerText = modB.specific_model;
  
  const famContainer = document.getElementById('familyProbContainer');
  famContainer.innerHTML = Object.entries(modB.family_probabilities).map(([fam, prob]) => `
    <div class="metric-row">
      <span class="metric-label">${fam}</span>
      <span class="metric-val">${(prob * 100).toFixed(1)}%</span>
    </div>
  `).join('');

  // Module C: Robustness
  const modC = modules.module_c_robustness;
  document.getElementById('robustnessRating').innerText = modC.overall_stability_rating;
  document.getElementById('robustnessJpeg').innerText = modC.verdict_preserved_under_jpeg70 ? "Yes (Stable)" : "No";

  const jpegContainer = document.getElementById('jpegCurveContainer');
  jpegContainer.innerHTML = modC.jpeg_degradation_curve.map(row => `
    <div class="metric-row">
      <span class="metric-label">Quality ${row.quality}</span>
      <span class="metric-val">${(row.confidence_retained * 100).toFixed(1)}%</span>
    </div>
  `).join('');

  // Module D: Metadata
  const modD = modules.module_d_metadata;
  document.getElementById('metaHasExif').innerText = modD.has_exif ? "Yes (Valid Tags)" : "No EXIF";
  document.getElementById('metaHasC2pa').innerText = modD.has_c2pa ? "Yes (Signed)" : "No C2PA Manifest";
  document.getElementById('metaCamera').innerText = modD.camera_model ? `${modD.camera_make} ${modD.camera_model}` : "Unknown / Stripped";
  document.getElementById('metaAssessment').innerText = modD.assessment;

  // Module E: Multimodal
  const modE = modules.module_e_multimodal;
  document.getElementById('multiCaption').innerText = modE.caption_text || "None provided";
  document.getElementById('multiScore').innerText = modE.semantic_similarity ? `${(modE.semantic_similarity * 100).toFixed(1)}%` : "N/A";
  document.getElementById('multiAssessment').innerText = modE.assessment || "Skipped";

  // Module G: Active Defense
  const modG = modules.module_g_active_defense;
  const defenseContainer = document.getElementById('activeDefenseContainer');
  defenseContainer.innerHTML = modG.defense_evaluations.map(ev => `
    <div class="cue-item">
      <div class="cue-header">
        <span>${ev.attack_type}</span>
        <span>Retained Accuracy: ${ev.accuracy_retained}</span>
      </div>
      <div class="cue-desc"><strong>Mitigation:</strong> ${ev.mitigation}</div>
    </div>
  `).join('');
}

function renderBatchResult(batchData) {
  document.getElementById('batchSection').style.display = 'block';
  
  const summary = batchData.summary;
  document.getElementById('batchSummary').innerText = 
    `Summary: ${batchData.total_images} files analyzed in ${batchData.total_time_seconds}s | ` +
    `Likely AI-generated: ${summary.likely_ai_generated_count} | Likely Real: ${summary.likely_real_count}`;

  const tbody = document.getElementById('batchTableBody');
  tbody.innerHTML = batchData.results.map((item, idx) => `
    <tr>
      <td>${item.filename}</td>
      <td><span class="${item.verdict.is_ai_generated ? 'pill-ai' : 'pill-real'}">${item.verdict.label}</span></td>
      <td>${(item.verdict.confidence_score * 100).toFixed(1)}%</td>
      <td>${item.modules.module_b_attribution.specific_model}</td>
      <td><button class="btn btn-secondary" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;" onclick="viewBatchDetail(${idx})">Inspect</button></td>
    </tr>
  `).join('');
}

function viewBatchDetail(index) {
  if (currentBatchData && currentBatchData.results[index]) {
    currentResultData = currentBatchData.results[index];
    renderSingleResult(currentResultData);
    window.scrollTo({ top: document.getElementById('resultsGrid').offsetTop - 80, behavior: 'smooth' });
  }
}

function updateHeatmapOpacity(val) {
  document.getElementById('heatmapImage').style.opacity = val / 100;
}

function switchTab(tabId, btnElement) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

  document.getElementById(tabId).classList.add('active');
  btnElement.classList.add('active');
}
