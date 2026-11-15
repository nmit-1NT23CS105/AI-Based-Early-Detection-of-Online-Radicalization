const COLORS = {
    normal: "#2BB673",
    suspicious: "#F59E0B",
    radical: "#E34F4F",
};

const labelTitle = (value) => {
    if (!value) return "Unknown";
    return value
        .toString()
        .replaceAll("_", " ")
        .replace(/\b\w/g, (match) => match.toUpperCase());
};

const parseJSONScript = (id) => {
    const element = document.getElementById(id);
    if (!element) return {};
    try {
        return JSON.parse(element.textContent);
    } catch {
        return {};
    }
};

const state = {
    summary: parseJSONScript("initial-summary"),
    metrics: parseJSONScript("initial-metrics"),
    metadata: parseJSONScript("initial-metadata"),
    alerts: parseJSONScript("initial-alerts"),
};

const dom = {
    totalAnalyses: document.getElementById("totalAnalyses"),
    highRiskCount: document.getElementById("highRiskCount"),
    averageRisk: document.getElementById("averageRisk"),
    datasetSize: document.getElementById("datasetSize"),
    predictionLabel: document.getElementById("predictionLabel"),
    predictionBadge: document.getElementById("predictionBadge"),
    confidenceValue: document.getElementById("confidenceValue"),
    emotionValue: document.getElementById("emotionValue"),
    riskLevelValue: document.getElementById("riskLevelValue"),
    riskScoreValue: document.getElementById("riskScoreValue"),
    riskMeterBar: document.getElementById("riskMeterBar"),
    highlightedText: document.getElementById("highlightedText"),
    featureChips: document.getElementById("featureChips"),
    pieChart: document.getElementById("pieChart"),
    pieCenterLabel: document.getElementById("pieCenterLabel"),
    probabilityLegend: document.getElementById("probabilityLegend"),
    probabilityBars: document.getElementById("probabilityBars"),
    metricAccuracy: document.getElementById("metricAccuracy"),
    metricPrecision: document.getElementById("metricPrecision"),
    metricRecall: document.getElementById("metricRecall"),
    metricF1: document.getElementById("metricF1"),
    distributionStack: document.getElementById("distributionStack"),
    confusionMatrix: document.getElementById("confusionMatrix"),
    historyTableBody: document.getElementById("historyTableBody"),
    alertFeed: document.getElementById("alertFeed"),
    batchSummary: document.getElementById("batchSummary"),
    batchResultsShell: document.getElementById("batchResultsShell"),
    batchResultsBody: document.getElementById("batchResultsBody"),
    retrainButton: document.getElementById("retrainButton"),
};

const cell = (content, header = false) => {
    const element = document.createElement("div");
    element.className = `confusion-cell${header ? " header" : ""}`;
    element.textContent = content;
    return element;
};

const renderMetricCards = (metrics) => {
    if (!dom.metricAccuracy || !dom.metricPrecision || !dom.metricRecall || !dom.metricF1) return;
    dom.metricAccuracy.textContent = (metrics.accuracy ?? 0).toFixed(3);
    dom.metricPrecision.textContent = (metrics.precision ?? 0).toFixed(3);
    dom.metricRecall.textContent = (metrics.recall ?? 0).toFixed(3);
    dom.metricF1.textContent = (metrics.f1_score ?? 0).toFixed(3);
};

const renderDistribution = (distribution = {}) => {
    if (!dom.distributionStack) return;
    const total = Object.values(distribution).reduce((sum, value) => sum + value, 0);
    dom.distributionStack.innerHTML = "";
    ["normal", "suspicious", "radical"].forEach((label) => {
        const count = distribution[label] ?? 0;
        const ratio = total ? (count / total) * 100 : 0;
        const row = document.createElement("div");
        row.className = "distribution-row";
        row.innerHTML = `
            <div class="distribution-label">
                <span>${labelTitle(label)}</span>
                <strong>${count}</strong>
            </div>
            <div class="distribution-track">
                <div class="distribution-fill" style="width:${ratio}%; background:${COLORS[label]}"></div>
            </div>
        `;
        dom.distributionStack.appendChild(row);
    });
};

const renderConfusionMatrix = (metrics = {}) => {
    if (!dom.confusionMatrix) return;
    const labels = metrics.labels ?? ["normal", "suspicious", "radical"];
    const matrix = metrics.confusion_matrix ?? [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ];
    dom.confusionMatrix.innerHTML = "";
    dom.confusionMatrix.appendChild(cell("Actual / Predicted", true));
    labels.forEach((label) => dom.confusionMatrix.appendChild(cell(labelTitle(label), true)));
    matrix.forEach((row, rowIndex) => {
        dom.confusionMatrix.appendChild(cell(labelTitle(labels[rowIndex]), true));
        row.forEach((value) => dom.confusionMatrix.appendChild(cell(value)));
    });
};

const renderAlerts = (alerts = []) => {
    if (!dom.alertFeed) return;
    dom.alertFeed.innerHTML = "";
    if (!alerts.length) {
        const empty = document.createElement("div");
        empty.className = "alert-item low";
        empty.textContent = "No active alerts.";
        dom.alertFeed.appendChild(empty);
        return;
    }

    alerts.forEach((alert) => {
        const item = document.createElement("article");
        item.className = `alert-item ${alert.severity === "high" ? "" : "low"}`;
        item.innerHTML = `
            <strong>${labelTitle(alert.severity)} Alert</strong>
            <p>${alert.message}</p>
            <small>${new Date(alert.created_at).toLocaleString()}</small>
        `;
        dom.alertFeed.appendChild(item);
    });
};

const renderHistory = (history = []) => {
    dom.historyTableBody.innerHTML = "";
    history.forEach((row) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${new Date(row.created_at).toLocaleString()}</td>
            <td>${row.text_preview}</td>
            <td>${labelTitle(row.language)}</td>
            <td>${row.predicted_label}</td>
            <td>${(row.confidence * 100).toFixed(1)}%</td>
            <td>${row.risk_score}</td>
            <td>${labelTitle(row.emotion_label)}</td>
        `;
        dom.historyTableBody.appendChild(tr);
    });
};

const renderProbabilityView = (probabilities = {}, predictedLabel = "") => {
    const labels = ["normal", "suspicious", "radical"];
    let currentAngle = 0;
    const segments = labels.map((label) => {
        const value = probabilities[label] ?? 0;
        const nextAngle = currentAngle + value * 360;
        const segment = `${COLORS[label]} ${currentAngle.toFixed(1)}deg ${nextAngle.toFixed(1)}deg`;
        currentAngle = nextAngle;
        return segment;
    });

    dom.pieChart.style.background = `conic-gradient(${segments.join(", ")})`;
    dom.pieCenterLabel.textContent = labelTitle(predictedLabel);
    dom.probabilityLegend.innerHTML = "";
    dom.probabilityBars.innerHTML = "";

    labels.forEach((label) => {
        const value = probabilities[label] ?? 0;

        const legend = document.createElement("div");
        legend.className = "legend-item";
        legend.innerHTML = `
            <span><i class="legend-swatch" style="background:${COLORS[label]}"></i>${labelTitle(label)}</span>
            <strong>${(value * 100).toFixed(1)}%</strong>
        `;
        dom.probabilityLegend.appendChild(legend);

        const bar = document.createElement("div");
        bar.className = "bar-row";
        bar.innerHTML = `
            <div class="distribution-label">
                <span>${labelTitle(label)}</span>
                <strong>${(value * 100).toFixed(1)}%</strong>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width:${value * 100}%; background:${COLORS[label]}"></div>
            </div>
        `;
        dom.probabilityBars.appendChild(bar);
    });
};

const renderExplanation = (analysis) => {
    dom.highlightedText.classList.remove("empty-state");
    dom.highlightedText.innerHTML = analysis.explanation.highlighted_html;
    dom.featureChips.innerHTML = "";

    analysis.explanation.top_features.forEach((item) => {
        const chip = document.createElement("div");
        chip.className = "feature-chip";
        chip.innerHTML = `
            <strong>${item.term}</strong>
            <span>${item.direction} / ${item.weight.toFixed(3)}</span>
        `;
        dom.featureChips.appendChild(chip);
    });
};

const renderSingleAnalysis = (analysis) => {
    const prediction = analysis.prediction;
    const risk = analysis.risk;

    dom.predictionLabel.textContent = prediction.label;
    dom.predictionBadge.textContent = prediction.label;
    dom.predictionBadge.className = "status-badge";
    dom.predictionBadge.style.background = `${prediction.color}20`;
    dom.predictionBadge.style.color = prediction.color;
    dom.predictionBadge.style.border = `1px solid ${prediction.color}55`;
    dom.confidenceValue.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;
    dom.emotionValue.textContent = labelTitle(analysis.emotion.label);
    dom.riskLevelValue.textContent = risk.level;
    dom.riskScoreValue.textContent = `${risk.score} / 100`;
    dom.riskMeterBar.style.width = `${risk.score}%`;

    renderProbabilityView(prediction.probabilities, prediction.label_key);
    renderExplanation(analysis);
};

const renderBatchResults = (payload) => {
    dom.batchSummary.classList.remove("hidden");
    dom.batchResultsShell.classList.remove("hidden");
    dom.batchSummary.innerHTML = `
        <strong>${payload.summary.items} items</strong>
        <span>Avg risk ${payload.summary.average_risk}</span>
        <span>High risk ${payload.summary.high_risk}</span>
    `;

    dom.batchResultsBody.innerHTML = "";
    payload.results.forEach((item, index) => {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${item.prediction.label}</td>
            <td>${(item.prediction.confidence * 100).toFixed(1)}%</td>
            <td>${item.risk.score} (${item.risk.level})</td>
            <td>${labelTitle(item.emotion.label)}</td>
            <td>${item.anonymized_text.slice(0, 90)}</td>
        `;
        dom.batchResultsBody.appendChild(row);
    });
};

const renderSummary = (summary) => {
    if (dom.totalAnalyses) dom.totalAnalyses.textContent = summary.total_analyses ?? 0;
    if (dom.highRiskCount) dom.highRiskCount.textContent = summary.high_risk_count ?? 0;
    if (dom.averageRisk) dom.averageRisk.textContent = summary.average_risk ?? 0;
    renderDistribution(summary.label_distribution);
    renderHistory(summary.history ?? []);
    renderAlerts(summary.alerts ?? []);
};

const refreshDashboard = async () => {
    const response = await fetch("/api/dashboard");
    const payload = await response.json();
    state.summary = payload.summary;
    state.metrics = payload.metrics;
    state.metadata = payload.metadata;
    if (dom.datasetSize) dom.datasetSize.textContent = payload.metadata.dataset_size ?? 0;
    renderSummary(payload.summary);
    renderMetricCards(payload.metrics);
    renderConfusionMatrix(payload.metrics);
};

const postJSON = async (url, body) => {
    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) {
        throw new Error(payload.detail || "Request failed");
    }
    return payload;
};

const bindTabs = () => {
    const buttons = document.querySelectorAll(".tab-button");
    const panels = document.querySelectorAll(".tab-panel");
    buttons.forEach((button) => {
        button.addEventListener("click", () => {
            buttons.forEach((item) => item.classList.remove("active"));
            panels.forEach((item) => item.classList.remove("active"));
            button.classList.add("active");
            document.getElementById(`${button.dataset.tab}-tab`).classList.add("active");
        });
    });
};

const bindForms = () => {
    document.getElementById("singleForm").addEventListener("submit", async (event) => {
        event.preventDefault();
        const text = document.getElementById("singleText").value.trim();
        if (!text) return;

        try {
            const payload = await postJSON("/api/analyze", { text });
            renderSingleAnalysis(payload.analysis);
            renderSummary(payload.dashboard);
        } catch (error) {
            alert(error.message);
        }
    });

    document.getElementById("batchForm").addEventListener("submit", async (event) => {
        event.preventDefault();
        const texts = document
            .getElementById("batchTexts")
            .value.split("\n")
            .map((item) => item.trim())
            .filter(Boolean);
        if (!texts.length) return;

        try {
            const payload = await postJSON("/api/analyze-batch", { texts });
            renderBatchResults(payload);
            if (payload.results?.length) {
                renderSingleAnalysis(payload.results[0]);
            }
            renderSummary(payload.dashboard);
        } catch (error) {
            alert(error.message);
        }
    });
};

const bindRetrain = () => {
    dom.retrainButton.addEventListener("click", async () => {
        dom.retrainButton.disabled = true;
        dom.retrainButton.textContent = "Retraining...";
        try {
            await postJSON("/api/retrain", {});
            await refreshDashboard();
            dom.retrainButton.textContent = "Retrain";
        } catch (error) {
            alert(error.message);
            dom.retrainButton.textContent = "Retrain";
        } finally {
            dom.retrainButton.disabled = false;
        }
    });
};

const connectAlerts = () => {
    const source = new EventSource("/api/alerts/stream");
    source.onmessage = (event) => {
        try {
            const payload = JSON.parse(event.data);
            renderAlerts(payload.alerts ?? []);
        } catch {
            return;
        }
    };
};

const init = () => {
    renderSummary(state.summary);
    renderMetricCards(state.metrics);
    renderConfusionMatrix(state.metrics);
    bindTabs();
    bindForms();
    bindRetrain();
    connectAlerts();
};

init();
