// static/app.js
document.addEventListener('DOMContentLoaded', () => {
    // --- Get all necessary DOM elements ---
    const text = document.getElementById('text');
    const predictBtn = document.getElementById('predictBtn');
    const clearBtn = document.getElementById('clearBtn');

    // Result display elements
    const resultCard = document.getElementById('resultCard');
    const predLabel = document.getElementById('predLabel');
    const safety = document.getElementById('safety');
    const probaList = document.getElementById('probaList');

    // Explanation display elements
    const explanationCard = document.getElementById('explanationCard');
    const explanationList = document.getElementById('explanationList');

    // Error display elements
    const errorBanner = document.getElementById('errorBanner');
    const errorMessage = document.getElementById('errorMessage');

    // --- code carbon emission display element ---
    const emissionsDataContainer = document.getElementById('emissionsData');

    // Performance metrics display elements
    const accuracyEl = document.getElementById("metric-accuracy");
    const f1ScoreEl = document.getElementById("metric-f1-score");


    // --- Helper functions for UI updates ---
    const showLoader = (isLoading) => {
        predictBtn.disabled = isLoading;
        predictBtn.textContent = isLoading ? 'Predicting...' : 'Predict';
    };

    const showError = (message) => {
        errorMessage.textContent = message;
        errorBanner.classList.remove('hidden');
    };

    const hideError = () => {
        errorBanner.classList.add('hidden');
    };

    const clearResults = () => {
        text.value = '';
        resultCard.classList.add('hidden');
        explanationCard.classList.add('hidden');
        predLabel.textContent = '';
        safety.textContent = '';
        probaList.innerHTML = '';
        explanationList.innerHTML = '';
        hideError();
    };

    // --- Function to fetch and display CodeCarbon emissions data ---
    async function loadEmissionsData() {
        try {
            const resp = await fetch('/static/emissions_data.json');
            if (!resp.ok) throw new Error('Could not fetch emissions data.');

            const data = await resp.json();

            if (data.error) {
                emissionsDataContainer.innerHTML = `<p class="col-span-2 text-red-600">${data.error}</p>`;
                return;
            }

            const labels = {
                emissions_g: 'CO₂ Emissions',
                duration_seconds: 'Training Duration',
                energy_consumed_kwh: 'Energy Consumed',
                country_name: 'Location',
            };

            emissionsDataContainer.innerHTML = '';
            for (const key in labels) {
                if (data[key]) {
                    const labelDiv = document.createElement('div');
                    labelDiv.className = 'font-medium text-gray-600';
                    labelDiv.textContent = labels[key];
                    emissionsDataContainer.appendChild(labelDiv);

                    const valueDiv = document.createElement('div');
                    valueDiv.className = 'font-mono text-gray-800';
                    valueDiv.textContent = data[key];
                    emissionsDataContainer.appendChild(valueDiv);
                }
            }
        } catch (e) {
            console.error("CodeCarbon Load Error:", e);
            emissionsDataContainer.innerHTML = `<p class="col-span-2 text-red-600">Could not load emissions data. Check console for details.</p>`;
        }
    }

    /**
     * Loads and displays the model performance metrics from a JSON file.
     */
    const loadPerformanceMetrics = async () => {
        if (!accuracyEl || !f1ScoreEl) return;
        try {
            // FIX: Add a cache-busting query parameter
            const response = await fetch(`/static/performance_metrics.json?t=${new Date().getTime()}`);
            if (!response.ok) {
                throw new Error(`Metrics file not found (Status: ${response.status})`);
            }
            const data = await response.json();

            if (data.accuracy === undefined || data.macro_f1_score === undefined) {
                throw new Error('Metrics JSON is missing required keys.');
            }

            accuracyEl.textContent = (data.accuracy * 100).toFixed(2) + ' %';
            f1ScoreEl.textContent = data.macro_f1_score.toFixed(5);
        } catch (error) {
            console.error("Critical error loading performance metrics:", error);
            accuracyEl.textContent = "Failed";
            f1ScoreEl.textContent = "Failed";
            // Also display a user-facing error
            showError(`Could not load model performance metrics. ${error.message}`);
        }
    };

    // --- Main prediction logic ---
    async function doPredict() {
        const val = (text.value || '').trim();
        if (!val) {
            showError('Please enter text before predicting.');
            return;
        }

        hideError();
        showLoader(true);

        try {
            const resp = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: val })
            });
            const data = await resp.json();

            if (!resp.ok) {
                throw new Error(data.error || 'Prediction failed with an unknown error.');
            }

            predLabel.textContent = data.label || '—';
            safety.textContent = data.safety || '';
            if (data.proba && typeof data.proba === 'object') {
                probaList.innerHTML = Object.entries(data.proba)
                    .map(([k, v]) => `<div>${k}: <span class="font-semibold">${Math.round(v * 100)}%</span></div>`)
                    .join('');
            } else {
                probaList.textContent = 'No probabilities available';
            }
            resultCard.classList.remove('hidden');

            if (data.explanation && Array.isArray(data.explanation)) {
                explanationList.innerHTML = '';
                data.explanation.forEach(([word, weight]) => {
                    const span = document.createElement('span');
                    span.textContent = word;
                    const bgColor = weight > 0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800';
                    span.className = `px-2 py-1 rounded-md text-sm font-mono ${bgColor}`;
                    explanationList.appendChild(span);
                });
                explanationCard.classList.remove('hidden');
            } else {
                explanationCard.classList.add('hidden');
            }

        } catch (e) {
            showError(e.message);
        } finally {
            showLoader(false);
        }
    }

    // --- Attach event listeners ---
    predictBtn.addEventListener('click', doPredict);
    clearBtn.addEventListener('click', clearResults);
    text.addEventListener('input', hideError);

    // --- FIX 2: Call the function to load the data when the page loads ---
    loadEmissionsData();
    // -- call metrics
    loadPerformanceMetrics()
});
