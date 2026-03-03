"""
Layout constants for diffviews Gradio UI.

Contains CSS and JavaScript for the visualization interface.
"""

# JavaScript for Plotly click/hover event bridge to Gradio
# Handles both sample point and trajectory point interactions
PLOTLY_HANDLER_JS = r"""
// State
let clickBox = null;
let hoverBox = null;
let hoverTimeout = null;
let lastHoverKey = null;
let currentPlotDiv = null;
let initComplete = false;

// Find textbox inputs (Gradio 6 nesting)
function findTextboxes() {
    if (!clickBox) {
        const c = document.querySelector('#click-data-box');
        if (c) clickBox = c.querySelector('textarea') || c.querySelector('input');
    }
    if (!hoverBox) {
        const h = document.querySelector('#hover-data-box');
        if (h) hoverBox = h.querySelector('textarea') || h.querySelector('input');
    }
    return clickBox && hoverBox;
}

// Send data to Python via textbox
function sendData(box, data) {
    if (!box) return;
    box.value = JSON.stringify(data);
    box.dispatchEvent(new Event('input', { bubbles: true }));
    box.dispatchEvent(new Event('change', { bubbles: true }));
}

// Click handler - immediate (supports 2D and 3D plots)
function handlePlotlyClick(data) {
    if (!data?.points?.length) return;
    const point = data.points[0];
    const payload = {
        pointIndex: point.customdata,
        x: point.x,
        y: point.y,
        curveNumber: point.curveNumber
    };
    // Include z for 3D plots
    if (typeof point.z !== 'undefined') {
        payload.z = point.z;
    }
    sendData(clickBox, payload);
}

// Hover handler - debounced (supports 2D and 3D plots)
function handlePlotlyHover(data) {
    if (!data?.points?.length) return;
    const point = data.points[0];
    const traceName = point.data.name || '';
    const is3D = typeof point.z !== 'undefined';

    // Trajectory point hover (matches trajectory_N, traj_start_N, traj_end_N)
    const trajMatch = traceName.match(/^(?:trajectory|traj_start|traj_end)_(\d+)$/);
    if (trajMatch) {
        const trajIdx = parseInt(trajMatch[1]);
        // For start/end markers, stepIdx comes from customdata or infer from name
        let stepIdx = point.customdata;
        if (traceName.startsWith('traj_start_')) stepIdx = 1;
        if (traceName.startsWith('traj_end_')) stepIdx = -1;  // Signal last step
        const hoverKey = `traj_${trajIdx}_${traceName}_${stepIdx}`;
        if (hoverKey === lastHoverKey) return;
        clearTimeout(hoverTimeout);
        hoverTimeout = setTimeout(() => {
            lastHoverKey = hoverKey;
            const payload = {
                type: 'trajectory',
                trajIdx: trajIdx,
                stepIdx: stepIdx,
                x: point.x,
                y: point.y,
                sigma: point.text
            };
            if (is3D) payload.z = point.z;
            sendData(hoverBox, payload);
        }, 100);
        return;
    }

    // For 3D plots, accept any sigma slice trace (not just curve 0)
    // In 3D mode, traces are named "σ=X.XX" for data points
    const is3DDataTrace = is3D && traceName.startsWith('σ=');
    if (!is3DDataTrace && point.curveNumber !== 0) return;

    const idx = point.customdata;
    const hoverKey = `sample_${idx}`;
    if (hoverKey === lastHoverKey) return;
    clearTimeout(hoverTimeout);
    hoverTimeout = setTimeout(() => {
        lastHoverKey = hoverKey;
        const payload = {
            type: 'sample',
            pointIndex: idx,
            x: point.x,
            y: point.y
        };
        if (is3D) payload.z = point.z;
        sendData(hoverBox, payload);
    }, 100);
}

// Check if Plotly is ready
function isPlotlyReady(div) {
    return div && typeof div.on === 'function' && div.data && div.layout;
}

// Find Plotly div
function findPlotDiv() {
    return document.querySelector('#umap-plot .plotly-graph-div') ||
           document.querySelector('#umap-plot .js-plotly-plot') ||
           document.querySelector('.plotly-graph-div');
}

// Attach handlers (retries indefinitely until success)
function attachPlotlyHandlers() {
    const plotDiv = findPlotDiv();
    if (!plotDiv || !isPlotlyReady(plotDiv) || !findTextboxes()) {
        setTimeout(attachPlotlyHandlers, 300);
        return;
    }

    // Skip if already attached to this element
    if (plotDiv === currentPlotDiv && plotDiv._handlersAttached) return;

    // Clear existing handlers
    try {
        plotDiv.removeAllListeners('plotly_click');
        plotDiv.removeAllListeners('plotly_hover');
    } catch(e) {}

    // Attach
    currentPlotDiv = plotDiv;
    plotDiv._handlersAttached = true;
    initComplete = true;
    plotDiv.on('plotly_click', handlePlotlyClick);
    plotDiv.on('plotly_hover', handlePlotlyHover);
}

// MutationObserver for Gradio DOM replacement
function setupObserver() {
    const container = document.querySelector('#umap-plot');
    if (!container) {
        setTimeout(setupObserver, 500);
        return;
    }
    new MutationObserver(() => {
        if (initComplete) setTimeout(attachPlotlyHandlers, 100);
    }).observe(container, { childList: true, subtree: true });
}

// Initialize
setTimeout(() => {
    attachPlotlyHandlers();
    setupObserver();
}, 1500);

// Polling backup (1s interval after init)
setInterval(() => {
    if (!initComplete) return;
    const plotDiv = findPlotDiv();
    if (plotDiv && isPlotlyReady(plotDiv) && (plotDiv !== currentPlotDiv || !plotDiv._handlersAttached)) {
        attachPlotlyHandlers();
    }
}, 1000);
"""

# CSS for layout, plot sizing, and reduced chrome
# Module-level for Gradio 6 compatibility (passed to launch() not Blocks())
CUSTOM_CSS = """
    /* Main container fills viewport with min dimensions */
    .gradio-container {
        max-width: 100% !important;
        padding: 0.5rem !important;
        min-width: 900px !important;
        min-height: 600px !important;
        overflow: auto !important;
    }

    /* Main row - fixed height to prevent iframe expansion issues */
    #main-row {
        height: 1200px !important;
        max-height: 1200px !important;
        align-items: flex-start !important;
        flex-wrap: nowrap !important;
    }

    /* Sidebars: fixed width, scrollable */
    #left-sidebar, #right-sidebar {
        flex: 0 0 220px !important;
        max-height: 1200px !important;
        overflow-y: auto !important;
        padding: 0.25rem !important;
    }

    /* Center column stretches to fill remaining space */
    #center-column {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 auto !important;
        min-width: 600px !important;
        max-height: 1200px !important;
    }

    /* Plot container - fixed height, no vh units (iframe-safe) */
    #umap-plot {
        height: 1000px !important;
        max-height: 1000px !important;
        min-height: 400px !important;
    }

    /* Make Plotly fill its container with constrained height */
    #umap-plot > div,
    #umap-plot .js-plotly-plot,
    #umap-plot .plotly-graph-div {
        height: 100% !important;
        max-height: 1000px !important;
        width: 100% !important;
    }

    /* Hidden textboxes for JS bridge */
    #click-data-box,
    div:has(> #click-data-box) {
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }

    #hover-data-box,
    div:has(> #hover-data-box) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }

    /* Reduce group padding */
    .gr-group {
        padding: 0.5rem !important;
        margin-bottom: 0.10rem !important;
    }

    /* Compact markdown headers */
    .gr-group h3 {
        margin: 0 0 0.25rem 0 !important;
        font-size: 0.9rem !important;
    }

    /* Smaller image containers */
    .gr-image {
        margin: 0 !important;
    }

    /* Compact sliders and inputs */
    .gr-slider, .gr-number {
        margin-bottom: 0.25rem !important;
    }

    /* Compact buttons */
    .gr-button-sm {
        padding: 0.25rem 0.5rem !important;
        margin: 0.125rem 0 !important;
    }

    /* Gallery compact */
    .gr-gallery {
        margin: 0.25rem 0 !important;
    }

    /* Reduce title size */
    h1 {
        font-size: 1.5rem !important;
        margin: 0.25rem 0 0.5rem 0 !important;
    }

    /* Model/Layer/View selector rows: label + control inline */
    #model-row, #layer-row, #view-row {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 0.5rem !important;
        margin-bottom: 0 !important;
    }

    #model-row > div, #layer-row > div, #view-row > div {
        flex: 0 0 auto !important;
    }

    #model-label, #layer-label, #view-label {
        flex: 0 0 auto !important;
        width: auto !important;
        min-width: 0 !important;
        max-width: 60px !important;
    }

    #model-label p, #layer-label p, #view-label p {
        margin: 0 !important;
        font-size: 0.9rem !important;
    }

    #model-dropdown, #layer-dropdown {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        width: auto !important;
    }

    /* 2D/3D view toggle */
    #view-mode-radio {
        flex: 1 1 auto !important;
    }

    #view-mode-radio .wrap {
        gap: 0.5rem !important;
    }

    /* KNN row styling */
    #knn-row {
        flex-wrap: nowrap !important;
        align-items: center !important;
    }

    #knn-label {
        flex-shrink: 0 !important;
    }

    #knn-label p {
        margin: 0 !important;
        white-space: nowrap !important;
    }

    #knn-input {
        flex-shrink: 0 !important;
        max-width: 65px !important;
    }

    /* Hide spin buttons on K input */
    #knn-input input[type="number"]::-webkit-inner-spin-button,
    #knn-input input[type="number"]::-webkit-outer-spin-button {
        -webkit-appearance: none !important;
        margin: 0 !important;
    }

    #knn-input input[type="number"] {
        -moz-appearance: textfield !important;
    }

    #status-text {
        font-size: 0.8rem !important;
        color: #666 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    #status-text p {
        margin: 0 !important;
    }

    /* Compact generation params - all in one row with inline labels */
    #gen-params-row {
        gap: 0.25rem !important;
        align-items: center !important;
        flex-wrap: wrap !important;
    }

    #gen-params-row > div {
        flex-direction: row !important;
        align-items: center !important;
        gap: 0.2rem !important;
        flex: 0 1 auto !important;
    }

    #gen-params-row label {
        min-width: fit-content !important;
        margin: 0 !important;
        font-size: 0.75rem !important;
        white-space: nowrap !important;
    }

    #gen-params-row input {
        max-width: 50px !important;
        padding: 0.2rem 0.4rem !important;
        font-size: 0.85rem !important;
    }

    /* Hide number input spin buttons */
    #gen-params-row input[type="number"]::-webkit-inner-spin-button,
    #gen-params-row input[type="number"]::-webkit-outer-spin-button {
        -webkit-appearance: none !important;
        margin: 0 !important;
    }

    #gen-params-row input[type="number"] {
        -moz-appearance: textfield !important;
    }

    /* Smooth scaling for 64x64 images (auto = bilinear interpolation) */
    #preview-image img,
    #selected-image img,
    #generated-image img,
    #intermediate-gallery img,
    #neighbor-gallery img {
        image-rendering: auto !important;
    }

    /* Image containers: fill available space */
    #preview-image, #selected-image, #generated-image {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        overflow: hidden !important;
    }

    /* Preview image: force 300px height, preserve aspect ratio */
    #preview-image {
        width: 100% !important;
        min-height: 300px !important;
    }

    #preview-image img {
        height: 300px !important;
        width: auto !important;
        object-fit: contain !important;
    }

    /* Compact preview and selected details text */
    #preview-details,
    #selected-details {
        line-height: 1.3 !important;
    }

    #preview-details p,
    #selected-details p {
        margin: 0.15em 0 !important;
    }

    /* Selected/generated images: fill container */
    #selected-image img, #generated-image img {
        width: 100% !important;
        height: 100% !important;
        object-fit: contain !important;
        max-width: none !important;
        max-height: none !important;
    }

    /* Generated image: match preview size */
    #generated-image {
        width: 100% !important;
        min-height: 180px !important;
    }

    #generated-image img {
        min-height: 300px !important;
    }

    /* Gallery images also scale up */
    #intermediate-gallery img,
    #neighbor-gallery img {
        width: 100% !important;
        height: 100% !important;
        object-fit: contain !important;
    }

    /* Neighbor gallery: scrollable with more height */
    #neighbor-gallery {
        max-height: 280px !important;
        overflow-y: auto !important;
    }

    /* Lock scroll when gallery preview is active */
    #neighbor-gallery:has(.preview) {
        overflow-y: hidden !important;
    }

    /* Make preview fill the container */
    #neighbor-gallery .preview {
        max-height: 280px !important;
    }
"""
