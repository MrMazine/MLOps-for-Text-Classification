// Main JavaScript for DVC Text Classification Pipeline

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Show loading overlay
    function showLoading() {
        document.getElementById('loadingOverlay').classList.remove('d-none');
    }

    // Hide loading overlay
    function hideLoading() {
        document.getElementById('loadingOverlay').classList.add('d-none');
    }

    // Show error message
    function showError(message) {
        const alertHtml = `
            <div class="alert alert-danger alert-dismissible fade show">
                <i class="fas fa-exclamation-triangle me-2"></i> ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        // Insert at the top of the container
        const container = document.querySelector('.container');
        container.insertAdjacentHTML('afterbegin', alertHtml);
    }

    // Show success message
    function showSuccess(message) {
        const alertHtml = `
            <div class="alert alert-success alert-dismissible fade show">
                <i class="fas fa-check-circle me-2"></i> ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        // Insert at the top of the container
        const container = document.querySelector('.container');
        container.insertAdjacentHTML('afterbegin', alertHtml);
    }

    // Upload dataset
    const uploadDatasetBtn = document.getElementById('uploadDatasetBtn');
    if (uploadDatasetBtn) {
        uploadDatasetBtn.addEventListener('click', function() {
            const form = document.getElementById('uploadDatasetForm');
            const formData = new FormData(form);
            
            // Validate form
            const datasetName = formData.get('dataset_name');
            const file = formData.get('file');
            
            if (!datasetName || !file.name) {
                showError('Please fill in all fields');
                return;
            }
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('uploadDatasetModal'));
            modal.hide();
            
            // Show loading
            showLoading();
            
            // Send AJAX request
            fetch('/upload_dataset', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.error) {
                    showError(data.error);
                } else {
                    showSuccess(data.message);
                    // Reload page to show new dataset
                    setTimeout(() => {
                        window.location.reload();
                    }, 1500);
                }
            })
            .catch(error => {
                hideLoading();
                showError('An error occurred during upload: ' + error);
            });
        });
    }

    // Sync Remote button
    const syncRemoteBtn = document.getElementById('syncRemoteBtn');
    if (syncRemoteBtn) {
        syncRemoteBtn.addEventListener('click', function() {
            // Show sync modal
            const syncModal = new bootstrap.Modal(document.getElementById('syncRemoteModal'));
            syncModal.show();
        });
    }

    // Push to remote
    const pushRemoteBtn = document.getElementById('pushRemoteBtn');
    if (pushRemoteBtn) {
        pushRemoteBtn.addEventListener('click', function() {
            // Close sync modal
            const syncModal = bootstrap.Modal.getInstance(document.getElementById('syncRemoteModal'));
            syncModal.hide();
            
            // Show loading
            showLoading();
            
            // Send AJAX request
            fetch('/sync_remote', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action: 'push' })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.error) {
                    showError(data.error);
                } else {
                    showSuccess(data.message);
                }
            })
            .catch(error => {
                hideLoading();
                showError('An error occurred during sync: ' + error);
            });
        });
    }

    // Pull from remote
    const pullRemoteBtn = document.getElementById('pullRemoteBtn');
    if (pullRemoteBtn) {
        pullRemoteBtn.addEventListener('click', function() {
            // Close sync modal
            const syncModal = bootstrap.Modal.getInstance(document.getElementById('syncRemoteModal'));
            syncModal.hide();
            
            // Show loading
            showLoading();
            
            // Send AJAX request
            fetch('/sync_remote', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action: 'pull' })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.error) {
                    showError(data.error);
                } else {
                    showSuccess(data.message);
                    // Reload page to show updated datasets
                    setTimeout(() => {
                        window.location.reload();
                    }, 1500);
                }
            })
            .catch(error => {
                hideLoading();
                showError('An error occurred during sync: ' + error);
            });
        });
    }

    // View dataset versions
    const viewVersionsBtns = document.querySelectorAll('.view-versions-btn');
    if (viewVersionsBtns.length > 0) {
        viewVersionsBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                const datasetPath = this.getAttribute('data-dataset-path');
                
                // Show versions modal
                const versionsModal = new bootstrap.Modal(document.getElementById('versionsModal'));
                versionsModal.show();
                
                // Reset content
                document.getElementById('versionsContent').innerHTML = `
                    <div class="text-center">
                        <div class="spinner-border" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </div>
                `;
                
                // Send AJAX request
                fetch('/dataset_versions?dataset_path=' + encodeURIComponent(datasetPath))
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('versionsContent').innerHTML = `
                            <div class="alert alert-danger">
                                <i class="fas fa-exclamation-triangle me-2"></i> ${data.error}
                            </div>
                        `;
                    } else {
                        const versions = data.versions;
                        let html = `
                            <h6>Dataset: <span class="dataset-path">${datasetPath}</span></h6>
                            <div class="version-timeline mt-4">
                        `;
                        
                        if (versions && versions.length > 0) {
                            versions.forEach(version => {
                                html += `
                                    <div class="version-node">
                                        <div class="d-flex justify-content-between">
                                            <div>
                                                <strong>${version.message}</strong>
                                                <div class="text-muted small">${version.date} by ${version.author}</div>
                                            </div>
                                            <div>
                                                <button class="btn btn-sm btn-outline-primary checkout-version-btn"
                                                        data-dataset-path="${datasetPath}"
                                                        data-version="${version.hash}">
                                                    <i class="fas fa-clock-rotate-left"></i> Checkout
                                                </button>
                                            </div>
                                        </div>
                                        <div class="mt-1">
                                            <code class="small">${version.hash}</code>
                                        </div>
                                    </div>
                                `;
                            });
                        } else {
                            html += `
                                <div class="alert alert-info">
                                    <i class="fas fa-info-circle me-2"></i> No version history found for this dataset.
                                </div>
                            `;
                        }
                        
                        html += `</div>`;
                        document.getElementById('versionsContent').innerHTML = html;
                        
                        // Add event listeners to checkout buttons
                        const checkoutBtns = document.querySelectorAll('.checkout-version-btn');
                        checkoutBtns.forEach(btn => {
                            btn.addEventListener('click', checkoutVersion);
                        });
                    }
                })
                .catch(error => {
                    document.getElementById('versionsContent').innerHTML = `
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-triangle me-2"></i> An error occurred: ${error}
                        </div>
                    `;
                });
            });
        });
    }

    // Checkout version
    function checkoutVersion() {
        const datasetPath = this.getAttribute('data-dataset-path');
        const version = this.getAttribute('data-version');
        
        // Close versions modal
        const versionsModal = bootstrap.Modal.getInstance(document.getElementById('versionsModal'));
        versionsModal.hide();
        
        // Show loading
        showLoading();
        
        // Send AJAX request
        fetch('/checkout_version', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset_path: datasetPath,
                version: version
            })
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            if (data.error) {
                showError(data.error);
            } else {
                showSuccess(data.message);
                // Reload page to show updated dataset
                setTimeout(() => {
                    window.location.reload();
                }, 1500);
            }
        })
        .catch(error => {
            hideLoading();
            showError('An error occurred: ' + error);
        });
    }

    // View dataset statistics
    const statsBtns = document.querySelectorAll('.stats-btn');
    if (statsBtns.length > 0) {
        statsBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                const datasetPath = this.getAttribute('data-dataset-path');
                
                // Show stats modal
                const statsModal = new bootstrap.Modal(document.getElementById('statsModal'));
                statsModal.show();
                
                // Reset content
                document.getElementById('statsContent').innerHTML = `
                    <div class="text-center">
                        <div class="spinner-border" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </div>
                `;
                
                // Send AJAX request
                fetch('/dataset_stats?dataset_path=' + encodeURIComponent(datasetPath))
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('statsContent').innerHTML = `
                            <div class="alert alert-danger">
                                <i class="fas fa-exclamation-triangle me-2"></i> ${data.error}
                            </div>
                        `;
                    } else {
                        const stats = data.stats;
                        let html = `
                            <h6>Dataset Statistics: <span class="dataset-path">${datasetPath}</span></h6>
                            
                            <div class="row mt-3">
                                <div class="col-md-6">
                                    <div class="stats-badge">
                                        <i class="fas fa-file me-2"></i> Files: ${stats.file_count}
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="stats-badge">
                                        <i class="fas fa-table-list me-2"></i> Records: ${stats.record_count}
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="stats-badge">
                                        <i class="fas fa-text-height me-2"></i> Avg. Text Length: ${stats.avg_text_length.toFixed(2)}
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="stats-badge">
                                        <i class="fas fa-text-height me-2"></i> Max Text Length: ${stats.max_text_length}
                                    </div>
                                </div>
                            </div>
                            
                            <h6 class="mt-4">Fields:</h6>
                            <div class="mb-3">
                        `;
                        
                        stats.fields.forEach(field => {
                            html += `<span class="badge bg-secondary me-2 mb-2">${field}</span>`;
                        });
                        
                        html += `
                            </div>
                            
                            <h6 class="mt-4">Sample Records:</h6>
                            <div class="sample-records">
                                <pre class="mb-0"><code>${JSON.stringify(stats.sample_records, null, 2)}</code></pre>
                            </div>
                        `;
                        
                        document.getElementById('statsContent').innerHTML = html;
                    }
                })
                .catch(error => {
                    document.getElementById('statsContent').innerHTML = `
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-triangle me-2"></i> An error occurred: ${error}
                        </div>
                    `;
                });
            });
        });
    }

    // Preprocess dataset
    const preprocessBtns = document.querySelectorAll('.preprocess-btn');
    if (preprocessBtns.length > 0) {
        preprocessBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                const datasetPath = this.getAttribute('data-dataset-path');
                document.getElementById('preprocessDatasetPath').value = datasetPath;
            });
        });
    }

    // Process dataset
    const preprocessBtn = document.getElementById('preprocessBtn');
    if (preprocessBtn) {
        preprocessBtn.addEventListener('click', function() {
            const form = document.getElementById('preprocessForm');
            const datasetPath = document.getElementById('preprocessDatasetPath').value;
            
            // Get selected preprocessing steps
            const checkboxes = form.querySelectorAll('input[name="preprocessing_steps"]:checked');
            const preprocessingSteps = [...checkboxes].map(cb => cb.value);
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('preprocessModal'));
            modal.hide();
            
            // Show loading
            showLoading();
            
            // Send AJAX request
            fetch('/preprocess_dataset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    dataset_path: datasetPath,
                    preprocessing_steps: preprocessingSteps
                })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.error) {
                    showError(data.error);
                } else {
                    showSuccess(data.message);
                    // Reload page to show processed dataset
                    setTimeout(() => {
                        window.location.reload();
                    }, 1500);
                }
            })
            .catch(error => {
                hideLoading();
                showError('An error occurred during preprocessing: ' + error);
            });
        });
    }
});
