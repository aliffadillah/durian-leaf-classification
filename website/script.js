class DurianLeafClassifier {
    constructor() {
        this.initializeElements();
        this.attachEventListeners();
        this.setupDragAndDrop();
        this.selectedFile = null;
        this.isProcessing = false;
        this.testServerConnection();
    }

    async testServerConnection() {
        try {
            const response = await axios.get(
                `${API_CONFIG.BASE_URL}/health`,
                { 
                    timeout: 5000,
                    headers: {
                        'Accept': 'application/json',
                        'Content-Type': 'application/json'
                    }
                }
            );
            console.log('Server connection: OK');
            this.showConnectionStatus(true);
        } catch (error) {
            console.error('Server connection failed:', error);
            this.showConnectionStatus(false);
        }
    }

    showConnectionStatus(connected) {
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            statusElement.textContent = connected ? '🟢 Connected' : '🔴 Disconnected';
            statusElement.className = connected ? 'text-green-600' : 'text-red-600';
        }
    }

    initializeElements() {
        // File input elements
        this.fileInput = document.getElementById('fileInput');
        this.dropZone = document.getElementById('dropZone');
        this.fileInfo = document.getElementById('fileInfo');
        this.fileName = document.getElementById('fileName');
        this.fileSize = document.getElementById('fileSize');
        this.removeFileBtn = document.getElementById('removeFile');

        // Image preview
        this.imagePreview = document.getElementById('imagePreview');
        this.previewImage = document.getElementById('previewImage');

        // Action buttons
        this.classifyBtn = document.getElementById('classifyBtn');
        this.resetBtn = document.getElementById('resetBtn');
        this.classifyText = document.getElementById('classifyText');
        this.classifySpinner = document.getElementById('classifySpinner');

        // Status sections
        this.loadingSection = document.getElementById('loadingSection');
        this.errorSection = document.getElementById('errorSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.errorMessage = document.getElementById('errorMessage');
        this.progressBar = document.getElementById('progressBar');

        // Results elements
        this.predictionResult = document.getElementById('predictionResult');
        this.contrastValue = document.getElementById('contrastValue');
        this.correlationValue = document.getElementById('correlationValue');
        this.energyValue = document.getElementById('energyValue');
        this.homogeneityValue = document.getElementById('homogeneityValue');
        this.closestMatch = document.getElementById('closestMatch');
        this.topMatches = document.getElementById('topMatches');

        // Health check
        this.healthCheckBtn = document.getElementById('healthCheck');
        this.healthModal = document.getElementById('healthModal');
        this.healthContent = document.getElementById('healthContent');
        this.closeHealthModal = document.getElementById('closeHealthModal');
    }

    attachEventListeners() {
        // File input
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        this.dropZone.addEventListener('click', () => this.fileInput.click());
        this.removeFileBtn.addEventListener('click', () => this.resetFileSelection());

        // Action buttons
        this.classifyBtn.addEventListener('click', () => this.classifyImage());
        this.resetBtn.addEventListener('click', () => this.resetAll());

        // Health check
        this.healthCheckBtn.addEventListener('click', () => this.checkHealth());
        this.closeHealthModal.addEventListener('click', () => this.closeHealthModalHandler());

        // Modal outside click
        this.healthModal.addEventListener('click', (e) => {
            if (e.target === this.healthModal) {
                this.closeHealthModalHandler();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeHealthModalHandler();
            }
        });
    }

    setupDragAndDrop() {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.dropZone.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            this.dropZone.addEventListener(eventName, () => this.highlight(), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            this.dropZone.addEventListener(eventName, () => this.unhighlight(), false);
        });

        this.dropZone.addEventListener('drop', (e) => this.handleDrop(e), false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    highlight() {
        this.dropZone.classList.add('drag-over');
    }

    unhighlight() {
        this.dropZone.classList.remove('drag-over');
    }

    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }

    handleFile(file) {
        if (!this.validateFile(file)) {
            return;
        }

        this.selectedFile = file;
        this.displayFileInfo(file);
        this.displayImagePreview(file);
        this.showActionButtons();
        this.hideError();
    }

    validateFile(file) {
        const allowedTypes = ['image/png', 'image/jpg', 'image/jpeg', 'image/gif', 'image/bmp'];
        const maxSize = 10 * 1024 * 1024; // 10MB

        if (!allowedTypes.includes(file.type)) {
            this.showError('Please select a valid image file (PNG, JPG, JPEG, GIF, BMP)');
            return false;
        }

        if (file.size > maxSize) {
            this.showError('File size must be less than 10MB');
            return false;
        }

        return true;
    }

    displayFileInfo(file) {
        this.fileName.textContent = file.name;
        this.fileSize.textContent = this.formatFileSize(file.size);
        this.fileInfo.classList.remove('hidden');
    }

    displayImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.imagePreview.classList.remove('hidden');
            this.imagePreview.classList.add('fade-in');
        };
        reader.readAsDataURL(file);
    }

    showActionButtons() {
        this.classifyBtn.classList.remove('hidden');
        this.resetBtn.classList.remove('hidden');
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async classifyImage() {
        if (!this.selectedFile || this.isProcessing) {
            return;
        }

        this.isProcessing = true;
        this.showLoading();
        this.hideError();
        this.hideResults();
        this.setClassifyButtonLoading(true);

        const formData = new FormData();
        formData.append('image', this.selectedFile);

        try {
            // Simulate progress
            this.animateProgress();

            console.log('Sending request to:', `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.PREDICT}`);

            const response = await axios.post(
                `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.PREDICT}`,
                formData,
                {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                        'Accept': 'application/json'
                    },
                    timeout: API_CONFIG.TIMEOUT,
                    withCredentials: false
                }
            );

            console.log('Response received:', response.data);

            if (response.data.success || response.data.model_prediction) {
                this.displayResults(response.data);
                this.showSuccess();
            } else {
                throw new Error(response.data.message || 'Classification failed');
            }

        } catch (error) {
            console.error('Classification error:', error);
            this.handleError(error);
        } finally {
            this.isProcessing = false;
            this.hideLoading();
            this.setClassifyButtonLoading(false);
        }
    }

    animateProgress() {
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) {
                progress = 90;
                clearInterval(interval);
            }
            this.progressBar.style.width = `${progress}%`;
        }, 200);

        // Complete progress when done
        setTimeout(() => {
            clearInterval(interval);
            this.progressBar.style.width = '100%';
        }, 3000);
    }

    displayResults(data) {
        // Display prediction dengan styling khusus untuk "noclass"
        const prediction = data.model_prediction;
        this.predictionResult.textContent = prediction;
        
        // Tambahkan styling khusus untuk hasil "Bukan Daun Durian"
        if (prediction.includes("Bukan Daun Durian")) {
            // Ubah dari text-red-600 menjadi text-white untuk warna putih
            this.predictionResult.classList.remove('text-red-600');
            this.predictionResult.classList.add('text-white');
            this.predictionResult.parentElement.classList.remove('from-green-500', 'to-emerald-600');
            this.predictionResult.parentElement.classList.add('from-red-500', 'to-red-600');
        } else {
            this.predictionResult.classList.remove('text-red-600');
            this.predictionResult.classList.add('text-white');
            this.predictionResult.parentElement.classList.add('from-green-500', 'to-emerald-600');
            this.predictionResult.parentElement.classList.remove('from-red-500', 'to-red-600');
        }
        
        this.predictionResult.classList.add('success-bounce');

        // Display GLCM features
        const features = data.input_features;
        this.contrastValue.textContent = this.formatFeatureValue(features.contrast);
        this.correlationValue.textContent = this.formatFeatureValue(features.correlation);
        this.energyValue.textContent = this.formatFeatureValue(features.energy);
        this.homogeneityValue.textContent = this.formatFeatureValue(features.homogeneity);

        // Hanya tampilkan matches jika bukan "Bukan Daun Durian"
        if (prediction.includes("Bukan Daun Durian")) {
            // Sembunyikan section matches untuk noclass
            document.getElementById('closestMatchSection').style.display = 'none';
            document.getElementById('topMatchesSection').style.display = 'none';
        } else {
            // Tampilkan section matches untuk prediksi valid
            document.getElementById('closestMatchSection').style.display = 'block';
            document.getElementById('topMatchesSection').style.display = 'block';
            
            // Display closest match
            if (data.closest_match) {
                const closest = data.closest_match;
                this.closestMatch.innerHTML = `
                    <div class="flex justify-between items-center">
                        <div>
                            <p class="font-semibold text-blue-800">${closest.label}</p>
                            <p class="text-sm text-blue-600">Index: ${closest.index}</p>
                        </div>
                        <div class="text-right">
                            <p class="text-sm text-gray-600">Distance</p>
                            <p class="font-bold text-blue-800">${this.formatFeatureValue(closest.distance)}</p>
                        </div>
                    </div>
                `;
            }

            // Display top matches
            this.topMatches.innerHTML = '';
            if (data.top_5_matches && data.top_5_matches.length > 0) {
                data.top_5_matches.forEach((match, index) => {
                    const matchElement = document.createElement('div');
                    matchElement.className = 'match-item bg-gray-50 rounded-lg p-4 flex justify-between items-center slide-in';
                    matchElement.style.animationDelay = `${index * 0.1}s`;
                    
                    const confidencePercent = Math.max(0, 100 - (match.distance * 10));
                    const confidenceColor = confidencePercent > 70 ? 'text-green-600' : 
                                           confidencePercent > 40 ? 'text-yellow-600' : 'text-red-600';
                    
                    matchElement.innerHTML = `
                        <div class="flex items-center space-x-3">
                            <div class="w-8 h-8 bg-emerald-100 rounded-full flex items-center justify-center">
                                <span class="text-sm font-bold text-emerald-600">${index + 1}</span>
                            </div>
                            <div>
                                <p class="font-medium text-gray-800">${match.label}</p>
                                <p class="text-sm text-gray-500">Index: ${match.index}</p>
                            </div>
                        </div>
                        <div class="text-right">
                            <p class="text-sm text-gray-600">Distance</p>
                            <p class="font-bold text-gray-800">${this.formatFeatureValue(match.distance)}</p>
                            <p class="text-xs ${confidenceColor}">${confidencePercent.toFixed(1)}% match</p>
                        </div>
                    `;
                    
                    this.topMatches.appendChild(matchElement);
                });
            }
        }

        this.showResults();
    }

    formatFeatureValue(value) {
        return typeof value === 'number' ? value.toFixed(6) : value;
    }

    async checkHealth() {
        try {
            const response = await axios.get(
                `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.HEALTH}`,
                { timeout: 5000 }
            );

            this.displayHealthInfo(response.data);
        } catch (error) {
            this.displayHealthError(error);
        }
        
        this.showHealthModal();
    }

    displayHealthInfo(data) {
        this.healthContent.innerHTML = `
            <div class="space-y-3">
                <div class="flex justify-between items-center">
                    <span class="text-gray-600">Status:</span>
                    <span class="px-2 py-1 bg-green-100 text-green-800 rounded text-sm font-medium">
                        ${data.status || 'Healthy'}
                    </span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-gray-600">Model:</span>
                    <span class="px-2 py-1 ${data.model_status === 'loaded' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'} rounded text-sm font-medium">
                        ${data.model_status || 'Unknown'}
                    </span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-gray-600">Data:</span>
                    <span class="px-2 py-1 ${data.data_status === 'loaded' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'} rounded text-sm font-medium">
                        ${data.data_status || 'Unknown'}
                    </span>
                </div>
                ${data.timestamp ? `
                <div class="flex justify-between items-center">
                    <span class="text-gray-600">Last Check:</span>
                    <span class="text-sm text-gray-800">${data.timestamp}</span>
                </div>
                ` : ''}
            </div>
        `;
    }

    displayHealthError(error) {
        this.healthContent.innerHTML = `
            <div class="text-center text-red-600">
                <div class="text-2xl mb-2">⚠️</div>
                <p class="font-medium">Health Check Failed</p>
                <p class="text-sm text-gray-600 mt-1">
                    ${error.message || 'Unable to connect to the server'}
                </p>
            </div>
        `;
    }

    showHealthModal() {
        this.healthModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }

    closeHealthModalHandler() {
        this.healthModal.classList.add('hidden');
        document.body.style.overflow = 'auto';
    }

    handleError(error) {
        let errorMsg = 'An unexpected error occurred';
        
        if (error.code === 'ECONNABORTED') {
            errorMsg = 'Request timeout. The server is taking too long to respond.';
        } else if (error.code === 'ERR_NETWORK') {
            errorMsg = 'Network error. Please check your internet connection and make sure the server is running.';
        } else if (error.response) {
            // Server responded with error status
            const status = error.response.status;
            const data = error.response.data;
            
            switch (status) {
                case 400:
                    errorMsg = data?.message || 'Bad request. Please check your input.';
                    break;
                case 413:
                    errorMsg = 'File too large. Please select a smaller image.';
                    break;
                case 500:
                    errorMsg = data?.message || 'Server error. Please try again later.';
                    break;
                default:
                    errorMsg = data?.message || `Server error (${status})`;
            }
        } else if (error.request) {
            // Request was made but no response received
            errorMsg = 'Unable to connect to the server. Please ensure:\n\n' +
                      '1. The Flask server is running (python app.py)\n' +
                      '2. Server is accessible at ' + API_CONFIG.BASE_URL + '\n' +
                      '3. Your firewall/antivirus is not blocking the connection\n' +
                      '4. CORS is properly configured';
        } else {
            errorMsg = error.message;
        }

        this.showError(errorMsg);
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorSection.classList.remove('hidden');
        this.errorSection.classList.add('error-shake');
        
        // Remove shake animation after it completes
        setTimeout(() => {
            this.errorSection.classList.remove('error-shake');
        }, 500);
    }

    hideError() {
        this.errorSection.classList.add('hidden');
    }

    showLoading() {
        this.loadingSection.classList.remove('hidden');
        this.loadingSection.classList.add('fade-in');
    }

    hideLoading() {
        this.loadingSection.classList.add('hidden');
        this.progressBar.style.width = '0%';
    }

    showResults() {
        this.resultsSection.classList.remove('hidden');
        this.resultsSection.classList.add('fade-in');
        
        // Smooth scroll to results
        setTimeout(() => {
            this.resultsSection.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }, 300);
    }

    hideResults() {
        this.resultsSection.classList.add('hidden');
    }

    showSuccess() {
        // You can add any success notification here
        console.log('Classification completed successfully!');
    }

    setClassifyButtonLoading(loading) {
        if (loading) {
            this.classifyText.textContent = 'Processing...';
            this.classifySpinner.classList.remove('hidden');
            this.classifyBtn.disabled = true;
        } else {
            this.classifyText.textContent = 'Classify Leaf';
            this.classifySpinner.classList.add('hidden');
            this.classifyBtn.disabled = false;
        }
    }

    resetFileSelection() {
        this.selectedFile = null;
        this.fileInput.value = '';
        this.fileInfo.classList.add('hidden');
        this.imagePreview.classList.add('hidden');
        this.classifyBtn.classList.add('hidden');
        this.resetBtn.classList.add('hidden');
        this.hideError();
    }

    resetAll() {
        this.resetFileSelection();
        this.hideResults();
        this.hideLoading();
        this.hideError();
        this.isProcessing = false;
        this.setClassifyButtonLoading(false);
        
        // Scroll back to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DurianLeafClassifier();
});

// Service Worker Registration (Optional for PWA)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}