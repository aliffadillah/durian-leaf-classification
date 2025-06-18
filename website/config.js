const API_CONFIG = {
    BASE_URL: 'http://localhost:5000',
    ENDPOINTS: {
        PREDICT: '/predict',
        HEALTH: '/health'
    },
    TIMEOUT: 30000, 
    MAX_FILE_SIZE: 10 * 1024 * 1024, 
    ALLOWED_TYPES: ['image/png', 'image/jpg', 'image/jpeg', 'image/gif', 'image/bmp']
};

window.API_CONFIG = API_CONFIG;

window.testConnection = async function() {
    try {
        const response = await fetch(`${API_CONFIG.BASE_URL}/health`, {
            method: 'GET',
            mode: 'cors'
        });
        console.log('Connection test:', response.ok ? 'SUCCESS' : 'FAILED');
        return response.ok;
    } catch (error) {
        console.error('Connection test failed:', error);
        return false;
    }
};

document.addEventListener('DOMContentLoaded', () => {
    setTimeout(testConnection, 1000);
});