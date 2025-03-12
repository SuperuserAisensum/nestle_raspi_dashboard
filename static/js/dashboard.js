// dashboard.js - Nestlé SKU Inventory Dashboard

// Initialize socket connection
const socket = io();

// Global state
let currentPage = 1;
const pageSize = 10;
let totalEvents = 0;
let events = [];
let skuData = {};

// DOM elements
const eventsTableBody = document.getElementById('eventsTableBody');
const paginationInfo = document.getElementById('paginationInfo');
const startCount = document.getElementById('startCount');
const endCount = document.getElementById('endCount');
const totalCount = document.getElementById('totalCount');
const prevPageBtn = document.getElementById('prevPage');
const nextPageBtn = document.getElementById('nextPage');
const eventDetailModal = document.getElementById('eventDetailModal');
const closeModalBtn = document.getElementById('closeModal');

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    initializeDashboard();
    
    // Pagination controls
    prevPageBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            fetchEvents();
        }
    });
    
    nextPageBtn.addEventListener('click', () => {
        if (currentPage * pageSize < totalEvents) {
            currentPage++;
            fetchEvents();
        }
    });
    
    // Modal close button
    closeModalBtn.addEventListener('click', () => {
        eventDetailModal.classList.add('hidden');
    });
});

// Socket event listeners
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('new_detection', async (data) => {
    // Format timestamp
    const timestamp = new Date(data.timestamp).toLocaleString();
    
    // Create toast message with image path check
    const imagePath = data.image_path ? `/${data.image_path}` : '/static/placeholder.png';
    
    const message = `
        <div class="flex items-start space-x-4">
            <div class="flex-1">
                <div class="font-medium text-gray-900">New Detection</div>
                <div class="text-sm text-gray-600">Device: ${data.device_id}</div>
                <div class="text-sm mt-1">
                    <span class="text-blue-600">Nestlé: ${data.nestle_count}</span> | 
                    <span class="text-red-600">Competitor: ${data.competitor_count}</span>
                </div>
                <div class="text-xs text-gray-500 mt-1">${timestamp}</div>
            </div>
            ${data.image_path ? `
            <div class="flex-shrink-0">
                <img src="${imagePath}" 
                     alt="Detection" 
                     onerror="this.src='/static/placeholder.png'"
                     class="h-16 w-16 object-cover rounded shadow">
            </div>
            ` : ''}
        </div>
    `;
    
    showToastNotification(message);
    
    // Update dashboard data
    await fetchDashboardData();
    currentPage = 1;
    await fetchEvents();
});

// Initialize the dashboard
function initializeDashboard() {
    fetchDashboardData().then(() => {
        fetchEvents();
        startAutoRefresh();
    });
}

// Show toast notification
function showToastNotification(message) {
    const toast = document.createElement('div');
    toast.className = 'toast-notification';
    toast.innerHTML = message;
    
    document.body.appendChild(toast);
    
    // Show with animation
    setTimeout(() => {
        toast.classList.add('show');
    }, 100);
    
    // Remove after 5 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 300);
    }, 5000);
}

// Fetch dashboard data from server
async function fetchDashboardData() {
    try {
        const response = await fetch('/api/dashboard_data');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        skuData = data;
        
    } catch (error) {
        console.error("Error fetching dashboard data:", error);
    }
}

// Function to get IQI color class
function getIQIColorClass(iqi) {
    if (iqi >= 80) return 'bg-green-100 text-green-800'; // Excellent quality
    if (iqi >= 60) return 'bg-blue-100 text-blue-800';   // Good quality
    if (iqi >= 40) return 'bg-yellow-100 text-yellow-800'; // Fair quality
    return 'bg-red-100 text-red-800';                    // Poor quality
}

// Function to get IQI quality text
function getIQIQualityText(iqi) {
    if (iqi >= 80) return 'Excellent quality';
    if (iqi >= 60) return 'Good quality';
    if (iqi >= 40) return 'Fair quality';
    return 'Poor quality';
}

// Function to format location display
function formatLocation(location) {
    if (!location) return 'Unknown';
    
    let displayText = '';
    if (typeof location === 'string') {
        return location;
    }
    
    // Format address
    if (location.address && location.address !== 'Unknown') {
        displayText = location.address;
    }
    
    // Add Google Maps link if coordinates available
    if (location.coordinates) {
        const { latitude, longitude } = location.coordinates;
        if (latitude !== 'N/A' && longitude !== 'N/A') {
            displayText = `
                <div class="location-container">
                    <div class="location-address">${displayText || 'Address not available'}</div>
                    <a href="https://www.google.com/maps?q=${latitude},${longitude}" 
                       target="_blank" 
                       class="text-blue-600 hover:text-blue-800 text-sm">
                        View on Maps
                    </a>
                </div>
            `;
        }
    }
    
    return displayText || 'Location not available';
}

// Function to render events table
function renderEventsTable() {
    eventsTableBody.innerHTML = '';
    
    if (!events || !events.length) {
        eventsTableBody.innerHTML = `
            <tr>
                <td colspan="7" class="px-4 py-4 text-center text-gray-500">
                    No detection events found
                </td>
            </tr>
        `;
        return;
    }

    // Keep Nestlé Products count fixed at 10
    const nestleHeader = document.querySelector('th[data-column="nestle"]') || 
                        document.querySelector('th:nth-child(6)') ||
                        document.querySelector('th:contains("NESTLÉ")');
    if (nestleHeader) {
        nestleHeader.textContent = `NESTLÉ PRODUCTS(10)`;
    }
    
    events.forEach(event => {
        const row = document.createElement('tr');
        
        // Get counts directly from event data
        const nestleCount = event.nestle_count || 0;
        const compCount = event.competitor_count || 0;
        const total = nestleCount + compCount;
        const nestlePercentage = total > 0 ? Math.round((nestleCount / total) * 100) : 0;
        const compPercentage = total > 0 ? Math.round((compCount / total) * 100) : 0;
        const iqiScore = event.iqi_score ? parseFloat(event.iqi_score).toFixed(2) : "0.00";
        const iqiColorClass = getIQIColorClass(event.iqi_score);
        const iqiQualityText = getIQIQualityText(event.iqi_score);
        
        row.innerHTML = `
            <td class="px-4 py-4 whitespace-nowrap text-sm text-gray-900">#${event.id}</td>
            <td class="px-4 py-4 whitespace-nowrap text-sm text-gray-500">${event.device_id}</td>
            <td class="px-4 py-4 whitespace-nowrap text-sm text-gray-500">${formatDate(event.timestamp)}</td>
            <td class="px-4 py-4 whitespace-nowrap text-sm">
                <span class="font-medium text-gray-900">${iqiScore}</span>
                <span class="ml-2 px-2 py-1 text-xs font-medium ${iqiColorClass} rounded-full">${iqiQualityText}</span>
            </td>
            <td class="px-4 py-4 text-sm text-gray-500">${formatLocation(event.location)}</td>
            <td class="px-4 py-4 whitespace-nowrap text-sm">
                <span class="font-medium text-gray-900">${nestleCount}</span>
                <span class="ml-2 px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">${nestlePercentage}%</span>
            </td>
            <td class="px-4 py-4 whitespace-nowrap text-sm">
                <span class="font-medium text-gray-900">${compCount}</span>
                <span class="ml-2 px-2 py-1 text-xs font-medium bg-red-100 text-red-800 rounded-full">${compPercentage}%</span>
            </td>
            <td class="px-4 py-4 whitespace-nowrap text-sm text-blue-600 hover:text-blue-800">
                <div class="flex space-x-2">
                    <button onclick="viewEventDetails(${event.id})" class="hover:underline">View</button>
                    <a href="/download/${event.image_path ? event.image_path.split('/').pop() : ''}" class="text-gray-600 hover:text-gray-800">Download</a>
                </div>
            </td>
        `;
        eventsTableBody.appendChild(row);
    });
}

// Format date for display
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

// Update pagination information
function updatePagination() {
    const start = ((currentPage - 1) * pageSize) + 1;
    const end = Math.min(currentPage * pageSize, totalEvents);
    
    startCount.textContent = totalEvents > 0 ? start : 0;
    endCount.textContent = end;
    totalCount.textContent = totalEvents;
    
    // Enable/disable pagination buttons
    prevPageBtn.disabled = currentPage <= 1;
    prevPageBtn.classList.toggle('opacity-50', currentPage <= 1);
    
    nextPageBtn.disabled = currentPage * pageSize >= totalEvents;
    nextPageBtn.classList.toggle('opacity-50', currentPage * pageSize >= totalEvents);
}

// Function to view event details
async function viewEventDetails(eventId) {
    try {
        const response = await fetch(`/api/events/${eventId}`);
        if (!response.ok) {
            throw new Error('Failed to fetch event details');
        }
        
        const data = await response.json();
        
        // Update modal event info
        document.getElementById('modalEventInfo').textContent = `Event #${data.id} | Device: ${data.device_id}`;
        
        // Calculate total Nestlé products from products data
        let nestleTotal = 0;
        if (data.products && data.products.nestle_products) {
            nestleTotal = Object.values(data.products.nestle_products).reduce((sum, count) => sum + count, 0);
        }
        data.nestleCount = nestleTotal;

        // Calculate total competitor products
        let compTotal = 0;
        if (data.products && data.products.competitor_products) {
            if (Array.isArray(data.products.competitor_products)) {
                compTotal = data.products.competitor_products.length;
            } else if (typeof data.products.competitor_products === 'object') {
                compTotal = Object.values(data.products.competitor_products).reduce((sum, count) => sum + count, 0);
            }
        }
        data.compCount = compTotal;
        
        // Calculate percentages
        const total = data.nestleCount + data.compCount;
        const nestlePercent = total > 0 ? Math.round((data.nestleCount / total) * 100) : 0;
        const compPercent = total > 0 ? Math.round((data.compCount / total) * 100) : 0;

        // Update counts with percentages
        const nestleCountElement = document.getElementById('modalNestleCount');
        const compCountElement = document.getElementById('modalCompCount');
        const timestampElement = document.getElementById('modalTimestamp');
        
        nestleCountElement.innerHTML = `
            ${data.nestleCount}
            <span class="ml-2 px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
                ${nestlePercent}%
            </span>
        `;
        
        compCountElement.innerHTML = `
            ${data.compCount}
            <span class="ml-2 px-2 py-1 text-xs font-medium bg-red-100 text-red-800 rounded-full">
                ${compPercent}%
            </span>
        `;

        timestampElement.textContent = formatDate(data.timestamp);

        // Add IQI score display with 2 decimal places
        const iqiColorClass = getIQIColorClass(data.iqi_score);
        const iqiQualityText = getIQIQualityText(data.iqi_score);
        const iqiElement = document.getElementById('modalIQI');
        if (iqiElement) {
            iqiElement.innerHTML = `
                <div class="text-sm text-gray-500 mb-1">Image Quality Index (IQI)</div>
                <div class="font-semibold text-gray-800 text-xl flex items-center">
                    ${parseFloat(data.iqi_score).toFixed(2)}
                    <span class="ml-2 px-2 py-1 text-xs font-medium ${iqiColorClass} rounded-full">
                        ${iqiQualityText}
                    </span>
                </div>
            `;
        }
        
        // Update product breakdown
        const detectedProducts = document.getElementById('detectedProducts');
        detectedProducts.innerHTML = '';
        if (data.products && (data.products.nestle_products || data.products.competitor_products)) {
            // Add Nestle products
            if (data.products.nestle_products && Object.keys(data.products.nestle_products).length > 0) {
                const nestleSection = document.createElement('div');
                nestleSection.innerHTML = `
                    <div class="font-medium text-blue-700 mb-2">Nestlé Products:</div>
                    ${Object.entries(data.products.nestle_products).map(([product, count]) => `
                        <div class="flex justify-between items-center text-sm pl-2 mb-1">
                            <span class="text-gray-700">${product}</span>
                            <span class="font-medium bg-blue-50 px-2 py-1 rounded">${count}</span>
                        </div>
                    `).join('')}
                `;
                detectedProducts.appendChild(nestleSection);
            }
        
            // Add Competitor products
            if (data.products.competitor_products) {
                if (detectedProducts.children.length > 0) {
                    detectedProducts.appendChild(document.createElement('hr'));
                }
                
                const compSection = document.createElement('div');
                compSection.innerHTML = `<div class="font-medium text-red-700 mt-4 mb-2">Uncategorised Products:</div>`;
                
                // Check if competitor_products is array-like (has numeric indices)
                if (Array.isArray(data.products.competitor_products) || 
                    Object.keys(data.products.competitor_products).every(key => !isNaN(parseInt(key)))) {
                    // It's an array or object with numeric keys - show total count
                    const count = Array.isArray(data.products.competitor_products) ? 
                        data.products.competitor_products.length : 
                        Object.keys(data.products.competitor_products).length;
                        
                    compSection.innerHTML += `
                        <div class="flex justify-between items-center text-sm pl-2 mb-1">
                            <span class="text-gray-700">unclassified</span>
                            <span class="font-medium bg-red-50 px-2 py-1 rounded">${count}</span>
                        </div>
                    `;
                } else {
                    // It's a proper object with named keys
                    Object.entries(data.products.competitor_products).forEach(([product, count]) => {
                        compSection.innerHTML += `
                            <div class="flex justify-between items-center text-sm pl-2 mb-1">
                                <span class="text-gray-700">${product}</span>
                                <span class="font-medium bg-red-50 px-2 py-1 rounded">${count}</span>
                            </div>
                        `;
                    });
                }
                
                detectedProducts.appendChild(compSection);
            }
        } else {
            detectedProducts.innerHTML = '<div class="text-sm text-gray-500">No detailed product data available</div>';
        }

        // Update image
        const eventImage = document.querySelector('#eventImage img');
        if (data.image_path) {
            // Ensure we're using the correct path
            eventImage.src = '/' + data.image_path;
            eventImage.classList.remove('hidden');
        } else {
            eventImage.classList.add('hidden');
        }

        // Show modal
        document.getElementById('eventDetailModal').classList.remove('hidden');
        
    } catch (error) {
        console.error('Error viewing event details:', error);
        showToastNotification('Error loading event details');
    }
}

// Add auto-refresh functionality
function startAutoRefresh() {
    // Refresh every 30 seconds
    setInterval(async () => {
        await fetchDashboardData();
        if (currentPage === 1) {
            await fetchEvents();
        }
    }, 30000); // 30 seconds
}

// Fetch event data from server
async function fetchEvents() {
    try {
        let url = `/api/events?page=${currentPage}&limit=${pageSize}`;
        
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Fetched events:', data); // Debug log
        
        if (!data || !data.data) {
            throw new Error('Invalid data format received from server');
        }
        
        // Map the data to ensure consistent property names and correct counts
        events = await Promise.all(data.data.map(async event => {
            try {
                // Fetch detailed data for each event
                const detailResponse = await fetch(`/api/events/${event.id}`);
                if (!detailResponse.ok) {
                    throw new Error(`HTTP error! status: ${detailResponse.status}`);
                }
                const detailData = await detailResponse.json();
                
                // Calculate counts from products data if available
                let nestleCount = 0;
                let compCount = 0;
                
                if (detailData.products) {
                    if (detailData.products.nestle_products) {
                        nestleCount = Object.values(detailData.products.nestle_products)
                            .reduce((sum, count) => sum + count, 0);
                    }
                    
                    if (detailData.products.competitor_products) {
                        if (Array.isArray(detailData.products.competitor_products)) {
                            compCount = detailData.products.competitor_products.length;
                        } else {
                            compCount = Object.values(detailData.products.competitor_products)
                                .reduce((sum, count) => sum + count, 0);
                        }
                    }
                }
                
                return {
                    ...event,
                    ...detailData,
                    nestle_count: nestleCount || detailData.nestle_count || event.nestle_count || 0,
                    competitor_count: compCount || detailData.competitor_count || event.competitor_count || 0,
                    iqi_score: detailData.iqi_score || event.iqi_score || 0
                };
            } catch (error) {
                console.error(`Error fetching details for event ${event.id}:`, error);
                return {
                    ...event,
                    nestle_count: event.nestle_count || 0,
                    competitor_count: event.competitor_count || 0,
                    iqi_score: event.iqi_score || 0
                };
            }
        }));
        
        totalEvents = data.pagination ? data.pagination.total : events.length;
        
        renderEventsTable();
        updatePagination();
        
        // Update the showing count text
        const showingText = document.querySelector('.text-gray-500');
        if (showingText) {
            if (totalEvents > 0) {
                const start = ((currentPage - 1) * pageSize) + 1;
                const end = Math.min(currentPage * pageSize, totalEvents);
                showingText.textContent = `Showing ${start} to ${end} of ${totalEvents} events`;
            } else {
                showingText.textContent = 'No detection events found';
            }
        }
        
    } catch (error) {
        console.error("Error fetching events:", error);
        eventsTableBody.innerHTML = `
            <tr>
                <td colspan="7" class="px-4 py-4 text-center text-red-500">
                    Error loading detection events: ${error.message}
                </td>
            </tr>
        `;
    }
}

// Tambahkan CSS untuk animasi toast
const style = document.createElement('style');
style.textContent = `
.toast-notification {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: white;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transform: translateX(100%);
    transition: transform 0.3s ease-in-out;
    z-index: 50;
    max-width: 400px;
    border-left: 4px solid #3B82F6;
}

.toast-notification.show {
    transform: translateX(0);
}
`;
document.head.appendChild(style);

// Add CSS for location display
const locationStyles = document.createElement('style');
locationStyles.textContent = `
.location-container {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.location-address {
    font-weight: 500;
    color: #374151;
}

.location-coords {
    font-family: monospace;
}
`;
document.head.appendChild(locationStyles);
