// Fetch and update dashboard stats
async function updateStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();

        // Update stat cards with animation
        animateNumber('active-users', data.active_users);
        animateNumber('total-sessions', data.total_sessions);
        animateNumber('brain-scans', data.brain_scans);
        animateNumber('lung-scans', data.lung_scans);
    } catch (error) {
        console.error('Failed to fetch stats:', error);
    }
}

// Animate number counting
function animateNumber(elementId, targetValue) {
    const element = document.getElementById(elementId);
    const startValue = parseInt(element.innerText) || 0;
    const duration = 1000; // 1 second
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function
        const easeOutQuad = progress * (2 - progress);
        const currentValue = Math.floor(startValue + (targetValue - startValue) * easeOutQuad);

        element.innerText = currentValue;

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            element.innerText = targetValue;
        }
    }

    requestAnimationFrame(update);
}

// Update stats on page load
document.addEventListener('DOMContentLoaded', () => {
    updateStats();

    // Refresh stats every 10 seconds
    setInterval(updateStats, 10000);
});
