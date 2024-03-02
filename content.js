// content.js

// Listen for messages from the popup script
chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
    if (message.action === 'highlightText') {
        highlightText(message.text);
    }
});

// Function to highlight text
function highlightText(text) {
    // Find all occurrences of the scraped text on the page
    var elements = document.querySelectorAll(":contains(" + text + ")");

    // Apply highlighting to each occurrence
    elements.forEach(function(element) {
        element.style.backgroundColor = "yellow";
    });
}
