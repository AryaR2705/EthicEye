document.addEventListener('DOMContentLoaded', function() {
    var sendLinkButton = document.getElementById('sendLinkButton');
    var statusMessage = document.getElementById('statusMessage');

    sendLinkButton.addEventListener('click', function() {
        chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
            var link = tabs[0].url; // Get the current tab's URL
            console.log("Sending link:", link);  // Log the link being sent
            sendLinkToBackend(link);
        });
    });

    function sendLinkToBackend(link) {
        fetch('http://localhost:8000/scrape', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ link: link })
        })
        .then(response => {
            console.log("Response status:", response.status);  // Log the response status
            return response.text();  // Return the response text
        })
        .then(data => {
            console.log("Response data:", data);  // Log the response data
            statusMessage.innerText = data;
            // Send a message to the content script to highlight the scraped text
            chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
                chrome.tabs.sendMessage(tabs[0].id, { action: 'highlightText', text: data });
            });
        })
        .catch(error => {
            statusMessage.innerText = 'Error sending link';
            console.error('Error:', error);
        });
    }
});
