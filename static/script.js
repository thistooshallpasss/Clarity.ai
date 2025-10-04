document.getElementById('imageUpload').addEventListener('change', function (event) {
    const file = event.target.files[0];
    if (!file) {
        return;
    }

    // Get DOM elements
    const originalImage = document.getElementById('originalImage');
    const upscaledImage = document.getElementById('upscaledImage');
    const resultsDiv = document.getElementById('results');
    const loader = document.getElementById('loader');
    const loadingText = document.getElementById('loading-text');

    // Show original image preview
    originalImage.src = URL.createObjectURL(file);
    resultsDiv.style.display = 'flex';
    upscaledImage.src = ""; // Clear previous result

    // Show loader
    loader.style.display = 'block';
    loadingText.style.display = 'block';

    // Prepare data for API
    const formData = new FormData();
    formData.append('image', file);

    // Call the Flask API
    fetch('/upscale', {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.blob();
        })
        .then(imageBlob => {
            // Display the upscaled image
            const imageUrl = URL.createObjectURL(imageBlob);
            upscaledImage.src = imageUrl;

            // Hide loader
            loader.style.display = 'none';
            loadingText.style.display = 'none';
        })
        .catch(error => {
            console.error('Error:', error);
            loadingText.innerText = "Sorry, an error occurred. Please try again.";
            loader.style.display = 'none';
        });
});