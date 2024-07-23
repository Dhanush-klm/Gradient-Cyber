document.addEventListener('DOMContentLoaded', () => {
    const fileUpload = document.getElementById('file-upload');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadStatus = document.getElementById('upload-status');
    const filePreview = document.getElementById('file-preview');
    const queryInput = document.getElementById('query-input');
    const queryBtn = document.getElementById('query-btn');
    const queryResult = document.getElementById('query-result');

    uploadBtn.addEventListener('click', () => {
        const file = fileUpload.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const data = new Uint8Array(e.target.result);
                const workbook = XLSX.read(data, {type: 'array'});
                const firstSheetName = workbook.SheetNames[0];
                const worksheet = workbook.Sheets[firstSheetName];
                const jsonData = XLSX.utils.sheet_to_json(worksheet);

                // Display preview
                filePreview.innerHTML = `<h3>Preview (first 5 rows):</h3>
                    <pre>${JSON.stringify(jsonData.slice(0, 5), null, 2)}</pre>`;

                // Send data to backend
                uploadToBackend(jsonData);
            };
            reader.readAsArrayBuffer(file);
        } else {
            uploadStatus.textContent = 'Please select a file first.';
        }
    });

    queryBtn.addEventListener('click', () => {
        const query = queryInput.value;
        if (query) {
            sendQueryToBackend(query);
        } else {
            queryResult.textContent = 'Please enter a query.';
        }
    });

    function uploadToBackend(data) {
        fetch('/api/upload', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(result => {
            uploadStatus.textContent = `Upload successful. Tokens used: ${result.totalTokensUsed}, Requests made: ${result.totalRequests}`;
        })
        .catch(error => {
            console.error('Error:', error);
            uploadStatus.textContent = `Error uploading data: ${error.message}`;
        });
    }

    function sendQueryToBackend(query) {
        fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({query}),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(result => {
            if (result.error) {
                throw new Error(result.error);
            }
            queryResult.innerHTML = `<h3>Answer:</h3><p>${result.answer}</p>`;
        })
        .catch(error => {
            console.error('Error:', error);
            queryResult.textContent = `Error processing query: ${error.message}`;
        });
    }
});