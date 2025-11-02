
        const apiUrl = "http://127.0.0.1:8000"; 
        // const apiUrl = "api_key";
        const inputText = document.getElementById('inputText');
        const outputText = document.getElementById('outputText');
        const translateBtn = document.getElementById('translateBtn');
        const clearBtn = document.getElementById('clearBtn');
        const statusDiv = document.getElementById('status');
        const charCount = document.getElementById('charCount');

        inputText.addEventListener('input', () => {
            charCount.textContent = `${inputText.value.length} characters`;
        });

        function showStatus(message, type) {
            statusDiv.textContent = message;
            statusDiv.className = `status ${type}`;
            statusDiv.style.display = 'block';

            if (type !== 'loading') {
                setTimeout(() => {
                    statusDiv.style.display = 'none';
                }, 5000);
            }
        }

        function showLoadingStatus() {
            statusDiv.innerHTML = '<span class="loading-spinner"></span>Translating...';
            statusDiv.className = 'status loading';
            statusDiv.style.display = 'block';
        }

        async function translateText() {
            const text = inputText.value.trim();

            if (!text) {
                showStatus('Please enter some Bangla text to translate', 'error');
                return;
            }

            if (!apiUrl) {
                showStatus('Please enter your API URL', 'error');
                return;
            }

            translateBtn.disabled = true;
            showLoadingStatus();

            try {
                const response = await fetch(`${apiUrl}/translate`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: text,
                        
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                outputText.value = data.translation;
                showStatus('Translation completed successfully!', 'success');
            } catch (err) {
                showStatus(`Translation failed: ${err.message}. Please check your API URL and try again.`, 'error');
                console.error('Error:', err);
            } finally {
                translateBtn.disabled = false;
            }
        }

        function clearAll() {
            inputText.value = '';
            outputText.value = '';
            charCount.textContent = '0 characters';
            statusDiv.style.display = 'none';
        }

        translateBtn.addEventListener('click', translateText);
        clearBtn.addEventListener('click', clearAll);

        inputText.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                translateText();
            }
        });
