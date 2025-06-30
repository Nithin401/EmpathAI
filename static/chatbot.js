document.addEventListener('DOMContentLoaded', () => {
    const sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');
    const sidebar = document.getElementById('sidebar');
    const newChatBtn = document.getElementById('new-chat-btn');
    const chatSessionsList = document.getElementById('chat-sessions-list');
    const deletedChatSessionsList = document.getElementById('deleted-chat-sessions-list');
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const currentChatTitle = document.getElementById('current-chat-title');
    const summarizeBtn = document.getElementById('summarize-chat-btn');
    const questionBtn = document.getElementById('insightful-question-btn');
    const copingBtn = document.getElementById('coping-mechanism-btn');
    const reframeBtn = document.getElementById('positive-reframe-btn');
    const deleteBtn = document.getElementById('delete-chat-btn');
    const uploadImageBtn = document.getElementById('upload-image-btn');
    const cameraBtn = document.getElementById('camera-btn');
    const confirmationModal = document.getElementById('confirmation-modal');
    const infoModal = document.getElementById('info-modal');
    const confirmationModalTitle = document.getElementById('confirmation-modal-title');
    const confirmationModalMessage = document.getElementById('confirmation-modal-message');
    const modalConfirmBtn = document.getElementById('modal-confirm-btn');
    const modalCancelBtn = document.getElementById('modal-cancel-btn');
    const infoModalTitle = document.getElementById('info-modal-title');
    const infoModalMessage = document.getElementById('info-modal-message');
    const infoModalCloseBtn = document.getElementById('info-modal-close-btn');
    const cameraModal = document.getElementById('camera-modal');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('capture-btn');
    const submitCameraBtn = document.getElementById('submit-camera-btn');
    const cameraCancelBtn = document.getElementById('camera-cancel-btn');

    let currentChatId = null;
    let videoStream = null;

    // Sidebar toggle for mobile
    sidebarToggleBtn.addEventListener('click', () => {
        sidebar.classList.toggle('active');
        sidebarToggleBtn.querySelector('.fa-bars').classList.toggle('hidden');
        sidebarToggleBtn.querySelector('.fa-times').classList.toggle('hidden');
    });

    // Load chat sessions
    function loadChatSessions() {
        fetch('/get_chat_sessions')
            .then(response => response.json())
            .then(data => {
                chatSessionsList.innerHTML = '';
                deletedChatSessionsList.innerHTML = '';
                data.active.forEach(session => {
                    const li = document.createElement('li');
                    li.dataset.chatId = session.id;
                    li.innerHTML = `<span class="chat-preview">${session.preview}</span><span class="chat-timestamp">${new Date(session.timestamp).toLocaleString()}</span>`;
                    li.addEventListener('click', () => loadChatHistory(session.id));
                    chatSessionsList.appendChild(li);
                });
                data.deleted.forEach(session => {
                    const li = document.createElement('li');
                    li.dataset.chatId = session.id;
                    li.innerHTML = `<span class="chat-preview">${session.preview}</span><button class="restore-btn">Restore</button>`;
                    li.querySelector('.restore-btn').addEventListener('click', () => restoreChat(session.id));
                    deletedChatSessionsList.appendChild(li);
                });
                if (currentChatId) {
                    document.querySelector(`[data-chat-id="${currentChatId}"]`)?.classList.add('active');
                }
            })
            .catch(error => {
                console.error('Error loading chat sessions:', error);
                showInfoModal('Error', 'Failed to load chat sessions.');
            });
    }

    // Load chat history
    function loadChatHistory(chatId) {
        currentChatId = chatId;
        chatMessages.innerHTML = '';
        document.getElementById('message-loading-indicator').style.display = 'block';
        document.querySelectorAll('.chat-sessions-list li').forEach(li => li.classList.remove('active'));
        document.querySelector(`[data-chat-id="${chatId}"]`)?.classList.add('active');
        fetch(`/get_history?chat_id=${chatId}`)
            .then(response => response.json())
            .then(messages => {
                document.getElementById('message-loading-indicator').style.display = 'none';
                messages.forEach(msg => displayMessage(msg.sender, msg.message));
                currentChatTitle.textContent = messages.length ? messages[0].message.slice(0, 30) + '...' : 'Chat';
                chatMessages.scrollTop = chatMessages.scrollHeight;
            })
            .catch(error => {
                console.error('Error loading history:', error);
                showInfoModal('Error', 'Failed to load chat history.');
            });
    }

    // Display message
    function displayMessage(sender, message) {
        const div = document.createElement('div');
        div.className = `message ${sender}`;
        div.textContent = message;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Send message
    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;
        displayMessage('user', message);
        userInput.value = '';
        document.getElementById('typing-animation').style.display = 'block';
        fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, chat_id: currentChatId || 'new' })
        })
            .then(response => response.json())
            .then(data => {
                document.getElementById('typing-animation').style.display = 'none';
                displayMessage('bot', data.response);
                currentChatId = data.chat_id;
                loadChatSessions();
            })
            .catch(error => {
                document.getElementById('typing-animation').style.display = 'none';
                console.error('Error sending message:', error);
                showInfoModal('Error', 'Failed to send message.');
            });
    }

    // New chat
    newChatBtn.addEventListener('click', () => {
        currentChatId = null;
        chatMessages.innerHTML = '';
        currentChatTitle.textContent = 'New Chat';
        document.querySelectorAll('.chat-sessions-list li').forEach(li => li.classList.remove('active'));
    });

    // Send message on button click or Enter key
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // Show confirmation modal
    function showConfirmationModal(title, message, onConfirm) {
        confirmationModalTitle.textContent = title;
        confirmationModalMessage.textContent = message;
        confirmationModal.style.display = 'flex';
        modalConfirmBtn.onclick = () => {
            onConfirm();
            confirmationModal.style.display = 'none';
        };
        modalCancelBtn.onclick = () => {
            confirmationModal.style.display = 'none';
        };
    }

    // Show info modal
    function showInfoModal(title, message) {
        infoModalTitle.textContent = title;
        infoModalMessage.textContent = message;
        infoModal.style.display = 'flex';
        infoModalCloseBtn.onclick = () => {
            infoModal.style.display = 'none';
        };
    }

    // Delete chat
    deleteBtn.addEventListener('click', () => {
        if (!currentChatId) {
            showInfoModal('Info', 'Please select a chat to delete.');
            return;
        }
        showConfirmationModal('Delete Chat', 'Are you sure you want to delete this chat?', () => {
            fetch(`/delete_chat_session/${currentChatId}`, { method: 'DELETE' })
                .then(response => response.json())
                .then(data => {
                    showInfoModal('Success', data.message);
                    currentChatId = null;
                    chatMessages.innerHTML = '';
                    currentChatTitle.textContent = 'Select a Chat or Start New';
                    loadChatSessions();
                })
                .catch(error => {
                    console.error('Error deleting chat:', error);
                    showInfoModal('Error', 'Failed to delete chat.');
                });
        });
    });

    // Restore chat
    function restoreChat(chatId) {
        showConfirmationModal('Restore Chat', 'Are you sure you want to restore this chat?', () => {
            fetch(`/restore_chat_session/${chatId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
                .then(response => response.json())
                .then(data => {
                    showInfoModal('Success', data.message);
                    loadChatSessions();
                })
                .catch(error => {
                    console.error('Error restoring chat:', error);
                    showInfoModal('Error', 'Failed to restore chat.');
                });
        });
    }

    // Summarize chat
    summarizeBtn.addEventListener('click', () => {
        if (!currentChatId) {
            showInfoModal('Info', 'Please select a chat to summarize.');
            return;
        }
        fetch('/summarize_chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chat_id: currentChatId })
        })
            .then(response => response.json())
            .then(data => {
                showInfoModal('Chat Summary', data.summary || 'No summary available.');
            })
            .catch(error => {
                console.error('Error summarizing chat:', error);
                showInfoModal('Error', 'Failed to summarize chat.');
            });
    });

    // Generate insightful question
    questionBtn.addEventListener('click', () => {
        if (!currentChatId) {
            showInfoModal('Info', 'Please select a chat to generate a question.');
            return;
        }
        fetch('/generate_insightful_question', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chat_id: currentChatId })
        })
            .then(response => response.json())
            .then(data => {
                showInfoModal('Insightful Question', data.question || 'No question generated.');
            })
            .catch(error => {
                console.error('Error generating question:', error);
                showInfoModal('Error', 'Failed to generate question.');
            });
    });

    // Suggest coping mechanism
    copingBtn.addEventListener('click', () => {
        if (!currentChatId) {
            showInfoModal('Info', 'Please select a chat to suggest a coping mechanism.');
            return;
        }
        fetch('/suggest_coping_mechanism', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chat_id: currentChatId })
        })
            .then(response => response.json())
            .then(data => {
                showInfoModal('Coping Mechanism', data.suggestion || 'No suggestion available.');
            })
            .catch(error => {
                console.error('Error suggesting coping mechanism:', error);
                showInfoModal('Error', 'Failed to suggest coping mechanism.');
            });
    });

    // Positive reframe
    reframeBtn.addEventListener('click', () => {
        if (!currentChatId) {
            showInfoModal('Info', 'Please select a chat to reframe.');
            return;
        }
        fetch('/reframe_positively', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chat_id: currentChatId })
        })
            .then(response => response.json())
            .then(data => {
                showInfoModal('Positive Reframe', data.reframe || 'No reframe available.');
            })
            .catch(error => {
                console.error('Error reframing:', error);
                showInfoModal('Error', 'Failed to reframe message.');
            });
    });

    // Image upload
    uploadImageBtn.addEventListener('click', () => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('file', file);
            formData.append('chat_id', currentChatId || 'new');
            fetch('/upload_file', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    displayMessage('user', data.response);
                    currentChatId = data.chat_id;
                    loadChatSessions();
                })
                .catch(error => {
                    console.error('Error uploading image:', error);
                    showInfoModal('Error', 'Failed to upload image.');
                });
        };
        input.click();
    });

    // Camera capture
    cameraBtn.addEventListener('click', () => {
        cameraModal.style.display = 'flex';
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                videoStream = stream;
                video.srcObject = stream;
            })
            .catch(error => {
                console.error('Error accessing camera:', error);
                showInfoModal('Error', 'Failed to access camera.');
                cameraModal.style.display = 'none';
            });
    });

    captureBtn.addEventListener('click', () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        video.style.display = 'none';
        canvas.style.display = 'block';
        captureBtn.style.display = 'none';
        submitCameraBtn.style.display = 'inline-block';
    });

    submitCameraBtn.addEventListener('click', () => {
        canvas.toBlob(blob => {
            const formData = new FormData();
            formData.append('file', blob, 'capture.jpg');
            formData.append('chat_id', currentChatId || 'new');
            fetch('/upload_file', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    displayMessage('user', data.response);
                    currentChatId = data.chat_id;
                    loadChatSessions();
                    closeCameraModal();
                })
                .catch(error => {
                    console.error('Error uploading camera image:', error);
                    showInfoModal('Error', 'Failed to upload camera image.');
                    closeCameraModal();
                });
        }, 'image/jpeg');
    });

    cameraCancelBtn.addEventListener('click', closeCameraModal);

    function closeCameraModal() {
        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
            videoStream = null;
        }
        video.style.display = 'block';
        canvas.style.display = 'none';
        captureBtn.style.display = 'inline-block';
        submitCameraBtn.style.display = 'none';
        cameraModal.style.display = 'none';
    }

    // Initial load
    loadChatSessions();
});