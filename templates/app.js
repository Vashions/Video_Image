let selectedDesign = "";
let currentDesign = null;
let stream;
let isStreamActive = false;
let currentNecklacePath = "";
let currentEarringPath = "";
let currentIsLarge = false; // Default value
let currentBanglePath = ''; // Default value
let currentRingPath = ''; // Default value
const FRAME_THROTTLE = 1000 / 120;
const frameBuffer = [];
const bufferSize = 5;
let lastFrameTime = 0;
const targetFPS = 120;


let currentJewelryPaths = {
    necklace: "",
    earring: "",
    bangle: "",
    ring: ""
};

function generateNecklaceGallery() {
    const necklaces = [
    
        // {file: 'new_2_choker.png', name: 'New 1'},
        // {file: 'new 1.png', name: 'New 2'},
        {file: 'multani1.jpg', name: 'multani jewellers'},
        {file: 'multani2.jpg', name: 'multani jewellers'},
        {file: 'multani3.jpg', name: 'multani jewellers'},
        {file: 'multani4.jpg', name: 'multani jewellers'},
        {file: 'multani5.jpg', name: 'multani jewellers'},
        {file: 'multani6.jpg', name: 'multani jewellers'},
        {file: 'multani7.jpg', name: 'multani jewellers'},
        {file: 'multani8.jpg', name: 'multani jewellers'},
        {file: 'multani9.jpg', name: 'multani jewellers'},
        {file: 'necklace_1.png', name: 'Necklace 1'},
        {file: 'necklace_13.png', name: 'Necklace 2'},
        {file: 'necklace_81.png', name: 'Necklace 3'},
        {file: 'necklace_18.png', name: 'Necklace 4'},
        {file: 'necklace_20.png', name: 'Necklace 5'},
        {file: 'necklace_21_choker.png', name: 'Necklace 6'},
        {file: 'necklace_23.png', name: 'Necklace 7'},
        {file: 'necklace_24.png', name: 'Necklace 8'},
        {file: 'necklace_26.png', name: 'Necklace 9'},
        {file: 'necklace_27.png', name: 'Necklace 10'},
        {file: 'necklace_28.png', name: 'Necklace 11'},
        {file: 'necklace_29.png', name: 'Necklace 12'},
        {file: 'necklace_30.png', name: 'Necklace 13'},
        {file: 'necklace_47.png', name: 'Necklace 14'},
        {file: 'necklace_65.png', name: 'Necklace 15'},
        {file: 'necklace_70.png', name: 'Necklace 16'},
        {file: 'necklace_75.png', name: 'Necklace 17'},
        {file: 'necklace_76.png', name: 'Necklace 18'},
        {file: 'necklace_78.png', name: 'Necklace 19'},
        {file: 'necklace_72.png', name: 'Necklace 20'}
        
    ];


    const container = document.getElementById('necklaceContainer');
    necklaces.forEach(necklace => {
        const div = document.createElement('div');
        div.className = 'image-container';
        // Check if the necklace file name contains '_large'
        const isLarge = necklace.file.includes('_large');
        div.onclick = () => selectNecklace(necklace.file, isLarge);

        const img = document.createElement('img');
        img.src = `/static/Image/Necklace/${necklace.file}`;
        img.alt = necklace.name;

        const p = document.createElement('p');
        p.textContent = necklace.name + (isLarge ? ' (Large)' : '');

        div.appendChild(img);
        div.appendChild(p);
        container.appendChild(div);
    });
}
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}




function selectNecklace(necklaceName) {
    const isLarge = necklaceName.includes('_large');

    fetch(`/select_necklace_image/${necklaceName}`, {
        method: 'GET',
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentDesign = "NecklaceDesign";
            currentNecklacePath = data.path;
            currentIsLarge = isLarge;

            if (!isMultiJewelEnabled) {
                // Clear other selections only if multi-jewel is disabled
                currentEarringPath = "";
                currentBanglePath = "";
                currentRingPath = "";
            }

            scrollToTop();

            // Update UI for necklace
            const selectedNecklaceImage = document.getElementById('selectedNecklaceImage');
            selectedNecklaceImage.src = currentNecklacePath;
            document.getElementById('selectedNecklaceDisplay').style.display = 'block';

            if (!isMultiJewelEnabled) {
                // Hide other displays only if multi-jewel is disabled
                document.getElementById('selectedBangleDisplay').style.display = 'none';
                document.getElementById('selectedRingDisplay').style.display = 'none';
                document.getElementById('selectedEarringDisplay').style.display = 'none';
            }

            const processedVideo = document.getElementById('processed-video');
            processedVideo.style.display = 'none';

            if (stream && stream.active) {
                // Ensure proper frame processing initialization
                setTimeout(() => {
                    sendFramesToServer();
                }, 100);
            }

            // Force immediate processing for multi-jewel mode
            if (isMultiJewelEnabled) {
                setTimeout(() => {
                    sendFramesToServer();
                }, 100);
            }

            console.log(`Selected necklace: ${necklaceName}, Large: ${isLarge}, MultiJewel: ${isMultiJewelEnabled}`);
        } else {
            console.error('Failed to select necklace:', data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}


// Add camera initialization function
async function initializeCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.getElementById('video');
        video.srcObject = stream;
        video.classList.add('mirror-video');
        document.getElementById('videoContainer').classList.remove('hidden');
        isStreamActive = true;
        return true;
    } catch (error) {
        console.error("Camera initialization failed:", error);
        return false;
    }
}


function showProcessedVideo() {
    const processedVideo = document.getElementById('processed-video');
    processedVideo.classList.remove('hidden');
}



function generateEarringGallery() {
    const earrings = [
        {file: 'earring_1.png', name: 'Earring 1'},
        {file: 'earring_2.png', name: 'Earring 2'},
        {file: 'earring_3.png', name: 'Earring 3'},
        {file: 'earring_4.png', name: 'Earring 4'},
        {file: 'earring_5.png', name: 'Earring 5'},
    ];

    const container = document.getElementById('earringContainer');
    earrings.forEach(earring => {
        const div = document.createElement('div');
        div.className = 'image-container';
        div.onclick = () => selectEarring(earring.file);

        const img = document.createElement('img');
        img.src = `/static/Image/Earring/${earring.file}`;
        img.alt = earring.name;

        const p = document.createElement('p');
        p.textContent = earring.name;

        div.appendChild(img);
        div.appendChild(p);
        container.appendChild(div);
    });
}


function selectEarring(earringName) {
    fetch(`/select_earring_image/${earringName}`, {
        method: 'GET',
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentDesign = "EarringDesign";
            currentEarringPath = data.path;
            
        

            if (!isMultiJewelEnabled) {
                currentNecklacePath = "";
                currentBanglePath = "";
                currentRingPath = "";
            }

            scrollToTop();
            
            const selectedEarringImage = document.getElementById('selectedEarringImage');
            selectedEarringImage.src = currentEarringPath;
            document.getElementById('selectedEarringDisplay').style.display = 'block';
            
            if (!isMultiJewelEnabled) {
                document.getElementById('selectedNecklaceDisplay').style.display = 'none';
                document.getElementById('selectedBangleDisplay').style.display = 'none';
                document.getElementById('selectedRingDisplay').style.display = 'none';
            }

            const processedVideo = document.getElementById('processed-video');
            processedVideo.style.display = 'none';


            if (stream && stream.active) {
                setTimeout(() => {
                    sendFramesToServer();
                }, 100);
            }

            updateJewelrySelection('Earring', earringName);

            console.log(`Selected earring: ${earringName}, MultiJewel: ${isMultiJewelEnabled}`);
            updateJewelrySelection('earring', earringName);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function generateBangleGallery() {
    const bangles = [
        { file: 'bangle_1.png', name: 'Bangle 1' },
        { file: 'bangle_2.png', name: 'Bangle 2' },
        { file: 'bangle_3.png', name: 'Bangle 3' },
        { file: 'bangle_4.png', name: 'Bangle 4' },
        { file: 'bangle_5.png', name: 'Bangle 5' },
    ];

    const container = document.getElementById('bangleContainer');
    bangles.forEach(bangle => {
        const div = document.createElement('div');
        div.className = 'image-container';
        div.onclick = () => selectBangle(bangle.file);

        const img = document.createElement('img');
        img.src = `/static/Image/Bangle/${bangle.file}`;
        img.alt = bangle.name;

        const p = document.createElement('p');
        p.textContent = bangle.name;

        div.appendChild(img);
        div.appendChild(p);
        container.appendChild(div);
    });
}

function selectBangle(bangleName) {
    const isLarge = bangleName.includes('_large');

    fetch(`/select_bangle_image/${bangleName}`, {
        method: 'GET',
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentDesign = "BangleDesign";
            currentBanglePath = data.path;

            if (!isMultiJewelEnabled) {
                // Clear other selections only if multi-jewel is disabled
                currentNecklacePath = "";
                currentEarringPath = "";
                currentRingPath = "";
            }

            scrollToTop();

            // Update UI for bangle
            const selectedBangleImage = document.getElementById('selectedBangleImage');
            selectedBangleImage.src = currentBanglePath;
            document.getElementById('selectedBangleDisplay').style.display = 'block';

            if (!isMultiJewelEnabled) {
                // Hide other displays only if multi-jewel is disabled
                document.getElementById('selectedNecklaceDisplay').style.display = 'none';
                document.getElementById('selectedEarringDisplay').style.display = 'none';
                document.getElementById('selectedRingDisplay').style.display = 'none';
            }

            const processedVideo = document.getElementById('processed-video');
            processedVideo.style.display = 'none';

            if (stream && stream.active) {
                sendFramesToServer();
            }

            console.log(`Selected bangle: ${bangleName}, Large: ${isLarge}, MultiJewel: ${isMultiJewelEnabled}`);
        } else {
            console.error('Failed to select bangle:', data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function generateRingGallery() {
    const rings = [
        {file: 'ring_1.png', name: 'Ring 1'},
        {file: 'ring_2.png', name: 'Ring 2'},
        {file: 'ring_3.png', name: 'Ring 3'},
        {file: 'ring_4.png', name: 'Ring 4'},
        {file: 'ring_5.png', name: 'Ring 5'},
        {file: 'ring_6.png', name: 'Ring 6'},
        {file: 'ring_7.png', name: 'Ring 7'},
    ];

    const container = document.getElementById('ringContainer');
    rings.forEach(ring => {
        const div = document.createElement('div');
        div.className = 'image-container';
        div.onclick = () => selectRing(ring.file);

        const img = document.createElement('img');
        img.src = `/static/Image/Ring/${ring.file}`;
        img.alt = ring.name;

        const p = document.createElement('p');
        p.textContent = ring.name;

        div.appendChild(img);
        div.appendChild(p);
        container.appendChild(div);
    });
}

function selectRing(ringName) {
    const isLarge = ringName.includes('_large');

    fetch(`/select_ring_image/${ringName}`, {
        method: 'GET',
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentDesign = "RingDesign";
            currentRingPath = data.path;
            currentIsLarge = isLarge;

            if (!isMultiJewelEnabled) {
                currentNecklacePath = "";
                currentEarringPath = "";
                currentBanglePath = "";
            }

            scrollToTop();

            // Update UI for ring
            const selectedRingImage = document.getElementById('selectedRingImage');
            selectedRingImage.src = currentRingPath;
            document.getElementById('selectedRingDisplay').style.display = 'block';

            if (!isMultiJewelEnabled) {
                // Hide other displays only if multi-jewel is disabled
                document.getElementById('selectedNecklaceDisplay').style.display = 'none';
                document.getElementById('selectedBangleDisplay').style.display = 'none';
                document.getElementById('selectedEarringDisplay').style.display = 'none';
            }

            const processedVideo = document.getElementById('processed-video');
            processedVideo.style.display = 'none';

            // Reset video display
            const video = document.getElementById('video');
            video.style.display = 'block';

            if (stream && stream.active) {
                // Ensure proper frame processing initialization
                setTimeout(() => {
                    sendFramesToServer();
                }, 100);
            }

            // Update jewelry selection tracking
            updateJewelrySelection('ring', ringName);

            console.log(`Selected ring: ${ringName}, Large: ${isLarge}, MultiJewel: ${isMultiJewelEnabled}`);
        } else {
            console.error('Failed to select ring:', data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function initializeMultiJewelProcessing() {
    const multiJewelToggle = document.getElementById('multiJewelToggle');
    if (multiJewelToggle && multiJewelToggle.checked) {
        isMultiJewelEnabled = true;
        if (stream && stream.active) {
            sendFramesToServer();
        }
    }
}



document.addEventListener('DOMContentLoaded', () => {
    generateNecklaceGallery();
    generateEarringGallery();
    generateBangleGallery();
    generateRingGallery();
    initializeMultiJewelProcessing();
});



function updateJewelrySelection(type, name) {
    const selectionElement = document.getElementById(`selected${type.charAt(0).toUpperCase() + type.slice(1)}Name`);
    if (selectionElement) {
        selectionElement.textContent = `Selected ${type}: ${name}`;
    }
}

// Improved selectDesign function
function selectDesign(design) {
    currentDesign = design; 
    console.log(`Design selected: ${design}`);

    const jewelryPath = getCurrentJewelryPath(); // 

    fetch(`/update_design?design=${design}`, { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log(`Design updated successfully to: ${design}`);
                if (stream && stream.active) {
                    sendFramesToServer();
                }
            } else {
                console.error('Failed to update design on the server:', data.error || 'Unknown error');
            }
        })
        .catch(error => {
        });
}

// Ensure the jewelry type and design are properly initialized
document.addEventListener('DOMContentLoaded', () => {
    if (!currentDesign) {
        currentDesign = 'NecklaceDesign'; // Default design to necklace
        console.log('Defaulting to NecklaceDesign on page load.');
    }

    // Trigger initial frame processing for the default design
    const initialJewelryPath = getCurrentJewelryPath();
    console.log('Initial jewelry path:', initialJewelryPath);
    sendFramesToServer();
});

document.getElementById('cameraButton').addEventListener('click', async () => {
    if (!isStreamActive) {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.getElementById('video');
        video.srcObject = stream;
        video.classList.add('mirror-video');
        document.getElementById('video').srcObject = stream;
        document.getElementById('cameraButton').textContent = 'End Camera';
        isStreamActive = true;
        document.getElementById('videoContainer').classList.remove('hidden');
        sendFramesToServer();
    } catch (error) {
        console.error("Error accessing the camera: ", error);
    }
    } else {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        document.getElementById('video').srcObject = null;
        document.getElementById('cameraButton').textContent = 'Live Try on';
        isStreamActive = false;
        document.getElementById('videoContainer').classList.add('hidden');
    }
    }
});


function selectJewelry(type, fileName) {
    fetch(`/select_${type}_image/${fileName}`, { method: 'GET' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                currentJewelryPaths[type] = data.path;
                if (stream && stream.active) {
                    sendFramesToServer();
                }
                updateJewelryUI(type, fileName);
            } else {
                console.error(`Failed to select ${type}:`, data.message);
            }
        })
        .catch(error => console.error("Error:", error));
}



async function sendFramesToServer() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { alpha: false }); // Performance optimization
    const video = document.getElementById('video');
    const processedVideo = document.getElementById('processed-video');

    // Mirror video elements for better UX
    video.classList.add('mirror-video');
    processedVideo.classList.add('mirror-video');

    let processingTimes = { local: [], client: [] };
    const frameBuffer = [];
    const bufferSize = 10;
    const FRAME_THROTTLE = 100;
    // let lastProcessedTime = performance.now();

    async function processFrame() {
        if (stream && stream.active) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;     
            ctx.drawImage(video, 0, 0);       
            // const now = performance.now();
            // const elapsed = now - lastProcessedTime;
            // if (elapsed < FRAME_THROTTLE) {
            //     requestAnimationFrame(processFrame);
            //     return;
            // }
            // lastProcessedTime = now;
    
            // // Get original video dimensions and viewport
            // const originalWidth = video.videoWidth;
            // const originalHeight = video.videoHeight;
            // const viewportWidth = video.offsetWidth;
            // const viewportHeight = video.offsetHeight;
    
            // Match canvas exactly to video's display size
            // canvas.width = viewportWidth;
            // canvas.height = viewportHeight;

            // Draw video at viewport size to match display exactly

            
            // // Calculate scaling to preserve content proportions
            // const scale = Math.min(canvas.width / originalWidth, canvas.height / originalHeight);
            // const scaledWidth = originalWidth * scale;
            // const scaledHeight = originalHeight * scale;
            // const offsetX = (canvas.width - scaledWidth) / 2;
            // const offsetY = (canvas.height - scaledHeight) / 2;
    
            // // Clear canvas and draw with proper scaling
            // ctx.fillStyle = '#000';
            // ctx.fillRect(0, 0, canvas.width, canvas.height);
            // ctx.drawImage(video, offsetX, offsetY, scaledWidth, scaledHeight);
    
            // Convert canvas to a compressed JPEG blob
            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
            const formData = new FormData();
            formData.append('frame', blob);
            formData.append('multi_jewel_enabled', isMultiJewelEnabled);
            // formData.append('original_width', originalWidth);
            // formData.append('original_height', originalHeight);
            // formData.append('aspect_ratio', originalAspectRatio);
            // formData.append('scale_factor', scale);
    
            // Append jewelry details based on the selected mode
            if (isMultiJewelEnabled) {
                appendMultiJewelData(formData);
            } else {
                await appendSingleJewelData(formData);
            }
    
            try {
                // Manage frame buffer for smoother processing
                // if (frameBuffer.length >= bufferSize) {
                //     frameBuffer.shift();
                // }
                // frameBuffer.push(formData);
    
                // Send frame to the server for processing
                const response = await fetch('/process_frame', {
                    method: 'POST',
                    body: formData
                });
    
                if (!response.ok) throw new Error('Network response was not ok');
    
                // Handle server response and update UI
                const frameData = await response.json();
    
                processedVideo.src = `data:image/jpeg;base64,${frameData.image}`;
                processedVideo.style.display = 'block';
                video.style.display = 'none';                                 
            } catch (error) {
                console.error('Error processing frame:', error);
                processedVideo.style.display = 'none';
                video.style.display = 'block'
            }
    
            requestAnimationFrame(processFrame);
        } else {
            processedVideo.style.display = 'none';
            video.style.display = 'block';
        }
    }
        
    function appendMultiJewelData(formData) {
        formData.append('design', currentDesign || '');

        if (currentNecklacePath) {
            formData.append('necklace_path', currentNecklacePath);
            formData.append('necklace_type', 'necklace');
            formData.append('is_necklace_large', currentIsLarge ? 'true' : 'false');
        }

        if (currentEarringPath) {
            formData.append('earring_path', currentEarringPath);
            formData.append('earring_type', 'earring');
        }

        if (currentBanglePath) {
            formData.append('bangle_path', currentBanglePath);
            formData.append('bangle_type', 'bangle');
        }

        if (currentRingPath) {
            formData.append('ring_path', currentRingPath);
            formData.append('ring_type', 'ring');
        }

        console.log('Appended multi-jewel data:', Array.from(formData.entries()));
    }

    async function appendSingleJewelData(formData) {
        formData.append('design', currentDesign);
        const jewelryPath = getCurrentJewelryPath();

        if (jewelryPath && currentDesign) {
            if (jewelryPath.startsWith('http')) {
                // Fetch jewelry image from the provided URL
                const jewelryBlob = await fetch(jewelryPath).then(r => r.blob());
                formData.append('jewelry_image', jewelryBlob, 'jewelry.png');
                formData.append('jewelry_url', jewelryPath);
            } else {
                formData.append('jewelry_path', jewelryPath);
            }

            formData.append('is_large', currentIsLarge ? 'true' : 'false');

            let jewelryType = '';
            switch (currentDesign) {
                case 'NecklaceDesign':
                    jewelryType = 'necklace';
                    break;
                case 'EarringDesign':
                    jewelryType = 'earring';
                    break;
                case 'BangleDesign':
                    jewelryType = 'bangle';
                    break;
                case 'RingDesign':
                    jewelryType = 'ring';
                    break;
            }

            if (jewelryType) {
                formData.append('jewelry_type', jewelryType);
                console.log('Appended single-jewel data:', Array.from(formData.entries()));
            }
        }
    }
    processFrame();
}



function handleJewelryUrlInput(inputElement) {
    const url = inputElement.value.trim();
    if (url) {
        switch (currentDesign) {
            case 'NecklaceDesign':
                currentNecklacePath = url;
                console.log(`Set Necklace URL to: ${url}`);
                break;
            case 'EarringDesign':
                currentEarringPath = url;
                console.log(`Set Earring URL to: ${url}`);
                break;
            case 'BangleDesign':  // Ensure this case is properly handling the URL for bangle
                currentBanglePath = url;
                console.log(`Set Bangle URL to: ${url}`);
                break;
            case 'RingDesign':  // Ensure this case is properly handling the URL for ring
                currentRingPath = url;
                console.log(`Set Ring URL to: ${url}`);
                break;
            default:
                console.log("Unknown design type");
        }
    }
}


function getJewelryType(design) {
    const types = {
        'NecklaceDesign': 'necklace',
        'EarringDesign': 'earring',
        'BangleDesign': 'bangle',
        'RingDesign': 'ring'
    };
    return types[design] || '';
}

function getCurrentJewelryPath() {
    const urlInput = document.getElementById('jewelryUrlInput');

    // Debug log for current design
    console.log('Current design:', currentDesign);

    // If the user has provided a URL in the input field, return the URL value
    if (urlInput && urlInput.value.trim()) {
        const inputUrl = urlInput.value.trim();
        console.log('Using URL provided by user:', inputUrl);
        return inputUrl;
    }

    // Handle jewelry paths based on the current design type
    switch (currentDesign) {
        case 'NecklaceDesign':
            console.log('Selected design: Necklace');
            return currentNecklacePath || '/static/Image/Necklace/default_necklace.png';
        case 'EarringDesign':
            console.log('Selected design: Earring');
            return currentEarringPath || '/static/Image/Earring/default_earring.png';
        case 'BangleDesign':
            console.log('Selected design: Bangle');
            return currentBanglePath || '/static/Image/Bangle/default_bangle.png';
        case 'RingDesign':
            console.log('Selected design: Ring');
            return currentRingPath || '/static/Image/Ring/default_ring.png';
        default:
            // Fallback for unknown design types
            console.warn('Unknown design type! Falling back to default jewelry.');
            return '/static/Image/Default/default_jewelry.png';
    }
}



function showDesign(selectedDesign) {
    // Hide all jewelry containers initially
    const designContainers = [
        'necklaceContainer', 'earringContainer', 'bangleContainer', 'ringContainer', 
        'selectedNecklaceDisplay', 'selectedEarringDisplay', 'selectedBangleDisplay', 'selectedRingDisplay'
    ];

    designContainers.forEach(container => {
        const element = document.getElementById(container);
        if (element) {
            element.style.display = 'none';
        }
    });

    // Show the relevant design container
    if (document.getElementById(`${selectedDesign}Container`)) {
        document.getElementById(`${selectedDesign}Container`).style.display = 'flex';
    }
    if (document.getElementById(`selected${selectedDesign}Display`)) {
        document.getElementById(`selected${selectedDesign}Display`).style.display = 'block';
    }
}




function uploadAndProcess() {
    // Set the jewelry path input field value
    document.getElementById('jewelryPathInput').value = getCurrentJewelryPath();

    const fileInput = document.getElementById('fileInput');

    if (fileInput.files.length === 0) {
        alert('Please select a file to upload.');
        return false;
    }

    if (!getCurrentJewelryPath()) {
        alert('Please select a jewelry design.');
        return false;
    }

    const formData = new FormData(document.getElementById('uploadForm'));
    
    // Determine jewelry type based on the current design
    let jewelryType = '';
    if (currentDesign === 'NecklaceDesign') {
        jewelryType = 'necklace';
    } else if (currentDesign === 'EarringDesign') {
        jewelryType = 'earring';
    } else if (currentDesign === 'RingDesign') {
        jewelryType = 'ring';  // Add logic for ring design
    }

    // Append jewelry type to form data
    formData.append('jewelry_type', jewelryType);

    // Create a loading animation
    const loadingAnimation = document.createElement('div');
    loadingAnimation.className = 'loader';
    loadingAnimation.style.position = 'fixed';
    loadingAnimation.style.top = '50%';
    loadingAnimation.style.left = '50%';
    loadingAnimation.style.transform = 'translate(-50%, -50%)';
    loadingAnimation.style.zIndex = '9999';

    document.body.appendChild(loadingAnimation);

    function updateLoaderPosition() {
        const scrollY = window.scrollY || window.pageYOffset;
        loadingAnimation.style.top = `50%`;
    }

    window.addEventListener('scroll', updateLoaderPosition);

    fetch('/', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.blob();
    })
    .then(blob => {
        const imageUrl = URL.createObjectURL(blob);
        const processedImageContainer = document.getElementById('processedImageContainer');
        const processedImage = document.getElementById('processedImage');
        const downloadButton = document.getElementById('downloadButton');
       
        processedImage.onload = function() {
            processedImageContainer.classList.remove('hidden');
            processedImageContainer.style.display = 'block';
            processedImage.style.display = 'block';
            downloadButton.style.display = 'inline-block';
        };
       
        processedImage.src = imageUrl;

        // Remove loading animation and event listener
        document.body.removeChild(loadingAnimation);
        window.removeEventListener('scroll', updateLoaderPosition);
    })
    .catch(error => {
        console.error('Error:', error);
        document.body.removeChild(loadingAnimation);
        window.removeEventListener('scroll', updateLoaderPosition);
        alert('An error occurred while processing the image. Please try again.');
    });

    return false;
}




document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('uploadForm').style.display = 'none';

    document.getElementById('imageTryOnButton').addEventListener('click', function() {
    const form = document.getElementById('uploadForm');
    form.style.display = form.style.display === 'none' ? 'block' : 'none';
    });
});

function downloadProcessedImage() {
    const processedImage = document.getElementById('processedImage');
    if (processedImage.src) {
        const link = document.createElement('a');
        link.href = processedImage.src;
        link.download = 'processed_image.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}


// contact us
function openZohoForm() {
    window.open('https://vashions.in/contact-us/', '_blank');
}


// toggle button

function toggleJewelry(designType) {
    const sections = {
        'NecklaceDesign': document.getElementById('NecklaceDesign'),
        'EarringDesign': document.getElementById('EarringDesign'),
        // 'BangleDesign': document.getElementById('BangleDesign'),
        'RingDesign': document.getElementById('RingDesign')
    };

    const buttons = {
        'NecklaceDesign': document.getElementById('necklaceToggle'),
        'EarringDesign': document.getElementById('earringToggle'),
        // 'BangleDesign': document.getElementById('bangleToggle'),
        'RingDesign': document.getElementById('ringToggle')
    };

    const toggleContainer = document.querySelector('.toggle-container');

    // Reset all sections and buttons
    Object.keys(sections).forEach(sectionKey => {
        const section = sections[sectionKey];
        section.style.transition = 'opacity 0.5s cubic-bezier(0.4, 0, 0.2, 1), transform 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.display = 'none'; // Hide all sections initially
    });

    // Show the selected section
    sections[designType].style.display = 'block';

    // Add the active class to the toggle button
    Object.keys(buttons).forEach(buttonKey => {
        const button = buttons[buttonKey];
        button.classList.remove('active');
    });
    buttons[designType].classList.add('active');

    // Handle the active state on the container
    toggleContainer.classList.remove('necklace-active', 'earring-active', 'bangle-active', 'ring-active');
    toggleContainer.classList.add(`${designType.toLowerCase()}-active`);

    // Transition the section into view
    requestAnimationFrame(() => {
        sections[designType].style.opacity = '1';
        sections[designType].style.transform = 'translateY(0)';
    });
}

function toggleTryOnMode(mode) {
    const videoTryOn = document.getElementById('videoContainer');
    const imageTryOn = document.getElementById('imageTryOnContainer');
    const cameraButton = document.getElementById('cameraButton');
    const imageTryOnButton = document.getElementById('imageTryOnButton');

    if (mode === 'video') {
        videoTryOn.classList.remove('hidden');
        imageTryOn.classList.add('hidden');
        cameraButton.classList.add('active');
        imageTryOnButton.classList.remove('active');
    } else if (mode === 'image') {
        videoTryOn.classList.add('hidden');
        imageTryOn.classList.remove('hidden');
        cameraButton.classList.remove('active');
        imageTryOnButton.classList.add('active');
    }
}


// scroll to top button

// Back to Top Button
const backToTopButton = document.getElementById("backToTopBtn");

window.addEventListener('scroll', function() {
    if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 100) {
        backToTopButton.classList.add("show");
    } else {
        backToTopButton.classList.remove("show");
    }
});


backToTopButton.onclick = function() {
    scrollToTop();
};


document.addEventListener('DOMContentLoaded', function() {
    const cameraButton = document.getElementById('cameraButton');
    const zoomInButton = document.getElementById('zoomInButton');
    const takePhotoButton = document.getElementById('takephoto');
    const video = document.getElementById('video');
    const processedVideo = document.getElementById('processed-video');
    let isZoomed = false;
    let isCameraActive = false;

    // Hide zoom and take photo buttons initially
    zoomInButton.style.display = 'none';
    takePhotoButton.style.display = 'none';

    cameraButton.addEventListener('click', function() {
        isCameraActive = !isCameraActive;
        if (isCameraActive) {
            zoomInButton.style.display = 'block';
            takePhotoButton.style.display = 'block';
            // Your existing code to start the camera
        } else {
            zoomInButton.style.display = 'none';
            takePhotoButton.style.display = 'none';
            // Your existing code to stop the camera
            // Reset zoom when camera is deactivated
            video.style.transform = 'scale(-1, 1) translate(0, 0)';
            processedVideo.style.transform = 'scale(-1, 1) translate(0, 0)';
            zoomInButton.textContent = 'Zoom In';
            isZoomed = false;
        }
    });

    zoomInButton.addEventListener('click', function() {
        if (!isZoomed) {
            video.style.transform = 'scale(-1.5, 1.5) translate(0, 0)';
            video.style.transformOrigin = 'center center';
            processedVideo.style.transform = 'scale(-1.5, 1.5) translate(0, 0)';
            processedVideo.style.transformOrigin = 'center center';
            zoomInButton.textContent = 'Zoom Out';
            isZoomed = true;
        } else {
            video.style.transform = 'scale(-1, 1) translate(0, 0)';
            processedVideo.style.transform = 'scale(-1, 1) translate(0, 0)';
            zoomInButton.textContent = 'Zoom In';
            isZoomed = false;
        }
    });

    // Ensure initial mirroring by flipping horizontally
    video.style.transform = 'scaleX(-1)';
    processedVideo.style.transform = 'scaleX(-1)';
});


//  mutlti jewel

let isMultiJewelEnabled = false; // Initially, Multi Jewel is disabled

// Function to toggle the Multi Jewel feature based on slider state
function toggleMultiJewel() {
    const multiJewelToggle = document.getElementById('multiJewelToggle');
    isMultiJewelEnabled = multiJewelToggle.checked;

    if (isMultiJewelEnabled) {
        enableMultiJewelFeature();
        
        // Initialize video processing immediately
        if (stream && stream.active) {
            const video = document.getElementById('video');
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // Force immediate frame processing
            const formData = new FormData();
            formData.append('multi_jewel_enabled', 'true');
            formData.append('design', currentDesign);
            
            // Add current jewelry paths if they exist with their types
            if (currentNecklacePath) {
                formData.append('necklace_path', currentNecklacePath);
                formData.append('necklace_type', 'necklace');
            }
            if (currentEarringPath) {
                formData.append('earring_path', currentEarringPath);
                formData.append('earring_type', 'earring');
            }
            if (currentBanglePath) {
                formData.append('bangle_path', currentBanglePath);
                formData.append('bangle_type', 'bangle');
            }
            if (currentRingPath) {
                formData.append('ring_path', currentRingPath);
                formData.append('ring_type', 'ring');
            }
            
            // Capture and process current frame
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob((blob) => {
                formData.append('frame', blob);
                sendFramesToServer();
            }, 'image/jpeg');
            
            // Schedule another processing after a short delay
            setTimeout(() => {
                sendFramesToServer();
            }, 100);
        }
    } else {
        disableMultiJewelFeature();
    }

    // Update UI and restart frame processing if stream is active
    updateMultiJewelUI();
    if (stream && stream.active) {
        sendFramesToServer();
    }

    // Notify server about multi-jewel state change with current jewelry data
    fetch('/toggle_multi_jewel', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            enabled: isMultiJewelEnabled,
            necklace_path: currentNecklacePath,
            earring_path: currentEarringPath,
            bangle_path: currentBanglePath,
            ring_path: currentRingPath,
            current_design: currentDesign
        })
    });
}




  function getCurrentActiveJewelry() {
    return {
        necklace: currentNecklacePath,
        earring: currentEarringPath,
        bangle: currentBanglePath,
        ring: currentRingPath
    };
}

function clearJewelrySelections() {
    currentNecklacePath = "";
    currentEarringPath = "";
    currentBanglePath = "";
    currentRingPath = "";
    updateMultiJewelUI();
}

function updateMultiJewelUI() {
    const displays = {
        necklace: document.getElementById('selectedNecklaceDisplay'),
        earring: document.getElementById('selectedEarringDisplay'),
        bangle: document.getElementById('selectedBangleDisplay'),
        ring: document.getElementById('selectedRingDisplay')
    };

    Object.entries(displays).forEach(([type, display]) => {
        if (display) {
            display.style.display = isMultiJewelEnabled || currentDesign.toLowerCase().includes(type) ? 'block' : 'none';
        }
    });
}



// Function to enable multi-jewel feature (example)
function enableMultiJewelFeature() {
  // Code to allow multiple jewels selection or change UI state
  console.log("Multi Jewel Feature Enabled");
}

// Function to disable multi-jewel feature (example)
function disableMultiJewelFeature() {
  // Code to disable the multi-jewel feature or revert UI changes
  console.log("Multi Jewel Feature Disabled");
}


function updateMultiJewelUI() {
    const displays = {
        necklace: document.getElementById('selectedNecklaceDisplay'),
        earring: document.getElementById('selectedEarringDisplay'),
        bangle: document.getElementById('selectedBangleDisplay'),
        ring: document.getElementById('selectedRingDisplay')
    };

    Object.entries(displays).forEach(([type, display]) => {
        if (display) {
            const jewelryPath = getCurrentJewelryPath();
            if (isMultiJewelEnabled) {
                // Only show if the specific jewelry is selected
                display.style.display = eval(`current${type.charAt(0).toUpperCase() + type.slice(1)}Path`) ? 'block' : 'none';
            } else {
                // Show only the current design in single jewelry mode
                display.style.display = currentDesign.toLowerCase().includes(type) ? 'block' : 'none';
            }
        }
    });
}


function unselectJewelry(type) {
    // Clear the respective jewelry path
    switch(type) {
        case 'necklace':
            currentNecklacePath = "";
            document.getElementById('selectedNecklaceImage').src = "";
            document.getElementById('selectedNecklaceDisplay').style.display = 'none';
            break;
        case 'earring':
            currentEarringPath = "";
            document.getElementById('selectedEarringImage').src = "";
            document.getElementById('selectedEarringDisplay').style.display = 'none';
            break;
        case 'bangle':
            currentBanglePath = "";
            document.getElementById('selectedBangleImage').src = "";
            document.getElementById('selectedBangleDisplay').style.display = 'none';
            break;
        case 'ring':
            currentRingPath = "";
            document.getElementById('selectedRingImage').src = "";
            document.getElementById('selectedRingDisplay').style.display = 'none';
            break;
    }

    // If video stream is active, update the processing
    if (stream && stream.active) {
        sendFramesToServer();
    }
}


// // Add this function
// function logVideoDimensions() {
//     const video = document.getElementById('video');
//     const processedVideo = document.getElementById('processed-video');
    
//     console.log('Original Video:', {
//         width: video.offsetWidth,
//         height: video.offsetHeight,
//         ratio: video.offsetWidth/video.offsetHeight
//     });
    
//     console.log('Processed Video:', {
//         width: processedVideo.offsetWidth,
//         height: processedVideo.offsetHeight,
//         ratio: processedVideo.offsetWidth/processedVideo.offsetHeight
//     });
// }

document.getElementById('takephoto').addEventListener('click', function() {
    const processedVideo = document.getElementById('processed-video');
    const videoContainer = document.getElementById('videoContainer');
    
    // Add flash effect
    const flash = document.createElement('div');
    flash.style.position = 'absolute';
    flash.style.top = '0';
    flash.style.left = '0';
    flash.style.right = '0';
    flash.style.bottom = '0';
    flash.style.backgroundColor = 'white';
    flash.style.opacity = '0';
    flash.style.transition = 'opacity 0.1s ease-in-out';
    videoContainer.appendChild(flash);

    // Play animation
    flash.style.opacity = '0.7';
    
    setTimeout(() => {
        flash.style.opacity = '0';
        setTimeout(() => {
            flash.remove();
        }, 100);
    }, 100);

    // Capture and download flipped image
    const canvas = document.createElement('canvas');
    canvas.width = processedVideo.width;
    canvas.height = processedVideo.height;
    const ctx = canvas.getContext('2d');

    // Flip the image horizontally
    ctx.translate(canvas.width, 0); // Move to the right edge of the canvas
    ctx.scale(-1, 1); // Flip horizontally
    ctx.drawImage(processedVideo, 0, 0); // Draw the flipped image

    const link = document.createElement('a');
    link.download = 'vashions-tryon-' + Date.now() + '.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
});


