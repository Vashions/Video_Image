<style>
    .uploadImg-sec {
      width: 50%;
      padding-left: 30px;
  }
    .try-now {
      max-width: 1620px;
      margin: 70px auto;
          padding: 0 20px;
  }
    .try-now h1 {
      text-align: center;
  }
    .p-try-now {
      text-align: left;
      padding: 19px 0 30px;
  }
    .p-try-now h5 {
      font-size: 20px;
          font-weight: 500;
  }
  ::file-selector-button {
      display: none;
  }
    .input-row {
  position: relative;
      width: 58%;
      height: 60px;
      border: 1px solid #cccc;
      border-radius: 3px;
          overflow: hidden;
  }
    .uploadImg-sec p {
      font-weight: bold;
  }
    .uploadImg-sec p~p {
      margin: 0 0 34px;
  }
  .input-row:after {
    content: attr(data-text);
    font-size: 18px;
    position: absolute;
    top: 0;
    left: 0;
    background: #fff;
    padding: 10px 15px;
    display: block;
    width: calc(100% - 40px);
    pointer-events: none;
    z-index: 20;
    height: 40px;
    line-height: 40px;
    color: #999;
    border-radius: 5px 10px 10px 5px;
    font-weight: 300;
  }
  .input-row:before {
    content: "Select";
    position: absolute;
    top: 0;
    right: 0;
    display: inline-block;
    height: 60px;
    background: #000;
    color: #fff;
    font-weight: 700;
    z-index: 999;
    font-size: 16px;
    line-height: 60px;
    padding: 0 15px;
    text-transform: uppercase;
    pointer-events: none;
    border-radius: 0 5px 5px 0;
    
  }
  .input-row:hover:before {
    background: #000;
  }
  .input-row input {
    opacity: 1;
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    left: 0;
    z-index: 99;
    height: 100%;
    margin: 0;
    padding: 10px 0px 5px 5px;
    display: block;
    cursor: pointer;
    width: 100%;
  }
  
  .sprucecss {
    align-items: flex-start;
    background-color: white;
    border-radius: 0.25rem;
    box-shadow: 0 0 0.5rem rgba(0, 0, 0, 0.05);
    color: #444;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    inset: auto auto 1rem 1rem;
    line-height: 1.5;
    max-width: 11rem;
    padding: 1.5rem;
    position: fixed;
    text-decoration: none;
  }
   .display-img {
      width: 450px;
      height: 300px;
  }
  .uploadImg-sec form {
      margin: 0 0 30px;
  }
  p.img-grn {
      color: green;
  }
  @media only screen and (max-width: 767px) {
    div#productInfo {
      display: block !important;
      margin: 0 auto;
  }
    div#productImg {
      width: 100%;
      margin: 0 0 20px;
  }
    .input-row{width: 100%;}
    div#productImg {
      width: 100%;
  }
  .uploadImg-sec {
      width: 100%;
      padding-left: 0;
  }
    .display-img {
      width: 100%;
      height: auto;
  }
  }
  #videoContainer {
    position: relative;
    width: 640px;
    height: 480px;
    margin: 0 auto;
  }
  
  #video-loader {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 10;
  }

  .action-button {
    background-color: #000;
    color: #fff;
    font-weight: 700;
    font-size: 16px;
    text-transform: uppercase;
    padding: 0 15px;
    height: 60px;
    border: none;
    border-radius: 5px 5px 5px 5px;
    cursor: pointer;
    }

    .action-button:hover {
    background: #333;
    }

    #uploadBtn, #live-try {
    display: inline-block;
    margin-right: 10px;
    }
    #uploadBtn {
    margin-right: 20px;
    }

    #videoContainer {
    position: relative;
    width: 450px;
    height: 300px;
    display: inline-block;
    vertical-align: top;
    margin-left: 20px;
    border-radius: 25px 25px 25px 25px;
    }



  
  
  </style>
  <div class="try-now">
    <h1>Try Now Page</h1>
    <div class="p-try-now">
      <h5>You can try the Jewellery on your image or video.</h5>
      <h5>Just upload your image or start the video to check the result.</h5>
    </div>
    <div id="productInfo">
      <div id="productImg">
        <img id="tryNowImg" src="" alt="My Image">

      </div>
  
      <div class="uploadImg-sec">
        <div id="select-img">
          <form method="post" enctype="multipart/form-data">
            <p>Please Upload your image or try video</p>
            <br>
            <label><b>Select the image</b></label>
            <div class="input-row"> <input type="file" name="user-img" id="user-img"></div>
            <br>
            <button type="button" id="uploadBtn" class="action-button" onclick="upload_image()">Upload</button>
            <button type="button" id="live-try" class="action-button" onclick="startVideoProcessing()">Try Video</button>

          </form>
        </div>
        <div id="img-loader" style="display:none">
          <img src="https://www.vashions.com/cdn/shop/files/Loading_icon_72cb4f5d-86ca-409d-8361-012b023f2869.gif" id="loading-img">
        </div>
        <div class="display-img">
          <p id="img-grn" class="img-grn" style="display:none">Image Uploaded</p>
          <div id="returnImg"></div>
        </div>
      </div>
    </div>
    <div id="videoContainer" style="display:none;">
      <video id="video" width="640" height="480" autoplay style="display:none;"></video>
      <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
      <img id="output" width="640" height="480">
    </div>
    
  </div>
  
  <script>
    setTimeout(function(){
  },1000);

  var currentURL = window.location.href;
  var url = new URL(currentURL);
  var card = url.searchParams.get("card");

  const myArray = card.split("/");
  myFunction();

  function myFunction(){
    var productName = (myArray[2]);
    fetch(window.Shopify.routes.root + 'products/'+productName+'.js')
      .then(response => response.json())
      .then(product => {
        var imgrsRC = product.images;
        console.log('product',product);
        console.log('imgrsRC',imgrsRC[0]);
        splitImageSrc = imgrsRC[0].split("/files");
        console.log('splitImageSrc[2]',splitImageSrc[2]);
        let jwImgSrc = "https://www.vashions.com/cdn/shop/files" + splitImageSrc[2];
        console.log('jwImgSrc',jwImgSrc);
        document.getElementById("tryNowImg").src = jwImgSrc;
        setTimeout(function(){
          document.getElementById("productInfo").style.display = "block";                                                                                      
        },1000);
      }
    );
  }

    // Existing code for image upload
async function upload_image() {
    let userImgVal = document.getElementById("user-img").value;
    let userImg = document.getElementById("user-img").files[0];

    if (userImgVal.length == 0) {
        alert('please select user-img');
    } else {
        document.getElementById('select-img').style.display = "none";
        document.getElementById('img-loader').style.display = "block";
        const imgElement = document.getElementById('tryNowImg');
        const userimage = imgElement.src;

        const apiUrl = getApiUrlForJewelType(userimage);
        const formData = new FormData();
        formData.append('customer_image', userImg);

        let jewelryName = userimage.split('/').pop();
        jewelryName = jewelryName.split('?')[0]; 

        await fetch(userimage)
          .then(response => response.blob())
          .then(blob => {
                formData.append('image2', blob, jewelryName);
            });

        fetch(apiUrl, {
            method: 'POST',
            body: formData,
        })
          .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.blob();
            })
          .then(data => {
                const imageURL = URL.createObjectURL(data);
                const imgElement = document.createElement('img');
                imgElement.src = imageURL;
                document.getElementById("returnImg").innerHTML = '';
                document.getElementById('img-loader').style.display = "none";
                document.getElementById('img-grn').style.display = "block";
                document.getElementById("returnImg").appendChild(imgElement);
                console.log('image generated');
            })
          .catch(error => {
                console.error('There was a problem with the fetch operation:', error);
            });
    }
}

function getApiUrlForJewelType(userimage) {
    let jewelryName = userimage.split('/').pop();
    jewelryName = jewelryName.split('?')[0]; 

    const jewelryType = jewelryName.toLowerCase().includes('tryon_necklace')? 'Necklace' :
        jewelryName.toLowerCase().includes('tryon_earring')? 'Earring' :
        jewelryName.toLowerCase().includes('tryon_ring')? 'Ring' : 'Unknown';

    let apiUrl;
    switch (jewelryType) {
        case 'Necklace':
            apiUrl = 'https://vashions.co.in/necklace_process_image';
            break;
        case 'Earring':
            apiUrl = 'https://vashions.co.in/earring_process_image';
            break;
        case 'Ring':
            apiUrl = 'https://vashions.co.in/ring_process_image';
            break;
        default:
            console.error('Unknown jewelry type:', jewelryType);
            apiUrl = '';
    }

    return apiUrl;
}
  
  // New code for video processing
  let stream;
  let isStreamActive = false;
  let currentDesign = "NecklaceDesign";
  let currentJewelryPath = tryNowImg;
  
  
async function startVideoProcessing() {
    const videoContainer = document.getElementById('videoContainer');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const output = document.getElementById('output');
    const tryNowImg = document.getElementById('tryNowImg');
    const imgLoader = document.getElementById('img-loader');

    videoContainer.style.display = 'block';
    document.getElementById('select-img').style.display = 'none';
    imgLoader.style.display = 'block';

    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        await video.play();
        isStreamActive = true;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        output.width = video.videoWidth;
        output.height = video.videoHeight;

        let totalProcessingTime = 0;
        let frameCount = 0;
        let isProcessing = false;

        // Get the jewelry name from tryNowImg src
        const jewelryUrl = tryNowImg.src;
        let jewelryName = jewelryUrl.split('/').pop();
        jewelryName = jewelryName.split('?')[0]; // This will remove the query parameter

        // Determine jewelry type based on the image name
        const jewelryType = jewelryName.toLowerCase().includes('tryon_necklace')? 'Necklace' :
            jewelryName.toLowerCase().includes('tryon_earring')? 'Earring' :
            jewelryName.toLowerCase().includes('tryon_ring')? 'Ring' : 'Unknown';

        async function processFrame() {
            if (!isStreamActive) return;
            if (isProcessing) {
                requestAnimationFrame(processFrame);
                return;
            }

            isProcessing = true;

            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            const frameBlob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.5));
            const base64Frame = await blobToBase64(frameBlob);

            const jsonData = {
                frame: base64Frame,
                jewelry_name: jewelryName,
                jewelry_type: jewelryType,
                client_name: 'Vashions'
            };
            console.log('Jewelry Name:', jewelryName);
            console.log('Jewelry Type:', jewelryType);

            const startTime = performance.now();

            try {
                const response = await fetch('https://vashions.co.in/process_realtime_client_jewelry', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(jsonData)
                });

                if (response.ok) {
                    const data = await response.json();
                    const endTime = performance.now();
                    const processingTime = endTime - startTime;

                    totalProcessingTime += processingTime;
                    frameCount++;

                    console.log(`Frame processing time: ${processingTime.toFixed(2)} ms`);
                    console.log(`Average processing time: ${(totalProcessingTime / frameCount).toFixed(2)} ms`);

                    const processedImage = new Image();
                    processedImage.onload = () => {
                        context.clearRect(0, 0, canvas.width, canvas.height);
                        context.drawImage(processedImage, 0, 0, canvas.width, canvas.height);
                        output.src = canvas.toDataURL('image/jpeg');
                        imgLoader.style.display = 'none';
                        isProcessing = false;
                        requestAnimationFrame(processFrame);
                    };
                    processedImage.src = 'data:image/jpeg;base64,' + data.image;
                } else {
                    console.error('Error processing frame:', await response.text());
                    imgLoader.style.display = 'none';
                    isProcessing = false;
                    requestAnimationFrame(processFrame);
                }
            } catch (error) {
                console.error('Fetch error:', error);
                imgLoader.style.display = 'none';
                isProcessing = false;
                requestAnimationFrame(processFrame);
            }
        }

        // Start multiple processing loops for parallel processing
        for (let i = 0; i < 3; i++) {
            requestAnimationFrame(processFrame);
        }
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Unable to access camera. Please check your permissions and try again.');
        imgLoader.style.display = 'none';
    }
}


function blobToBase64(blob) {
    return new Promise((resolve, _) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result.split(',')[1]);
        reader.readAsDataURL(blob);
    });
}


  
  async function compressImageData(imageData) {
      return new Promise((resolve) => {
          const canvas = document.createElement('canvas');
          canvas.width = imageData.width;
          canvas.height = imageData.height;
          const ctx = canvas.getContext('2d');
          ctx.putImageData(imageData, 0, 0);
  
          canvas.toBlob((blob) => {
              resolve(blob);
          }, 'image/jpeg', 0.7); // Adjust quality as needed
      });
  }
  
  function toggleCamera() {
      if (!isStreamActive) {
          startVideoProcessing();
      } else {
          if (stream) {
              stream.getTracks().forEach(track => track.stop());
              isStreamActive = false;
              document.getElementById('videoContainer').style.display = 'none';
          }
      }
  }
  
  
  document.addEventListener('DOMContentLoaded', function() {
      const cameraButton = document.getElementById('cameraButton');
      if (cameraButton) {
          cameraButton.addEventListener('click', toggleCamera);
      } else {
          console.error('Camera button not found');
      }
  });
  </script>
  
  
  