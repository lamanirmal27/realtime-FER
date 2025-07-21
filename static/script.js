const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const predictionText = document.getElementById("prediction");
const cameraToggle = document.getElementById("cameraToggle");
const ctx = canvas.getContext("2d");

let isStreamActive = false;
let predictionInterval = null;

async function toggleCamera() {
  if (!isStreamActive) {
    await setupCamera();
    cameraToggle.textContent = "Stop Camera";
    cameraToggle.classList.add("stop");
  } else {
    stopCamera();
    cameraToggle.textContent = "Start Camera";
    cameraToggle.classList.remove("stop");
  }
  isStreamActive = !isStreamActive;

}

function stopCamera() {
  const stream = video.srcObject;
  const tracks = stream?.getTracks();
  tracks?.forEach((track) => track.stop());
  video.srcObject = null;
  clearInterval(predictionInterval);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  predictionText.textContent = "turn on to start prediction";
  latestBbox = null;
}

// Global variables to store the latest prediction results
let latestPrediction = "None";
let latestBbox = null;

// Function to continuously draw the video and the latest bounding box
function drawLoop() {
  // Draw the current video frame onto the canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // If there is a bounding box from a recent prediction, draw it
  if (latestBbox) {
    const [x, y, w, h] = latestBbox;
    ctx.strokeStyle = "green";
    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, w, h);

    ctx.font = "24px Arial";
    const text = latestPrediction;
    const textWidth = ctx.measureText(text).width;
    ctx.fillStyle = "rgba(0, 255, 0, 0.7)";
    ctx.fillRect(x, y > 30 ? y - 30 : y + h, textWidth + 10, 30);
    ctx.fillStyle = "black";
    ctx.fillText(text, x + 5, y > 30 ? y - 10 : y + h + 22);
  }

  // Ask the browser to call this function again before the next repaint
  requestAnimationFrame(drawLoop);
}

// Function to send a frame to the server for prediction
async function sendFrame() {
  if (!isStreamActive || !video.srcObject) return;

  // We need a temporary canvas to get the image data
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = video.videoWidth;
  tempCanvas.height = video.videoHeight;
  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
  const imageData = tempCanvas.toDataURL("image/png");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageData }),
    });
    const result = await response.json();

    if (isStreamActive) {
      if (result.error) {
        predictionText.textContent = `Prediction: ${result.error}`;
        latestBbox = null;
      } else {
        predictionText.textContent = `Prediction: ${result.prediction}`;
        latestPrediction = result.prediction;
        latestBbox = result.bbox;
      }
    }
  } catch (error) {
    if (isStreamActive) {
      predictionText.textContent = `Prediction: Error - ${error.message}`;
      latestBbox = null;
    }
  }
}

// Main function to set up the camera
async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
    });
    video.srcObject = stream;

    // Use the 'onplaying' event as it's a more reliable trigger
    video.onplaying = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Start the rendering loop
      requestAnimationFrame(drawLoop);

      // Start the prediction loop
      predictionInterval = setInterval(sendFrame, 500);
    };

    // Programmatically play the video to overcome browser autoplay restrictions
    await video.play();
  } catch (err) {
    console.error("Error setting up camera:", err);
    predictionText.textContent =
      "Error: Could not start camera. Please grant permission and refresh.";
    cameraToggle.textContent = "Start Camera";
  }
}

// setupCamera();
cameraToggle.addEventListener("click", toggleCamera);
