const modelUrl =
  "https://doodle-classifier-mini.s3.ap-northeast-1.amazonaws.com/model/";

const classLabels = ["ambulance", "cat", "dog", "house", "tree"];

const drawCanvas = document.getElementById("drawCanvas");

const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const resultP = document.getElementById("result");
const confP = document.getElementById("conf");

const drawCtx = drawCanvas.getContext("2d");

let isDrawing = false;

drawCtx.strokeStyle = "black";
drawCtx.lineWidth = 10;
drawCtx.lineCap = "round";
drawCtx.lineJoin = "round";

drawCanvas.addEventListener("mousedown", startDrawing);
drawCanvas.addEventListener("mousemove", draw);
drawCanvas.addEventListener("mouseup", stopDrawing);
drawCanvas.addEventListener("mouseout", stopDrawing);

drawCanvas.addEventListener("touchstart", startDrawing);
drawCanvas.addEventListener("touchmove", draw);
drawCanvas.addEventListener("touchend", stopDrawing);
drawCanvas.addEventListener("touchcancel", stopDrawing);

function startDrawing(e) {
  isDrawing = true;
  draw(e);
}

function draw(e) {
  if (!isDrawing) return;

  e.preventDefault(); // Prevent scrolling on touch devices

  const rect = drawCanvas.getBoundingClientRect();
  let x, y;

  if (e.touches) {
    x = e.touches[0].clientX - rect.left;
    y = e.touches[0].clientY - rect.top;
  } else {
    x = e.clientX - rect.left;
    y = e.clientY - rect.top;
  }

  drawCtx.lineTo(x, y);
  drawCtx.stroke();
  drawCtx.beginPath();
  drawCtx.moveTo(x, y);
}

function stopDrawing() {
  isDrawing = false;
  drawCtx.beginPath();
}

function clearCanvas() {
  drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  resultP.textContent = "";
  confP.textContent = "";
}

async function runModel() {
  document.getElementById("model_status").innerText = "Loading Model...";

  const model = await tf.loadLayersModel(modelUrl + "model.json");
  model.summary();
  document.getElementById("model_status").innerText = "Model ready!";
}

function processImage() {
  const imageData = drawCtx.getImageData(
    0,
    0,
    drawCanvas.width,
    drawCanvas.height
  );

  // Invert drawing
  for (let i = 0; i < imageData.data.length; i += 4) {
    imageData.data[i] = 255 - imageData.data[i]; // Red
    imageData.data[i + 1] = 255 - imageData.data[i + 1]; // Green
    imageData.data[i + 2] = 255 - imageData.data[i + 2]; // Blue
  }

  // rescale the image in temp cvs
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = drawCanvas.width;
  tempCanvas.height = drawCanvas.height;
  const tempCtx = tempCanvas.getContext("2d");

  // Draw the inverted image on the temporary canvas
  tempCtx.putImageData(imageData, 0, 0);

  // Create a second temporary canvas for the resized image
  const resizedCanvas = document.createElement("canvas");
  resizedCanvas.width = 28;
  resizedCanvas.height = 28;
  const resizedCtx = resizedCanvas.getContext("2d");

  // Draw the inverted image onto the resized canvas
  resizedCtx.drawImage(tempCanvas, 0, 0, 28, 28);

  // Get the rescaled image data
  const resizedImageData = resizedCtx.getImageData(0, 0, 28, 28);

  // Convert to tensor
  const inputData = new Float32Array(28 * 28);
  for (let i = 0; i < resizedImageData.data.length; i += 4) {
    const r = resizedImageData.data[i];
    const g = resizedImageData.data[i + 1];
    const b = resizedImageData.data[i + 2];
    const grayscale = (r + g + b) / 3;
    inputData[i / 4] = grayscale / 255.0; // Normalize to [0, 1]
  }

  // Return tensor
  return tf.tensor4d(inputData, [1, 28, 28, 1]);
}

async function predict() {
  document.getElementById("loader").style.display = "block";
  const model = await tf.loadLayersModel(modelUrl + "model.json");
  const inputTensor = processImage();
  const prediction = model.predict(inputTensor);
  const predictionArray = await prediction.array();

  inputTensor.print();

  console.log("predictionArray", predictionArray);

  const topPredictions = predictionArray[0]
    .map((prob, index) => ({ label: classLabels[index], probability: prob }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 5);

  let resultText = `Label: ${topPredictions[0].label}`;
  let confText = `Confidence: ${(topPredictions[0].probability * 100).toFixed(
    2
  )}%`;

  setTimeout(function () {
    document.getElementById("loader").style.display = "none";
    document.getElementById("result").style.display = "block";
    document.getElementById("conf").style.display = "block";

    resultP.textContent = resultText;
    confP.textContent = confText;
  }, 500);

  // topPredictions.forEach((pred) => {
  //   resultText += `${pred.label}: ${(pred.probability * 100).toFixed(2)}%\n`;
  // });
}

runModel();

predictBtn.addEventListener("click", predict);
clearBtn.addEventListener("click", clearCanvas);
