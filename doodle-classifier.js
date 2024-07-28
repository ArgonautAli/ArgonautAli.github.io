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

function startDrawing(e) {
  isDrawing = true;
  draw(e);
}

function draw(e) {
  if (!isDrawing) return;
  const rect = drawCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
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
  const model = await tf.loadLayersModel(modelUrl + "model.json");
  model.summary();
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
  // tempCtx.putImageData(imageData, 0, 0);

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

  console.log("Tensor Input Data:", inputData);

  // Return tensor
  return tf.tensor4d(inputData, [1, 28, 28, 1]);
}

async function predict() {
  const model = await tf.loadLayersModel(modelUrl + "model.json");
  const inputTensor = processImage();
  const prediction = model.predict(inputTensor);
  const predictionArray = await prediction.array();

  inputTensor.print();

  const topPredictions = predictionArray[0]
    .map((prob, index) => ({ label: classLabels[index], probability: prob }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 5);
  let resultText = "Top prediction:\n";
  let confText = "";
  console.log("topPredictions", topPredictions);
  resultText = `Prediction: ${topPredictions[0].label}`;
  confText = `Confidence: ${topPredictions[0].probability * 100}`;

  resultP.textContent = resultText;
  confP.textContent = confText;
}

runModel();

predictBtn.addEventListener("click", predict);
clearBtn.addEventListener("click", clearCanvas);
