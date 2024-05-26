let webCamera = null;
let model = null;
let letterBuffer = [];
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();

const imageWidth = 200;
const imageHeight = 200;

const classes = { 1: "А", 2: "Б", 3: "В", 4: "Г", 5: "Ґ", 6: "Д", 7: "Е", 8: "Є", 9: "Ж", 10: "З", 11: "И", 12: "І", 13: "Ї", 14: "Й", 15: "К", 16: "Л", 17: "М", 18: "Н", 19: "О", 20: "П", 21: "Р", 22: "С", 23: "Т", 24: "У", 25: "Ф", 26: "Х", 27: "Ц", 28: "Ч", 29: "Ш", 30: "Щ", 31: "Ь", 32: "Ю", 33: "Я" };

let handposeModel = null;

async function loadHandposeModel() {
  try {
    handposeModel = await handpose.load();
    console.log("Handpose model loaded");
  } catch (error) {
    console.error("Помилка при завантаженні моделі Handpose:", error);
  }
}

function drawHand(handPredictions, ctx) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  handPredictions.forEach(hand => {
    const { landmarks, annotations } = hand;
    drawFingers(annotations, ctx);
    drawKeypoints(landmarks, ctx);
    drawBlueDots(annotations, ctx);
  });
}

function drawKeypoints(landmarks, ctx) {
  landmarks.forEach(([x, y, z]) => {
    drawCircle(ctx, x, y, 5, "blue");
  });
}

function drawFingers(annotations, ctx) {
  for (const finger in annotations) {
    const points = annotations[finger];
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let j = 1; j < points.length; j++) {
      ctx.lineTo(points[j][0], points[j][1]);
    }
    ctx.strokeStyle = "yellow";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

function drawBlueDots(annotations, ctx) {
  ctx.fillStyle = "blue"; // Встановлюємо колір заливки в синій
  for (const finger in annotations) {
    const points = annotations[finger];
    points.forEach(point => {
      ctx.beginPath();
      ctx.arc(point[0], point[1], 5, 0, Math.PI * 2); // Малюємо коло у відповідних координатах
      ctx.fill();
    });
  }
}

function drawCircle(ctx, x, y, radius, color) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

const clearWord = () => {
  letterBuffer = [];
  processWord();
};

const captureImage = async () => {
  const img = await tf.browser.fromPixels(webcamElement);
  const resizedImg = tf.image.resizeBilinear(img, [imageWidth, imageHeight]);
  return resizedImg;
};

async function classifyImage() {
  const img = await captureImage();
  const activation = model.infer(img, "conv_preds");
  const result = await classifier.predictClass(activation);
  img.dispose();
  return result;
}

function processWord() {
  const word = letterBuffer.join('');
  const resultElement = document.getElementById("result2");
  if (resultElement) {
    resultElement.textContent = `Слово: ${word}`;
  }
}

async function loadClassifierDataset(url) {
  try {
    const response = await fetch(url);
    const jsonContent = await response.json();
    const tensorObj = Object.entries(jsonContent).reduce(
      (obj, [classId, data]) => {
        obj[classId] = tf.tensor(data.data, data.shape, data.dtype);
        return obj;
      },
      {}
    );
    classifier.setClassifierDataset(tensorObj);
    console.log("Classifier dataset loaded");
  } catch (error) {
    console.error("Помилка при завантаженні класифікатора з файлу JSON:", error);
  }
}

const initializeWebcam = async () => {
  try {
    webCamera = await tf.data.webcam(webcamElement);
    console.log("Webcam initialized");
  } catch (error) {
    console.error("Помилка при ініціалізації веб-камери:", error);
  }
};

const initializeModels = async () => {
  try {
    model = await mobilenet.load();
    await loadHandposeModel();
    await loadClassifierDataset('/ModeloFinal.json');
  } catch (error) {
    console.error("Помилка при завантаженні моделей:", error);
  }
};

const startRecognition = () => {
  const interval = setInterval(async () => {
    if (model && classifier.getNumClasses() > 0 && handposeModel) {
      const handPredictions = await handposeModel.estimateHands(webcamElement);
      if (handPredictions.length > 0) {
        const canvas = document.getElementById('canvas');
        if (canvas) {
          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          drawHand(handPredictions, ctx); // Виклик drawHand замість окремих функцій

          const result = await classifyImage();
          updateUI(result);
        }
      }
    }
  }, 1000);

  window.addEventListener("beforeunload", () => {
    clearInterval(interval);
  });
};

function showInstructions() {
  const modal = document.createElement('div');
  modal.classList.add('modal');

  const instructionsContent = `
        <h2>Інструкції з Використання</h2>
        <p>1. Дайте доступ до вашої камери для коректної роботи веб-додатка.</p>
        <p>2. Після отримання доступу, веб-додаток автоматично розпочне розпізнавання жестів.</p>
        <p>3. Виконуйте жести однієї з літер алфавіту української жестової мови перед камерою.</p>
        <p>4. Ви можете побачити результати розпізнавання жестів у правій частині екрана.</p>
        <p>5. У разі виникнення помилок, перезавантажте сторінку або перевірте своє інтернет-з'єднання.</p>
        <p>6. Розумійте з Rozymiy! :)</p>
    `;

  modal.innerHTML = instructionsContent;

  document.body.appendChild(modal);
  const closeButton = document.createElement('button');
  closeButton.innerText = 'Закрити';
  closeButton.onclick = function () {
    document.body.removeChild(modal);
  };

  modal.appendChild(closeButton);
}

const app = async () => {
  await initializeWebcam();
  await initializeModels();
  startRecognition();

  const btnClear = document.getElementById("btnClear");
  if (btnClear) {
    btnClear.addEventListener("click", () => {
      clearWord();
    });
  }
};

function updateUI(result) {
  const letter = classes[result.label];
  const confidence = result.confidences[result.label];

  const resultElement = document.getElementById("result");
  if (resultElement) {
    resultElement.innerHTML = `
      <p><b>Літера:</b> ${letter}</p>
      <p><b>Ймовірність:</b> ${confidence}</p>
    `;
  }

  letterBuffer.push(letter);
  processWord();
}

window.onload = () => {
  app();
};
