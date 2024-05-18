let webCamera = null;
let model = null;
let letterBuffer = [];
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
const btnLoad = document.getElementById('btnLoad');
const btnSave = document.getElementById('btnSave');
const btnTrain = document.getElementById('btnTrain');
let modelTraining = false;

const imageWidth = 200; // Зменшуємо розмір зображення для прискорення обробки
const imageHeight = 200;

// Класи літер алфавіту
const classes = { 1: "А", 2: "Б", 3: "В", 4: "Г", 5: "Ґ", 6: "Д", 7: "Е", 8: "Є", 9: "Ж", 10: "З", 11: "И", 12: "І", 13: "Ї", 14: "Й", 15: "К", 16: "Л", 17: "М", 18: "Н", 19: "О", 20: "П", 21: "Р", 22: "С", 23: "Т", 24: "У", 25: "Ф", 26: "Х", 27: "Ц", 28: "Ч", 29: "Ш", 30: "Щ", 31: "Ь", 32: "Ю", 33: "Я" };

// Змінна для зберігання моделі Handpose
let handposeModel = null;

async function loadHandposeModel() {
  handposeModel = await handpose.load();
  console.log("Handpose model loaded");
}

function drawHand(handPredictions, ctx) {
  // Очистити канвас
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  // Пройтися по кожному прогнозу руки
  handPredictions.forEach(hand => {
    const { landmarks, annotations } = hand;

    // Намалювати ключові точки руки
    drawKeypoints(landmarks, ctx);

    // Намалювати пальці руки
    drawFingers(annotations, ctx);
  });
}

// Функція для малювання ключових точок руки
function drawKeypoints(landmarks, ctx) {
  landmarks.forEach(([x, y, z]) => {
    // Малюємо круглу точку
    drawCircle(ctx, x, y, 5, "blue");
  });
}

// Функція для малювання пальців руки
function drawFingers(annotations, ctx) {
  for (const finger in annotations) {
    const points = annotations[finger];
    // Починаємо новий шлях
    ctx.beginPath();
    // Перша точка визначає початок лінії
    ctx.moveTo(points[0][0], points[0][1]);
    // З'єднуємо всі точки
    for (let j = 1; j < points.length; j++) {
      ctx.lineTo(points[j][0], points[j][1]);
    }
    // Налаштування властивостей лінії
    ctx.strokeStyle = "yellow";
    ctx.lineWidth = 2;
    // Малюємо лінію
    ctx.stroke();
  }
}

// Функція для малювання круглої точки
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

const addImgExample = async (classId) => {
  if (modelTraining) {
    const img = await captureImage();
    const activation = model.infer(img, "conv_preds");
    classifier.addExample(activation, classId);
    img.dispose();
  } else {
    swal("Помилка", "Модель не перебуває в режимі тренування! Натисніть кнопку 'Тренувати', щоб увімкнути режим тренування.", "error");
  }
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
  document.getElementById("result2").textContent = `Слово: ${word}`;
}

const app = async () => {
  try {
    webCamera = await tf.data.webcam(webcamElement);
    model = await mobilenet.load();
    await loadHandposeModel();

    const interval = setInterval(async () => {
      if (model !== null && classifier.getNumClasses() > 0 && handposeModel !== null) {
        const handPredictions = await handposeModel.estimateHands(webcamElement);
        if (handPredictions.length > 0) {
          const canvas = document.getElementById('canvas');
          const ctx = canvas.getContext('2d');
          drawHand(handPredictions, ctx);
          const result = await classifyImage();
          updateUI(result);
        }
      }
    }, 2000);

    const btnClear = document.getElementById("btnClear");
    btnClear.addEventListener("click", () => {
      clearWord();
    });

    window.addEventListener("beforeunload", () => {
      clearInterval(interval);
    });
  } catch (error) {
    console.error("Помилка при завантаженні моделі:", error);
  }
};

function updateUI(result) {
  const letter = classes[result.label];
  const confidence = result.confidences[result.label];

  document.getElementById("result").innerHTML = `
    <p><b>Літера:</b> ${letter}</p>
    <p><b>Ймовірність:</b> ${confidence.toFixed(2)}</p>
  `;

  letterBuffer.push(letter);
  processWord();
}

const alphabetButtons = document.getElementsByClassName('alpha-btn');
for (let i = 0; i < alphabetButtons.length; i++) {
  alphabetButtons[i].addEventListener("click", () => {
    const classId = alphabetButtons[i].getAttribute("data-position");
    addImgExample(classId);
  });
}

btnLoad.addEventListener("click", async () => {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".json";

  input.onchange = async () => {
    const file = input.files[0];
    if (file) {
      try {
        const jsonContent = await file.text();
        const loadedDataset = JSON.parse(jsonContent);
        const tensorObj = Object.entries(loadedDataset).reduce(
          (obj, [classId, data]) => {
            obj[classId] = tf.tensor(data.data, data.shape, data.dtype);
            return obj;
          },
          {}
        );

        classifier.setClassifierDataset(tensorObj);
      } catch (error) {
        console.error(
          "Помилка при завантаженні моделі з файлу JSON:",
          error
        );
      }
      swal("Відмінно!", "Модель успішно завантажено!", "success", { buttons: false, timer: 2000, });
    }
  };

  input.click();
});

btnTrain.addEventListener("click", () => {
  if (!model) {
    swal("Помилка", "Модель не завантажено, будь ласка, зачекайте кілька секунд", "error");
    return;
  }

  modelTraining = !modelTraining;
  btnTrain.innerText = "Тренуємо...";
  btnTrain.disabled = true;
  btnSave.disabled = false;
  swal("Відмінно!", "Початок тренування!", "success", {
    buttons: false,
    timer: 3000,
  });

  document.getElementById("alphabet-container").classList.toggle("alphabet-btn-visible", modelTraining);

});

btnSave.addEventListener("click", async () => {
  if (modelTraining) {
    const dataset = classifier.getClassifierDataset();
    const adjustedDataset = Object.entries(dataset).reduce(
      (obj, [classId, data]) => {
        obj[classId] = {
          data: Array.from(data.dataSync()),
          shape: data.shape,
          dtype: data.dtype,
        };
        return obj;
      },
      {}
    );

    const jsonDataset = JSON.stringify(adjustedDataset);

    const blob = new Blob([jsonDataset], { type: "application/json" });

    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "model_trained.json";

    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    swal("Відмінно!", "Модель успішно збережено!", "success");
  } else {
    swal("Увага", "Немає жодної навченої моделі. Спочатку навчіть одну, а потім збережіть її.", "error");
  }

  modelTraining = false;
  btnTrain.innerText = "Тренувати";
  btnTrain.disabled = false;
  btnSave.disabled = true;
});


// Функція для показу інструкцій
function showInstructions() {
  // Створення модального вікна
  const modal = document.createElement('div');
  modal.classList.add('modal');

  // Вміст інструкцій
  const instructionsContent = `
        <h2>Інструкції з використання</h2>
        <p>1. Натисніть кнопку "Тренувати", щоб розпочати тренування нової моделі.</p>
        <p>2. Коли тренування розпочалося, робіть жести з буквами алфавіту та натискайте відповідні кнопки для навчання моделі.</p>
        <p>3. Результати передбачення можна побачити справа від екрана.</p>
        <p>4. Для правильної роботи веб-додатка необхідний доступ до вашої веб-камери.</p>
        <p>5. Після успішного передбачення, збережіть модель.</p>
        <p>6. Якщо у вас вже є модель, ви можете завантажити її за допомогою кнопки "Завантажити модель".</p>
        <p>7. У разі помилки, перезавантажте сторінку або перевірте підключення до Інтернету :)</p>
        <p>8. Насолоджуйтесь Signa Lens! :)</p>
    `;

  modal.innerHTML = instructionsContent;

  // Додавання модального вікна на сторінку
  document.body.appendChild(modal);

  // Додавання кнопки для закриття модального вікна
  const closeButton = document.createElement('button');
  closeButton.innerText = 'Закрити';
  closeButton.onclick = function () {
    document.body.removeChild(modal);
  };

  modal.appendChild(closeButton);
}

const buttons = document.getElementsByClassName('alpha-btn');
for (let i = 0; i < buttons.length; i++) {
  buttons[i].addEventListener("click", () => {
    const classId = buttons[i].getAttribute("data-position");
    addImgExample(classId);
  });
}

window.onload = () => {
  app(); // Викликати функцію app() після завантаження сторінки
};
