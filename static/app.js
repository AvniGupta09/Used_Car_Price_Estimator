// Two small helpers for the form:
//   1. the Model list only shows models of the chosen brand
//   2. picking a model fills in its typical specifications

const carData = JSON.parse(document.getElementById("car-data").textContent);

const brandInput = document.getElementById("brand");
const modelInput = document.getElementById("model");
const modelList = document.getElementById("model-list");

function carKey() {
  return brandInput.value.trim().toLowerCase() + "|" + modelInput.value.trim().toLowerCase();
}

// remember the starting car, so reloading the page never overwrites what the user typed
let lastFilled = carKey();

function fillModelList() {
  const models = carData.models[brandInput.value.trim().toLowerCase()] || [];
  modelList.innerHTML = "";
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model;
    modelList.appendChild(option);
  }
}

const specFields = ["fuel_type", "transmission_type", "seats", "engine", "max_power", "mileage"];
const edited = new Set();
for (const name of specFields) {
  document.getElementById(name).addEventListener("input", () => edited.add(name));
}

function fillSpecs() {
  const key = carKey();
  const specs = carData.defaults[key];
  if (!specs || key === lastFilled) return;

  lastFilled = key;
  for (const [name, value] of Object.entries(specs)) {
    if (!edited.has(name)) {
      document.getElementById(name).value = value;
    }
  }
}

brandInput.addEventListener("input", () => { fillModelList(); fillSpecs(); });
modelInput.addEventListener("input", fillSpecs);
fillModelList();