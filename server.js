const tf = require('@tensorflow/tfjs-node');
const axios = require('axios');
const express = require('express');

const app = express();
const PORT = 3000;

let model;

// Load the TensorFlow.js model
(async function loadModel() {
    model = await tf.loadLayersModel('file://stress_lstm_model_tfjs/model.json');
    console.log("Model loaded successfully!");
})();

// Function to fetch sensor data from Blynk
async function fetchSensorData() {
    try {
        const bpmResponse = await axios.get('https://blr1.blynk.cloud/external/api/get?token=YOUR_TOKEN&v0=');
        const eegResponse = await axios.get('https://blr1.blynk.cloud/external/api/get?token=YOUR_TOKEN&v1=');
        const gsrResponse = await axios.get('https://blr1.blynk.cloud/external/api/get?token=YOUR_TOKEN&v2=');
        const ecgResponse = await axios.get('https://blr1.blynk.cloud/external/api/get?token=YOUR_TOKEN&v3=');

        return [
            parseFloat(bpmResponse.data),
            parseFloat(eegResponse.data),
            parseFloat(gsrResponse.data),
            parseFloat(ecgResponse.data)
        ];
    } catch (error) {
        console.error("Error fetching sensor data:", error);
        return null;
    }
}

// Endpoint for stress prediction
app.get('/predict', async (req, res) => {
    if (!model) {
        return res.status(500).json({ error: "Model not loaded yet." });
    }

    const sensorData = await fetchSensorData();
    if (!sensorData) {
        return res.status(500).json({ error: "Failed to fetch sensor data." });
    }

    // Normalize input (optional)
    const inputTensor = tf.tensor([sensorData]).reshape([1, 1, 4]);

    // Predict stress level
    const prediction = model.predict(inputTensor);
    const stressLevel = prediction.argMax(1).dataSync()[0];

    res.json({
        bpm: sensorData[0],
        eeg: sensorData[1],
        gsr: sensorData[2],
        ecg: sensorData[3],
        stress_level: stressLevel
    });
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
