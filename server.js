const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const axios = require('axios');
const cors = require('cors');

const app = express();
app.use(cors());
const PORT = process.env.PORT || 3000;

let model;

// Load TensorFlow.js Model
(async function loadModel() {
    try {
        model = await tf.loadLayersModel('file://stress_lstm_model_tfjs/model.json');
        console.log("✅ Model loaded successfully!");
    } catch (error) {
        console.error("❌ Error loading model:", error);
    }
})();

// Function to fetch sensor data from Blynk
async function fetchSensorData() {
    try {
        const response = await axios.get('https://blynk.cloud/external/api/get?token=YOUR_TOKEN&keys=v0,v1,v2,v3');
        return [
            parseFloat(response.data.v0),
            parseFloat(response.data.v1),
            parseFloat(response.data.v2),
            parseFloat(response.data.v3)
        ];
    } catch (error) {
        console.error("❌ Error fetching sensor data:", error);
        return null;
    }
}

// API Endpoint: Predict Stress Level
app.get('/predict', async (req, res) => {
    if (!model) return res.status(500).json({ error: "Model not loaded yet." });

    const sensorData = await fetchSensorData();
    if (!sensorData) return res.status(500).json({ error: "Failed to fetch sensor data." });

    // Reshape input for LSTM (1 sample, 10 timesteps, 4 features)
    const inputTensor = tf.tensor([sensorData]).reshape([1, 10, 4]);

    // Predict stress level
    const prediction = model.predict(inputTensor);
    const stressLevel = prediction.argMax(1).dataSync()[0];

    res.json({ bpm: sensorData[0], eeg: sensorData[1], gsr: sensorData[2], ecg: sensorData[3], stress_level: stressLevel });
});

app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
