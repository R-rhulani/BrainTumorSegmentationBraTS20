const express = require('express');
const multer = require('multer');
const sharp = require('sharp');
const tf = require('@tensorflow/tfjs-node');
const { readFile } = require('fs').promises;

const app = express();
const port = 4200;

const upload = multer({ dest: 'uploads/' });

app.post('/predict', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image file provided' });
        }

        // Convert the uploaded image to the desired format using sharp
        await sharp(req.file.path)
            .resize(128, 128)
            .toFile('uploads/lumpy_image.jpeg');

        const model = await tf.loadLayersModel('file://models/model.json');

        async function loadModelAndPredict() {
            // Read and preprocess the image
            const imageBuffer = await readFile('uploads/lumpy_image.jpeg');
            const image = tf.node.decodeImage(imageBuffer);
            const normalizedImage = image.div(255); // Normalize pixel values

            // Reshape and expand dimensions to match the model's input shape
            const reshapedImage = normalizedImage
                .resizeNearestNeighbor([128, 128])
                .reshape([1, 128, 128, 3, 4]);

            // Predict using the preprocessed image
            const predictions = model.predict(reshapedImage);
            const predictionData = predictions.arraySync()[0];

            return predictionData;
        }

        const prediction = await loadModelAndPredict();

        res.json({ prediction });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
