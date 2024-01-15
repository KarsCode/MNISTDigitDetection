
import  { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js'

const MnistVisualization = ({onDataUpdate}) => {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  // const [randomImage, setRandomImage] = useState<string | null>(null);
  // const [predictedClass, setPredictedClass] = useState<number | null>(null);
  // const [isPredictionCorrect, setIsPredictionCorrect] = useState<boolean | null>(null);

  useEffect(() => {
    const trainAndEvaluateModel = async () => {
      const INPUTS = TRAINING_DATA.inputs;
      const OUTPUTS = TRAINING_DATA.outputs;

      const INPUTS_TENSOR = tf.tensor2d(INPUTS);
      const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10);

      const myModel = tf.sequential();
      myModel.add(tf.layers.dense({ inputShape: [784], units: 32, activation: 'relu' }));
      myModel.add(tf.layers.dense({ units: 16, activation: 'relu' }));
      myModel.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

      myModel.summary();

      myModel.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
      });

      // Train the model asynchronously
      await myModel.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
        epochs: 50,
        validationSplit: 0.2,
        batchSize:512,
        shuffle: true,
        // callbacks: tfvis.show.fitCallbacks({ name: 'Training Performance' }, ['loss', 'val_loss', 'acc', 'val_acc']),
      });

      // Set the trained model to state
      setModel(myModel);

      onDataUpdate({
        randomImage: null,
        predictedClass: null,
        isPredictionCorrect: null,
        trainingComplete: false,
      });

      // Evaluate the model
      setInterval(() => {
        updateRandomImageAndPrediction(myModel, INPUTS);
      }, 3000);

    };

    const updateRandomImageAndPrediction = async (model: tf.LayersModel, inputs: number[][]) => {
      // Randomly select an input
      const randomIndex = Math.floor(Math.random() * inputs.length);
      const randomInput = inputs[randomIndex];

      // Expand dimensions and make predictions
      const inputTensor = tf.tensor2d(randomInput, [1, 784]);
      const predictions = model.predict(inputTensor) as tf.Tensor;

      // Find the index of the maximum value (argMax)
      const predicted = predictions.squeeze().argMax().dataSync()[0];

      // Determine if the prediction is correct
      const isCorrect = predicted === TRAINING_DATA.outputs[randomIndex];

      const imageArray = await tf.browser.toPixels(inputTensor);
      const originalWidth = 28;
      const originalHeight = 28;

      // Create a new ImageData with larger dimensions
      const scaledImageData = new ImageData(480, 480);

      // Scale the image data
      for (let y = 0; y < scaledImageData.height; y++) {
        for (let x = 0; x < scaledImageData.width; x++) {
          const originalX = Math.floor((x / scaledImageData.width) * originalWidth);
          const originalY = Math.floor((y / scaledImageData.height) * originalHeight);

          const originalIndex = (originalY * originalWidth + originalX) * 4;
          const scaledIndex = (y * scaledImageData.width + x) * 4;

          scaledImageData.data[scaledIndex] = imageArray[originalIndex];
          scaledImageData.data[scaledIndex + 1] = imageArray[originalIndex + 1];
          scaledImageData.data[scaledIndex + 2] = imageArray[originalIndex + 2];
          scaledImageData.data[scaledIndex + 3] = imageArray[originalIndex + 3];
        }
      }

// Create a canvas and put the scaled image data on it
      const canvas = document.createElement('canvas');
      canvas.width = 480;
      canvas.height = 480;
      const ctx = canvas.getContext('2d');
      ctx?.putImageData(scaledImageData, 0, 0);
      
      // Update state
      onDataUpdate({
        randomImage: canvas.toDataURL(),
        predictedClass: predicted,
        isPredictionCorrect: isCorrect,
        trainingComplete: true,
      });

      // Dispose of tensors to free up memory
      inputTensor.dispose();
      predictions.dispose();
    };


    trainAndEvaluateModel();
  }, [onDataUpdate]);


return null;
}
export default MnistVisualization;    
