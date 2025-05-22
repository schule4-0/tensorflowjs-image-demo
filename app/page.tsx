"use client";

import * as tf from "@tensorflow/tfjs";
import { useState } from "react";
import NextImage from "next/image";

export default function Home() {
  const [classNames, setClassNames] = useState([]);

  let model: tf.GraphModel;
  tf.loadGraphModel(
    "mobilenet-v3-tfjs-large-075-224-classification-v1/model.json"
  ).then((result) => {
    model = result;
  });

  const handleAnalyzeClick = async () => {
    const fileInput = document.getElementById("image-upload");
    const imageFile = fileInput.files[0];

    if (!imageFile) {
      alert("Please upload an image file.");
      return;
    }

    const canvas = document.getElementById("canvas");

    try {
      const imageTensor = tf.browser.fromPixels(canvas);
      const logits = model.predict(preprocess(imageTensor));
      // const classIndex = await tf.argMax(tf.squeeze(logits)).data();

      // extract propability
      // using softmax for converting output tensor values to probabilities
      const predictions = await getTopKPredictions(tf.softmax(logits), 10);
      setClassNames(
        predictions.map((value) => ({
          label: model.metadata["classNames"][value.index],
          probability: value.probability,
        }))
      );
    } catch (error) {
      console.error("Error analyzing the image:", error);
    }
  };

  const fileChangeHandler = async (e) => {
    if (e.target.files[0]) {
      fileToPixel(e.target.files[0]);
    }
  };

  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start">
        <div className="flex gap-2">
          <NextImage
            src="file.svg"
            width={20}
            height={20}
            alt="File Icon"
          ></NextImage>
          <input
            className="block font-bold py-2 px-4 rounded bg-gray-100"
            width={10}
            type="file"
            id="image-upload"
            onChange={fileChangeHandler}
          />
        </div>
        <button
          onClick={handleAnalyzeClick}
          className="font-bold py-2 px-4 rounded bg-green-500 text-white"
        >
          Analyze Image
        </button>
        <p>Result:</p>
        <ol>
          {classNames.map((value, index) => (
            <li key={index}>
              {value.label} ({value.probability}%)
            </li>
          ))}
        </ol>
        <canvas id="canvas"></canvas>
      </main>
    </div>
  );
}

function preprocess(imageTensor: any) {
  const widthToHeight = imageTensor.shape[1] / imageTensor.shape[0];
  let squareCrop;
  if (widthToHeight > 1) {
    const heightToWidth = imageTensor.shape[0] / imageTensor.shape[1];
    const cropTop = (1 - heightToWidth) / 2;
    const cropBottom = 1 - cropTop;
    squareCrop = [[cropTop, 0, cropBottom, 1]];
  } else {
    const cropLeft = (1 - widthToHeight) / 2;
    const cropRight = 1 - cropLeft;
    squareCrop = [[0, cropLeft, 1, cropRight]];
  }
  // Expand image input dimensions to add a batch dimension of size 1.
  const crop = tf.image.cropAndResize(
    tf.expandDims(imageTensor),
    squareCrop,
    [0],
    [224, 224]
  );
  return crop.div(255);
}

function fileToImage(file: File) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      const img = new Image();
      img.onload = () => resolve(img); // Image is fully loaded
      img.onerror = reject;
      img.src = reader.result; // Data URL from file
    };

    reader.onerror = reject;
    reader.readAsDataURL(file); // Convert file to base64 URL
  });
}

async function fileToPixel(imageFile: File) {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const image = await fileToImage(imageFile);
  canvas.width = image.width;
  canvas.height = image.height;
  ctx.drawImage(image, 0, 0);
}

async function getTopKPredictions(logits, k = 10) {
  const logitsData = await tf.squeeze(logits).data(); // flat typed array
  const logitsArray = Array.from(logitsData);

  // Get top-k indices with scores
  const topK = logitsArray
    .map((score, index) => ({ index, score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);

  // Optionally normalize to percentage (if logits are probabilities already)
  const total = logitsArray.reduce((sum, val) => sum + val, 0);
  const predictions = topK.map(({ index, score }) => ({
    index,
    probability: ((score / total) * 100).toFixed(2),
  }));

  return predictions;
}
