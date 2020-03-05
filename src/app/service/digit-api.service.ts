import { Injectable } from "@angular/core";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { DataApiService } from "./data-api.service";

@Injectable({
  providedIn: "root"
})
export class DigitApiService {
  constructor() {}
  classNames: any[] = [
    "Zero",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine"
  ];

  private async showExamples(data) {
    const surface = tfvis
      .visor()
      .surface({ name: "Input Data Example", tab: "Input Data" });

    const examples = data.nextTestBatch(2000);
    const numExamples = examples.xs.shape[0];

    for (let i = 0; i < numExamples; i++) {
      const imageTensor = tf.tidy(() => {
        // Reshape the image to 28x28 px
        return examples.xs
          .slice([i, 0], [1, examples.xs.shape[1]])
          .reshape([28, 28, 1]);
      });

      const canvas: any = document.createElement("canvas");
      canvas.width = 28;
      canvas.height = 28;
      canvas.style = "margin: 4px;";
      await tf.browser.toPixels(imageTensor, canvas);
      surface.drawArea.appendChild(canvas);
      imageTensor.dispose();
    }
  }

  public async run() {
    const data = new DataApiService();
    await data.load();
    await this.showExamples(data);

    const model = this.getModel();
    tfvis.show.modelSummary({ name: "Model architecture" }, model);
    await this.train(model, data);

    // Save model and weights
    await model.save('downloads://digit-recognition');

    await this.showAccuracy(model, data);
    await this.showConfusion(model, data);
  }

  getModel() {
    const model = tf.sequential();
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    // First CONV layer
    model.add(
      tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling"
      })
    );
    // MaxPooling layer
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    // Second CONV layer
    model.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling"
      })
    );
    // MaxPooling layer
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    // Flatten output from the 2D filters into a 1D vector
    // Feeding data to final classification output layer
    model.add(tf.layers.flatten());

    // Last DENSE layer
    const NUM_OUTPUT_CLASSES = 10;
    model.add(
      tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        activation: "softmax",
        kernelInitializer: "varianceScaling"
      })
    );

    // Optimize model
    const optimizer = tf.train.adam();
    model.compile({
      optimizer,
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"]
    });

    return model;
  }

  async train(model, data) {
    const metrics = ["loss", "val_loss", "acc", "val_acc"];
    const container = {
      name: "Model training",
      style: { height: "1000px" }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 55000;
    const TEST_DATA_SIZE = 10000;

    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TRAIN_DATA_SIZE);
      return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
    });

    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE);
      return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
    });

    return model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 10,
      shuffle: true,
      callbacks: fitCallbacks
    });
  }

  doPredict(model, data, testDataSize = 500) {
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([
      testDataSize,
      IMAGE_WIDTH,
      IMAGE_HEIGHT,
      1
    ]);
    const labels = testData.labels.argMax([-1]);
    const preds = model.predict(testxs).argMax([-1]);

    testxs.dispose();
    return [preds, labels];
  }

  async showAccuracy(model, data) {
    const [preds, labels] = this.doPredict(model, data);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = { name: "Accuracy", tab: "Evaluation" };

    tfvis.show.perClassAccuracy(container, classAccuracy, this.classNames);
    labels.dispose();
  }

  async showConfusion(model, data) {
    const [preds, labels] = this.doPredict(model, data);
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = { name: "Confusion matrix", tab: "Evaluation" };

    tfvis.render.confusionMatrix(container, { values: confusionMatrix });
    labels.dispose();
  }
}
