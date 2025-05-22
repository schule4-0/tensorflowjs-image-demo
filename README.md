# tensorflowjs-image-demo

Small image classification demo in the browser with tensorflowjs.
It can run any

1. `npm install`
2. `npm run dev`

Tech:

- NextJs
- tensorflowjs (public/tf.min.js)
- mobilenet-v3 as an example image classification model
  - the model.json file contains the layer and output labels information
  - the .bin files contain the model weights. These weight files are referenced in the model.json file
  - Downloaded from https://www.kaggle.com/models/google/mobilenet-v3/tfJs

More information about saving and loading models in tensorflowjs: https://www.tensorflow.org/js/guide/save_load
