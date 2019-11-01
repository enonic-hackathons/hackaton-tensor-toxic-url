const tf = require('@tensorflow/tfjs-node-gpu');
const toxicity = require('@tensorflow-models/toxicity');
const {print} = require('q-i');
const THRESHOLD = 0.9;
toxicity.load(THRESHOLD).then(model => {
  const sentences = ['you suck'];
  model.classify(sentences).then(predictions => {
    const h = {};
    predictions.forEach(({label, results}) => {
      const [{match}] = results;
      h[label] = match;
    })
    print(h);
  });
});
