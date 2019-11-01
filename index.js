const superagent = require('superagent');
const tf = require('@tensorflow/tfjs-node-gpu');
const toxicity = require('@tensorflow-models/toxicity');
const {print} = require('q-i');
const {default: surgeon} = require('surgeon');
//print(surgeon);
//const {cheerioEvaluator} = require('surgeon/dist/evaluators');

const x = surgeon(/*{
  evaluator: cheerioEvaluator()
}*/);

const THRESHOLD = 0.9;

const URL = process.argv[2];//print(URL);

superagent.get(URL).end((err, res) => {
  //print(err);
  //print(res);
  //print(res.text);
  const text = x(`select html | remove img {0,} | read property textContent`, res.text)
    .replace(/\n/g,'')
    .replace(/\s+/g, ' ').trim();
  print(text);
  toxicity.load(THRESHOLD).then(model => {
    const sentences = [res.text];
    model.classify(sentences).then(predictions => {
      const h = {};
      predictions.forEach(({label, results}) => {
        const [{match}] = results;
        h[label] = match;
      })
      print({URL, toxicity:h});
    });
  });
});
