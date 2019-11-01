const superagent = require('superagent');
const tf = require('@tensorflow/tfjs-node-gpu');
const toxicity = require('@tensorflow-models/toxicity');
const {print} = require('q-i');
const {default: surgeon} = require('surgeon');
const {sentences: findSentences} = require('sbd');
const setIn = require('set-value');
//const strShorten = require('str_shorten')
//print(surgeon);
//const {cheerioEvaluator} = require('surgeon/dist/evaluators');

const x = surgeon(/*{
  evaluator: cheerioEvaluator()
}*/);

//const LENGTH = 100;

const WHITELIST_PATTERN = `[^-a-zA-Z_ '''"ÆÐƎƏƐƔĲŊŒẞÞǷȜæðǝəɛɣĳŋœĸſßþƿȝĄƁÇĐƊĘĦĮƘŁØƠŞȘŢȚŦŲƯY̨Ƴąɓçđɗęħįƙłøơşșţțŧųưy̨ƴÁÀÂÄǍĂĀÃÅǺĄÆǼǢƁĆĊĈČÇĎḌĐƊÐÉÈĖÊËĚĔĒĘẸƎƏƐĠĜǦĞĢƔáàâäǎăāãåǻąæǽǣɓćċĉčçďḍđɗðéèėêëěĕēęẹǝəɛġĝǧğģɣĤḤĦIÍÌİÎÏǏĬĪĨĮỊĲĴĶƘĹĻŁĽĿʼNŃN̈ŇÑŅŊÓÒÔÖǑŎŌÕŐỌØǾƠŒĥḥħıíìiîïǐĭīĩįịĳĵķƙĸĺļłľŀŉńn̈ňñņŋóòôöǒŏōõőọøǿơœŔŘŖŚŜŠŞȘṢẞŤŢṬŦÞÚÙÛÜǓŬŪŨŰŮŲỤƯẂẀŴẄǷÝỲŶŸȲỸƳŹŻŽẒŕřŗſśŝšşșṣßťţṭŧþúùûüǔŭūũűůųụưẃẁŵẅƿýỳŷÿȳỹƴźżžẓ]+`;
const WHITELIST_REGEXP = new RegExp(WHITELIST_PATTERN, 'gi');

const THRESHOLD = 0.9;

const URL = process.argv[2]; print(URL);

superagent.get(URL).end((err, res) => {
  //print(err);
  //print(res);
  //print(res.text);
  const text = x(`select html | remove img {0,} | read property textContent`, res.text)
    .replace(/\n/g,'')
    .replace(/\s+/g, ' ').trim();

  const sentences = findSentences(text, {}).map(str => str.replace(WHITELIST_REGEXP, ''));
  //print(sentences);

  toxicity.load(THRESHOLD).then(model => {
    model.classify(sentences).then(predictions => {
      const s = {};
      //print(predictions);
      predictions.forEach(({label, results}) => {
        results.forEach((result, i) => {
          const sentence = sentences[i];
          const {match} = result;
          //setIn(s, `${sentence}.${label}`, result);
          if(match) {
            setIn(s, `${sentence}.${label}`, match);
          }
        });
      });
      print(s);
    });
  });

});
