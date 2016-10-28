# asr (Acoustic Speech Recognition)
Acoustic Speech Recognition with Hyper-Networks

(1) Baseline Model (Single Utterance)
- Stacked LSTM with single utterance train setting
- Test model with single utterance
- Test model with two concatenated utterances

(2) Baseline Model (Two Utterances)
- Stacked LSTM with two utterances train setting
- Test model with single utterance
- Test model with two concatenated utterances

(3) Baseline Model (Two Utterances) using I-Vector
- Considering I-Vector as addtional condition data
- Stacked LSTM with two utterances train setting
- Test model with single utterance
- Test model with two concatenated utterances

(4) Hyper Network Model (Two Utterances) without Regularizers
- Using hyper-networks
- Stacked LSTM with two utterances train setting
- Test model with single utterance
- Test model with two concatenated utterances

(5) Hyper Network Model (Two Utterances) with Regularizers (Slow-Feature)
- Using hyper-networks with additional cost terms (slow feature)
- Stacked LSTM with two utterances train setting
- Test model with single utterance
- Test model with two concatenated utterances
