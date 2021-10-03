# Building a conversational chatbot using Tensorflow Seq2seq model.

### Dataset

Cornell Movie Dialogue Corpus
``https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html``

This dataset contains over 200+ conversational dialogues. 

### Steps to run

##### Over later versions of Tensorflow, we notice a lot of dependancy issues and changes in inbuilt function. Let us spin up a docker container with tf-1.0.0 to run the model process.

*Docker Install Guide* : https://docs.docker.com/engine/install/

#### Clone the repo

```bash
cd /some/dir
git clone https://github.com/advaithsrao/Chatbot_Seq2seq_model.git
```
#### Pull tensorflow-1.15-gpu image from https://hub.docker.com/r/tensorflow/tensorflow/tags?page=8&ordering=last_updated

```bash 
docker pull tensorflow/tensorflow:1.0.0 
```

Get the IMAGE_ID corresponding to REPOSITORY tensorflow/tensorflow

```bash
docker images
```


#### Spin up the container

```bash
docker run -it  -v Chatbot_Seq2seq_model:/notebooks <img_id> /bin/bash 
```

#### train/test the model 
**Train**
```bash
python3 chatbot.py 
```
**Test** 
```bash
test.py
```

### Literature on seq2seq architecture

1. **https://arxiv.org/abs/1409.3215** -> Sequence to Sequence Learning with Neural Networks by *Ilya Sutskever, Oriol Vinyals, Quoc V. Le*
2. **https://arxiv.org/abs/1406.1078** -> Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation by *Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio*