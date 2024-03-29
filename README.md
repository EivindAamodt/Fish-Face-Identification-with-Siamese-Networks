# Fish Face Identification with Siamese Networks

Final project for IKT450-G Deep Neural Networks

UiA article about the project: https://www.uia.no/en/news/developing-facial-recognition-to-track-fish

New NRK article: https://www.nrk.no/vestland/slik-skal-laksens-eigen-face-id-skilja-mellom-oppdrettsfisk-og-villaks-1.15503423

#### Requirements:
1. Pytorch

2. Dataset must have each class in its own folder. I made a small python file called `makedirectories.py` which is located inside `siamesenetwork/data`

You can change what dataset to run it on (face sameway, face bothways, body sameway, body bothways) in the `Config` class inside `siamese_network.py`

### This project is largely based on harveyslash's siamese network on AT&T face dataset.

Link to his github project: https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch

And his article: https://hackernoon.com/one-shot-learning-with-siamese-networks-in-pytorch-8ddaab10340e


##### `Git clone` can take a while, because i uploaded the entire image dataset as well so it is runnable without having to download and modify the dataset yourself.
