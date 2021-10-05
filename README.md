# Speaker-diarization

The developed diarization model uses two pre-trained models to do the actual diarization task. The first model is the UISRNN model that receives the embeddings and predicts the speaker. A pre-trained UISRNN model can be found [here](saved_model.uisrnn_benchmark).

The second pre-trained model is the embedding model that receives the chunks of speech and transforms them into embeddings. That model can be reconstructed using the neural networks weights. The pre-trained weights are \textbf{VCTK\_weights.h5} which is the result of training the network on VCTK dataset. In the main function of the code, the embbedding network is re-constructed (109 classes for the VCTK dataset), and the wights are loaded.

The UIS-RNN model is originally proposed in the paper [Fully Supervised Speaker Diarization](https://arxiv.org/abs/1810.04719).
The UIS-RNN model that is used here is a modification of the original code [here](https://github.com/google/uis-rnn).

The Gostvlad model is originally proposed in the paper [GhostVLAD for set-based face recognition](https://arxiv.org/abs/1810.09951).
The Ghostvlad model that is used here is a modification of the original code [here](https://github.com/taylorlu/ghostvlad-speaker).


The model takes an audio file as input and 

The output is a pop-up window that can be played/paused using the space button on the keyboard. A pointer on the screen starts to move which points to the current time’s diarization output and is aligned with the played audio.

![image](https://user-images.githubusercontent.com/47835168/136057559-eee141bd-bd39-4604-98a9-a36215a29c8b.png)


