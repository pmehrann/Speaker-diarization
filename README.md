# Speaker-diarization

The developed diarization model uses two pre-trained models to do the actual diarization task. The first model is the UISRNN model that receives the embeddings and predicts the speaker. A pre-trained UISRNN model can be found [here](saved_model.uisrnn_benchmark). The second pre-trained model is the embedding model that receives the chunks of speech and transforms them into embeddings. That model can be reconstructed using the neural networks weights. In the main function of the code, the embbedding network is re-constructed (109 classes for the VCTK dataset), and the wights are loaded.

The UIS-RNN model is originally proposed in the paper [Fully Supervised Speaker Diarization](https://arxiv.org/abs/1810.04719).
The UIS-RNN model that is used here is a modification of the original code [here](https://github.com/google/uis-rnn).

The Gostvlad model is originally proposed in the paper [GhostVLAD for set-based face recognition](https://arxiv.org/abs/1810.09951).
The Ghostvlad model that is used here is a modification of the original code [here](https://github.com/taylorlu/ghostvlad-speaker).


Main code can be found in the script named speakerDiarization.py which mainly uses a new mapping and a function that transforms the mapped labels to speakerSlice format. The speakerSlice format is needed to visualize the diarization results. These two functions are imported from the DER_scripts folder and the Mappings script.

To test the code on an example audio, dump the audio in the wavs folder, specify its path in the last line of the code and run the code. There is an example audio already in that folder that can be used.

The output is a pop-up window that can be played/paused using the space button on the keyboard. A pointer on the screen starts to move which points to the current time’s diarization output and is aligned with the played audio.

![image](https://user-images.githubusercontent.com/47835168/136057559-eee141bd-bd39-4604-98a9-a36215a29c8b.png)


