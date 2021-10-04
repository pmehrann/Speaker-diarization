import pickle
from visualization.viewer import PlotDiar

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_gt(name ):
    with open('ground_truth/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# PROMPTING FOR THE NAME OF THE AUDIO FILE FOR DEMO
filename = input("Enter the audio file's name: ")
kind = input("Entre 0 for visualizing prediction or 1 for visualizing groundtruth: ")

wav_path = r'wavs/' + filename + '.wav'
embedding_per_second=1.2
overlap_rate=0.4
speakerSlice = load_obj(filename) if int(kind) == 0 else load_gt(filename)


p = PlotDiar(map=speakerSlice, wav=wav_path, gui=True, size=(25, 6))
p.draw()
p.plot.show()