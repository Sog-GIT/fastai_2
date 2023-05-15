from fastai.vision.all import *
import gradio as gr

def piclabels(x): return x.parent.name

learn =load_learner('modelCT.pkl')

categories = ('accordion', 'airplanes', 'anchor', 'ant', 
              'BACKGROUND_Google', 'barrel', 'bass', 'beaver', 
              'binocular', 'bonsai', 'brain', 'brontosaurus', 
              'buddha', 'butterfly', 'camera', 'cannon', 'car_side',
              'ceiling_fan', 'cellphone', 'chair', 'chandelier',
              'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 
              'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 
              'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 
              'ewer', 'Faces', 'Faces_easy', 'ferry', 'flamingo', 'flamingo_head', 
              'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 
              'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 
              'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'Leopards', 
              'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 
              'metronome', 'minaret', 'Motorbikes', 'nautilus', 'octopus', 
              'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 
              'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 
              'schooner', 'scissors', 'scorpion', 'sea_horse', 
              'snoopy', 'soccer_ball', 'stapler', 'starfish', 
              'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 
              'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 
              'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang')

def classify_image(img):
    pred,idx,probs =learn.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['c01.jpg', 'c02.jpg', 'c03.jpg', 'c04.jpg', 't01.jpg', 't02.jpg', 't03.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)

