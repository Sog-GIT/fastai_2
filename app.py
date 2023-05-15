import gradio as gr

def greet(name):
    return "Hello default" + name + "!!"

iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch()