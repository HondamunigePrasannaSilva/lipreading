import string
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import torch
import numpy as np

import os

def create_vocabulary(blank):
    """
        Function that returns a vocabulary\n
        Attributes:
        - blank: blank character used as a delimiter to double letters
        - start: character used as start string 
        - stop: character used as end string 

    """
    #return [blank] + list(string.ascii_lowercase) + ['.', '?', ',', '!',"’", "'", ';',':', ' ', '-'] + ['#']
    return [blank]+['#']+ list(string.ascii_lowercase)+[' '] + ['.', '?', ',', '!',"’", "'", ';',':', '-'] #


def process_string(input_string):
    
    output_string = ""
    current_char = ""

    for char in input_string:
        if char != current_char:
            if char.isalpha() or char == '0':
                if char == '0':
                    output_string += ' '
                else:
                    output_string += char   
            current_char = char

    return output_string.strip()



def plot_3d_point(x, y, z):
    fig = go.Figure(data=[go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', marker=dict(color='red', size=5))])
    fig.update_layout(scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z')))
    fig.show()


def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1, 1, 1
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                    transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                            ]
                                    )
                                ]
                    ),
                    frames=frames
            )

    return fig

def pcshow(xs,ys,zs):
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()

def process_string(input_string, blank= "@"):
    output_string = ""
    current_char = ""

    for char in input_string:
        if char != current_char:
            if char == blank:
                output_string += ''
            else:
                output_string += char
                
            current_char = char
        else:
            output_string += ''


    return output_string.strip()


def char_to_index_batch(label, vocabulary):
    #vocabulary = vocabulary(blank='-', start='@', stop='#')
    char_to_index = {char: index for index, char in enumerate(vocabulary)}
    labels = []
    for ilab in label:
        
        target_indices = [char_to_index[char] for char in ilab]
        #print(target_indices)
        labels.append(target_indices)


    labels = torch.tensor(labels)
    return labels

def save_results(file_name, real_sentences, predicted_sentences, overwrite=False):
    if overwrite:
        if os.path.exists(file_name):
            os.remove(file_name)
        
    f = open(file_name, "a")
 
    for i in range(len(predicted_sentences)):
        f.write(real_sentences[i]+"\n")
        f.write(predicted_sentences[i]+"\n")
        f.write("\n")
    f.close() 
    return

def write_results(len_label, label_list, output, batch_size, vocabulary, real_sentences, pred_sentences):
    len_label.cpu()
    real_sentences_temp = [x[:len_label[i]] for i, x in enumerate(label_list)]
    real_sentences = real_sentences + real_sentences_temp
    output_cpu = output.detach().cpu()

    for i in range(batch_size):
        e = torch.argmax(output_cpu[:, i, :], dim=1)
        output_sequence = ''.join([vocabulary[index] for index in e])
        pred_sentences.append(output_sequence)
    
    return real_sentences, pred_sentences

def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = torch.nn.functional.interpolate(features,size=output_len,align_corners=True,mode='linear')
    return output_features.transpose(1, 2)