import string
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

import numpy as np



def vocabulary(blank = '-', start = '@', stop = '#'):
    """
        Function that returns a vocabulary\n
        Attributes:
        - blank: blank character used as a delimiter to double letters
        - start: character used as start string 
        - stop: character used as end string 

    """
    return [blank] + list(string.ascii_lowercase) + ['.', '?', ',', '!', start, stop, ' ']


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
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
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