import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os 
import re
from matplotlib.colors import LogNorm

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

def load_csv_files(directory):
    f_data = []
    nome = []
    files = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')],
        key=lambda x: natural_sort_key(os.path.basename(x))  # Sort using natural order
    )
    
    for file in files:
        if os.path.basename(file).startswith('A'):
            f_data.append(np.loadtxt(file, unpack=True))
            nome.append((re.findall(r'A2_(\d+\.?\d*)_tau_(\d+\.?\d*)_beta_(\d+\.?\d*)',file)[0]))

    
    return f_data,nome

Dx_directory = "../Displ_x"
Dy_directory = "../Displ_y"
Dt_directory = "../Displ"

vetor_Dx,name_Dx = load_csv_files(Dx_directory)
vetor_Dy,name_Dy = load_csv_files(Dy_directory)
vetor_Dt,name_Dt = load_csv_files(Dt_directory)



plt.rcParams["mathtext.fontset"] = "cm" # Fonte matemática pro latex
plt.rc('font', family='serif',size=10) # fonte tipo serif, p fica paredico com latex msm
plt.rc('text', usetex=False) # esse vc deixa True e for salvar em pdf e False se for p salvar png






for i,j in enumerate(vetor_Dt):

    # Define the update function for the animation
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.tight_layout() # isso aq nsei bem qq faz mas ajuda a deixar menos espaço em branco
    fig.set_size_inches(18*0.393, 15*0.393) # esse fatir 0.393 é p converter polegadas p cm

    
    print(i)
    im1 = ax.imshow(j, cmap='YlGn',origin='lower',norm=LogNorm(), extent=[0, np.pi, 0,2*np.pi])
    plt.colorbar(im1)
    ax.set_aspect("auto")
    ax.set_aspect('auto')
    ax.set_ylabel(r'$y$')
    ax.set_xlabel(r'$x$')
    plt.savefig(f'../plot/Displ_{name_Dt[i]}.png',bbox_inches='tight',dpi=300)
    plt.close()


for i,j in enumerate(vetor_Dx):
    
    # Define the update function for the animation
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.tight_layout() # isso aq nsei bem qq faz mas ajuda a deixar menos espaço em branco
    fig.set_size_inches(18*0.393, 15*0.393) # esse fatir 0.393 é p converter polegadas p cm

    
    print(i)
    im1 = ax.imshow(j, cmap='PiYG',origin='lower', extent=[-np.pi, np.pi, -np.pi,np.pi])
    plt.colorbar(im1)
    ax.set_aspect("auto")
    ax.set_aspect('auto')
    ax.set_ylabel(r'$y$')
    ax.set_xlabel(r'$x$')
    plt.savefig(f'../plot/Displ_x_{name_Dx[i]}.png',bbox_inches='tight',dpi=300)
    plt.close()

for i,j in enumerate(vetor_Dy):
    
    # Define the update function for the animation
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.tight_layout() # isso aq nsei bem qq faz mas ajuda a deixar menos espaço em branco
    fig.set_size_inches(18*0.393, 15*0.393) # esse fatir 0.393 é p converter polegadas p cm

    
    print(i)
    im1 = ax.imshow(j, cmap='PiYG',origin='lower', extent=[-np.pi, np.pi, -np.pi,np.pi])
    plt.colorbar(im1)
    ax.set_aspect("auto")
    ax.set_aspect('auto')
    ax.set_ylabel(r'$y$')
    ax.set_xlabel(r'$x$')
    plt.savefig(f'../plot/Displ_y_{name_Dy[i]}.png',bbox_inches='tight',dpi=300)
    plt.close()






