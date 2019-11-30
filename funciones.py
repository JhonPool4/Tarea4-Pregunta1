import numpy as np
import pandas as pd
import matplotlib.collections as mc
import matplotlib.pyplot as plt


def epsilon_greedy(Q, state, nA, eps):
    """Selecciona una acción epsilon-greedy para un estado
    
    Parámetros
    ==========
        Q (diccionario): función de valor de acción
        state (int): estado actual
        nA (int): número de acciones en el entorno
        eps (float): epsilon
    """


    if np.random.random() > eps:
        return np.argmax(Q[state])
    else:
        return np.random.choice( np.arange(nA) )


def crear_grilla_uniforme(low, high, bins=(10, 10)):
    """Crea una grilla uniformemente espaciada para discretizar el espacio.
    
    Parámetros
    ----------
        low - Arreglo que indica los límites inferiores para cada dimensión del espacio continuo
        high - Arreglo que indica los límites superiores para cada dimensión del espacio continuo
        bins - Tupla que indica el número de "bins" en cada dimensión.
    
    Retorna
    -------
        grid - lista de arreglos que contiene los puntos divididos de cada dimensión.
    """
    dim_bins = len(bins)
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(dim_bins)]
    print("Grilla uniforme:")
    for l, h, b, splits in zip(low, high, bins, grid):
        print(" * nbins: {}, Límites: [{:.2f}, {:.2f}], Resultado: {}".format(b, l, h, np.round(splits, 2)))
    return grid


def discretizar(sample, grid):
    """Discretiza una muestra con la grilla provista
    
    Parámetros
    ----------
        sample - Arreglo que contiene una sola muestra del espacio continuo original
        grid - Lista de arreglos que contiene los puntos de separación de cada dimesión
    
    Retorna
    -------
        discretized_sample - Arreglo que contiene la secuencia de enteros con el mismo
                             número de dimensiones que la muestra
    """
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))



def visualizar_muestras(samples, discretized_samples, grid, low=None, high=None):
    """Visualizar la muestra original y discretizada en una grilla de dimensión 2."""

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Mostrar grilla
    ax.xaxis.set_major_locator(plt.FixedLocator(grid[0]))
    ax.yaxis.set_major_locator(plt.FixedLocator(grid[1]))
    ax.grid(True)
    
    # Si los límites (low, high) son especificados, usarlos para establecer límites en los ejes
    if low is not None and high is not None:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    else:
        # De otro modo, usar las localizaciones primer y última de la grilla (para mapeo de las muestras discretas)
        low = [splits[0] for splits in grid]
        high = [splits[-1] for splits in grid]

    # Mapa de cada muestra discretizada (que es un índice) al centro de la celda correspondiente
    grid_extended = np.hstack((np.array([low]).T, grid, np.array([high]).T))  # Añade low y high
    grid_centers = (grid_extended[:, 1:] + grid_extended[:, :-1]) / 2         # Calcula el centro de cada grilla
    locs = np.stack(grid_centers[i, discretized_samples[:, i]] for i in range(len(grid))).T  # Mapea las muestras discretizadas

    ax.plot(samples[:, 0], samples[:, 1], 'o')  # Mostrar las muestras originales
    ax.plot(locs[:, 0], locs[:, 1], 's')        # Mostrar las muestras discretizadas
    # Añadir una línea que conecte cada muestra a su discretización
    ax.add_collection(mc.LineCollection(list(zip(samples, locs)), colors='orange'))  
    ax.legend(['original', 'discretizado'])


def plot_G(scores, rolling_window=100):
    """Graficar los retornos y el promedio usando una ventana deslizante"""
    plt.plot(scores); plt.title("Scores");
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean);
    return rolling_mean