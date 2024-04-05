import numpy as np
import time

""" Algoritmo de relajación (No modificar estas funciones)"""

def init_grid(a:int ,b:int ,d:float):
    """ Inicializa la red discretizada de una región del
    primer cuadrante del plano XY en una red de
    a x b ubicaciones (gridpoints).

    Args:
        a: Cantidad de ubicaciones en una fila
        b: Cantidad de ubicaciones en una columna
        d: Separación de las ubicaciones
        
    Returns:
        Un array de ceros de numpy de dimensiones a x b 
        para el potencial eléctrico. 
    """
    # Podemos trabajar exclusivamente en el primer cuadrante 
    # del plano XY, por isotropía e isomorfismo espacial:
    xmin = 0.0
    ymin = 0.0
    
    # Sabemos cuantos puntos tenemos en la red, y su separación.
    # Podemos obtener con esto xmax y ymax:
    xmax = a*d
    ymax = b*d
    
    # Inicializamos la matriz para el potencial eléctrico.
    V = np.zeros((a, b))
    
    # Inicializamos matrices para coordenadas x y coordenadas y.
    x_coords = np.linspace(xmin, xmax, a)
    y_coords = np.linspace(ymin, ymax, b)
    
    return V, x_coords, y_coords

def init_boundaries(v: np.ndarray, right = 0.0, left = 0.0, up = 0.0, down = 0.0):
    """ Inicializa las condiciones de frontera para
    la red discretizada de V.

    Args:
        v: Array inicial de ceros para el potencial eléctrico
        right: Condición de frontera del borde derecho
        left: Condición de frontera del borde izquierdo
        up: Condición de frontera del borde superior
        down: Condición de frontera del borde inferior
        
    Returns:
        Un array de numpy de dimensiones a x b 
        para el potencial eléctrico con las condiciones de 
        frontera de Dirichlet implementadas.
        
    Raises:
        TypeError: v debe ser un numpy array 2D
    """
    # Verificar si v es un numpy array 2D
    if not isinstance(v, np.ndarray) or v.ndim != 2:
        raise TypeError("v debe ser un numpy array 2D")
    
    # Se extraen las dimensiones de V
    N = v.shape[0]-1 # La indexación va de 0 a N-1 en lugar de 1 a N
    M = v.shape[1]-1
    
    # Se reemplazan las condiciones de frontera
    v[:,0] = left
    v[:,N] = right
    v[0,:] = up
    v[M,:] = down
    
    return v

def relax(v: np.ndarray,tol = 1e-4):
    """ Relaja la matriz que reciba como argumento hasta completar
    10000 iteraciones o hasta que la diferencia entre iteraciones
    sea menor a 1x10^-4
    
    Args:
        v: Array 2D para el potencial con las condiciones de frontera
        tol: valor de tolerancia que se compara con la diferencia,
             entre iteraciones, si la diferencia es menor termina la
             relajación (default = 1.0e-4)
        
    Returns:
        Un array de numpy de dimensiones a x b 
        para el potencial eléctrico, con el método de relajación
        iterativamente implementado hasta alcanzar la solución.
    """
    # Necesitamos una matriz auxiliar para el método
    v_temp = v.copy()
    
    # la región a relajar no incluye las fronteras.
    N = v.shape[0]-1 # Por ende sus dimensiones son N-1, M-1.
    M = v.shape[1]-1 
    
    # Ahora si podemos iniciar las iteraciones:
    start = time.time()
    for r in range(10000):
        v = v_temp.copy()
        
        # En la operación a continuación recorremos filas, luego columnas
        # Reemplazando cada elemento de la matriz auxiliar por el promedio
        # de los elementos cercanos y luego asignamos la matriz auxiliar 
        # a la matriz real.
        for j in range(1,M):
            for i in range(1,N):
                v_temp[i,j]= 0.25*(v[i+1,j] + v[i-1,j] + v[i,j+1] + v[i,j-1])
        # Esto lo haremos hasta alcanzar un número máximo de iteraciones
        # o hasta que la variación por iteración sea menor a la tolerancia.}
        variacion = np.abs(v - v_temp)
        if np.max(variacion) < tol:
            print(f'Los valores de la solución han dejado de cambiar sensiblemente a partir de la iteración #{r}')
            break
    stop = time.time()
    runtime = stop-start
    print(f'El método tardó {round(runtime,4)} segundos en ser completado.')
    return v                