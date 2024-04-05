import Relax
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

""" TEST """

# Número de gridpoints de la red (ENTRADA)
N = 100
M = 100

# Tamaño del paso (ENTRADA)
h = 0.005

V,x,y = Relax.init_grid(N,M,h)
X,Y = np.meshgrid(x,y)

# Condiciones de frontera (ENTRADA)
v_0 = 1
R = 0
L = 0
U = -v_0
D = v_0

V = Relax.init_boundaries(V,R,L,U,D)
V = Relax.relax(V)
    
# 3D Plot
fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection='3d')
ax.plot_wireframe(X,Y,V,color = 'k',label='Potencial eléctrico computado')
ax.set_xlabel(r'$X$')
ax.set_ylabel(r'$Y$')
ax.set_zlabel(r'Potencial eléctrico $V$')
ax.set_title('Plot 3D de la solución')
    
# Curvas de nivel y mapa de calor
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Gráficas 2D de la solución')

# Curvas de nivel
ax1.grid()
ax1.set_xlabel(r'$X$')
ax1.set_ylabel(r'$Y$')
ax1.set_title('Curvas de nivel')
Lvl_crvs = ax1.contour(X, Y, V)
ax1.clabel(Lvl_crvs, Lvl_crvs.levels, inline = True, fontsize = 8)

# Mapa de calor
ax2.grid()
ax2.set_xlabel(r'$X$')
ax2.set_ylabel(r'$Y$')
ax2.set_title('Mapa de calor')
heatmap = ax2.pcolormesh(X, Y, V, cmap='hot')
plt.colorbar(heatmap, ax=ax2)

plt.show()

def V_a(x,y,b,a,n):
    
    # Constantes
    k = (np.pi*n)/a
    C = (4*v_0*(1-((-1)**n)))/(n*np.pi)*(1/(np.tanh((n*np.pi*b)/(a))))
    D = (-2*v_0*(1-((-1)**n)))/(n*np.pi)
    
    return C*((np.sin(k*x)*np.sinh(k*y)))+D*((np.sin(k*x)*np.cosh(k*y)))

V_analytical = V_a(X,Y,0.5,0.5,1)

# print(V_analytical.ndim, V_analytical.shape)

fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection='3d')
ax.plot_surface(X,Y,V,label='Potencial eléctrico computado',cmap = 'hot')
ax.plot_wireframe(X,Y,V_analytical,color = 'r',label='Potencial eléctrico analítico')
ax.set_xlabel(r'$X$')
ax.set_ylabel(r'$Y$')
ax.set_zlabel(r'Potencial eléctrico $V$')
ax.set_title('Plot 3D de la solución')
# ax.legend()
plt.show()