import numpy as np
import typing as tp
from scipy.special import factorial, genlaguerre

if __name__ == "__main__":
    # number space
    N=5
    N_x=np.arange(N,dtype=np.int32)
    N_y=np.arange(N,dtype=np.int32).T

    N_meshx,N_meshy=np.meshgrid(N_x,N_y)
    # TODO N_list_x_minus_y = np.unique(N_x-N_y)

    # wigner function space
    maxX=maxP=3.0
    nXP=11#must be odd
    XP_x=np.linspace(-nXP//2,nXP//2,nXP, dtype=np.int32)
    XP_y=np.linspace(-nXP//2,nXP//2,nXP, dtype=np.int32).T

    XP_meshx,XP_meshy=np.meshgrid(XP_x,XP_y)
    
    XPfloat_x=(XP_x*maxX/(nXP//2)).astype(np.float32)
    XPfloat_y=(XP_y*maxP/(nXP//2)).astype(np.float32)

    XPfloat_meshx,XPfloat_meshy=np.meshgrid(XPfloat_x,XPfloat_y)


    # Density matrix and Wigner function
    D=np.zeros_like(N_meshx,dtype=np.float32)
    rng=np.random.default_rng()
    D=rng.uniform(-1,1,size=N_meshx.size).astype(np.float32)
    W=np.zeros_like(XP_meshx,dtype=np.float32)

    _factorial=factorial(N_x)
    #N_x#list_lagn=np.unique(N_meshx)
    mesh_a=np.abs(N_meshx-N_meshy)
    list_a=np.unique(mesh_a)
    mesh_x2p2=XP_meshx**2+XP_meshy**2
    list_x2p2=np.unique(mesh_x2p2)
    laguerre_mesh_n,laguerre_mesh_a,laguerre_mesh_x2p2=np.meshgrid(N_x,list_a,list_x2p2)
    laguerre_values = np.zeros_like(laguerre_mesh_n,dtype=np.float32)
    print(laguerre_values.shape)
    for n in N_x:
        for a in list_a:
            laguerre_values[n,a,:]=genlaguerre(n,a)(2*maxX/nXP*list_x2p2)
    #laguerre_1234=laguerre_values[N_meshx,mesh_a].where()
    print(laguerre_values[2,2].shape)
    print(laguerre_values[0,2])
    print(laguerre_values[2,2])
    #exp_values=np.zeros_like(list_x2p2,dtype=np.complex64)
    
    N_meshx_23=np.expand_dims(N_meshx,[0,1])
    N_meshy_23=np.expand_dims(N_meshy,[0,1])
    mesh_a_23=np.expand_dims(mesh_a,[0,1])
    wig = 1/np.pi*np.exp(-mesh_x2p2)*(-1)**N_meshx_23*\
            (XP_meshx+1j*XP_meshy)**(mesh_a_23)*\
            np.sqrt(2**mesh_a_23*factorial(N_meshx_23)/factorial(N_meshy_23))
            

    def fromDtoW(D:np.ndarray, W:np.ndarray, XP_x:np.ndarray, XP_y:np.ndarray, N_x:np.ndarray, N_y:np.ndarray):
        for i in range(XP_x.size):
            for j in range(XP_y.size):
                W[i,j]=np.sum(D*np.exp(-2j*np.pi*(XP_x[i]*N_x+XP_y[j]*N_y)/N))
        return W