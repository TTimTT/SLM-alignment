# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:27:59 2021

@author: gnoeting
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:13:27 2019

@author: gnoeting
"""
import skimage.transform as skit
import scipy.ndimage as scim
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
import progressbar
import pickle
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


"FONCTIONS DE CREATION & MANIPULATION DE SEQUENCE "

def DMDRotationCalc(M,Ntot,Mconst=None,ThetaCorr=None,save=None,Mf=None):
    """
    Calcul d'une séquence d'images obtenues par rotation à partir d'une image initiale

    INPUT :
    M : matrice de type ndarray à faire tourner de taille = dimension du DMD ;
    Ntot: nombre d'images de la séquence
    ThetaCorr: correction d'angle
    save : nom de la séquence si on souhaite la sauvegarder

    OUTPUT : pas d'output pour Spyder
    Sauvegarde de la séquence comme un vecteur biplane affichable directement par le DMD dans un fichier nommé : nom + 'Seq'
    Sauvegarde de la séquence comme une pile de tableau ndarray dans un fichier nommé : nom + 'Stack'

    Ex :  GeomDMD.DMDRotationCalc(M,50,test,12)
    """
    seuil=240
    # + petit angle approx entre 2 pixels du DMD (en bonne approx) 0.0005 rad on peut se contenter de 0.01 rad ici soit  environ 0.5deg
    if Mconst is None :
        Mconst=np.zeros_like(M)
    dTheta=2*np.pi/Ntot # Pas d'angle pour faire un tour complet en Ntot images
    print('Ntot=',Ntot,' images  soit un pas d\'angle de', dTheta, ' rad' )#nombre d'images sur nombre total avec le pas d'angle choisi
                                                        # s'il n'y avait pas de saut d'images (à cause de la variable intervalle)
    Stack=[] # Création pile vide pour concaténation
    V=M
    imgSeq=np.asarray(Im2Bp(M)).ravel()
    get_ipython().run_line_magic('matplotlib', 'inline') # graphe en console pour affichage des images en console
    n_im=1
    bar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],max_value=Ntot).start()
    for i in range(0,Ntot):
        V=RotationPoint(M,0,0,i*dTheta*180/np.pi,ThetaCorr) #Création tableau tourné V=GeomDMD.RotationPoint(M,0,0,0.05*180/np.pi,12)
        V=V+Mconst
        V=255*(V>seuil)
        Stack.append(V)
        imgSeq=np.concatenate([imgSeq,Im2Bp(V)])#imgSeq=np.concatenate(imgSeq,[V.ravel()]),np.asarray(GeomDMD.Im2Bp(V.astype(int)))
        #print("Iteration numero ",i+1)
        plt.matshow(V)
        plt.show()
        n_im=n_im+1
        bar += 1
    if Mf is not None:
        imgSeq=np.concatenate([imgSeq,Im2Bp(Mf)])
        Stack.append(Mf)
    get_ipython().run_line_magic('matplotlib', 'qt5') #graphe en fenêtre
    # Allocate the onboard memory for the image sequence
    #imgSeq=imgSeq.astype(np.int32)
    #Sauvegarde du fichier
    if save:
        outfile= open(save+'Seq', 'wb')
        pickle.Pickler(outfile).dump(imgSeq)
        outfile.close()
        outfile= open(save+'Stack', 'wb')
        pickle.Pickler(outfile).dump(Stack)
        outfile.close()
        print("Fichier sauvegarde")
    return imgSeq,Stack
    print("Nombre d'images=",n_im)
    # Send the image sequence as a 1D list/array/numpy array
    # GeomDMD.DMDRotationCalc(GeomDMD.MotifCible(1200,1920,1000,960,20),12,'testBP',12)
    # GeomDMD.DMDRotationCalc(GeomDMD.MotifRectangle(1200,1920,1000,100),250,'testBP',12)

def Dédouble(nomdépart,angle,nomarrivée): #Pour dédoubler une stack en périodisant le motif tournant par rapport au centre de rotation
    #angle en degré

    with open(nomdépart,'rb') as f:
       Stack = pickle.Unpickler(f).load()
       f.close()
       print('Fichier récupéré')

    N_angle=np.round(np.shape(Stack)[0]*(angle/360)) #Nb de décalage d'image nécessaire pour décaler la séquence de l'angle indiquée en arg
    Stacktot=Stack
    #img_nav.mat_nav(np.transpose(Stack,(1,2,0)))

    Stack2=np.roll(Stack,np.int(N_angle),axis=0) #on fait une permutation circulaire
    Stacktot=Stacktot+Stack2

    #img_nav.mat_nav(np.transpose(Stack2,(1,2,0))) #Controle
    #img_nav.mat_nav(np.transpose(Stacktot,(1,2,0)))

    outfile=open(nomarrivée, 'wb')
    pickle.Pickler(outfile).dump(Stacktot)
    outfile.close()

    return(Stacktot)

    #Exemple : Dédouble('Secteur45degStack',90,'SecteurDédoubléQuadrature45degStack')

"FONCTIONS DE MANIPULATIONS DES IMAGES"

def RotationPoint(M,xc,yc,Theta,ThetaCorr=None): # Fonction de rotation d'un motif par rapport à un point
    #avec correction projection orthogonale Thetacorr optionelle
    #x_out=np.cos(Theta)*(x-xc)-np.sin(Theta)*(y-yc)+xc
    #y_out=np.sin(Theta)*(x-xc)+np.cos(Theta)*(y-yc)+yc
    #return(x_out,y_out)
    V=scim.shift(M,[-xc,-yc],mode='constant',cval=M[0,0])#,mode='wrap'
    V=scim.rotate(V,Theta,axes=(1,0),reshape=False,mode='constant',cval=M[0,0])
    V=scim.shift(V,[xc,yc],mode='constant',cval=M[0,0])#,mode='wrap'
    if ThetaCorr!=None:
        V=CorrectionAngle(V,ThetaCorr,kind='X')
    return(V)

def CorrectionAngle(V,ThetaCorr,kind='X'): # rotation puis élongation dans une direction et rerotation
    size=np.shape(V)
    c=V[0,0]
    if ThetaCorr!=None:
        if kind=='X':
            V=scim.rotate(V,45,axes=(1,0),reshape=False,mode='constant',cval=c)
            V=scim.zoom(V,(1,1/np.cos(ThetaCorr*np.pi/180)),mode='constant',cval=c)
            V=scim.rotate(V,-45,axes=(1,0),reshape=False,mode='constant',cval=c)
            #V=skit.resize(V,size, anti_aliasing=True, preserve_range=1,mode='constant',cval=c)
            V=V[:,np.shape(V)[1]//2-size[1]//2:np.shape(V)[1]//2+size[1]//2]
            V=255*(V > 50)
        else :
            V=scim.rotate(V,45,axes=(1,0),reshape=False,mode='constant',cval=c)
            V=scim.zoom(V,(np.cos(ThetaCorr*np.pi/180),1),mode='constant',cval=c)
            V=scim.rotate(V,-45,axes=(1,0),reshape=False,mode='constant',cval=c)
            #V=skit.resize(V,size, anti_aliasing=True, preserve_range=1,mode='constant',cval=c)
            l_pad=size[0]//2-np.shape(V)[0]//2
            V=np.pad(V,((l_pad,l_pad-l_pad%2),(0,0)))[:size[0],:size[1]]
            V=255*(V > 50)
    return(V)

def Period(M,N): #Pour périodiser une image par rotation par rapport au centre
    angle=360/N #angle de rotation
    V=M
    Vrot=M #initialisation
 #   xc,yc=np.shape(M)[0]//2,np.shape(M)[1]//2 #Coordonnées du centre de l'image
    for i in range(0,N-1):
        Vrot=scim.rotate(Vrot,angle,reshape=False)
        V=V+Vrot#On ajoute la matrice tournée
        V[np.nonzero(V>=255)]=255 #On fixe le max d'intensité à 255
        V[np.nonzero(V<1)]=0
    return V

def Complémentaire(M):
    V=Blanc(np.shape(M)[1],np.shape(M)[0])-M
    return V

"FORMES GEOMETRIQUES"

def Blanc(Lx,Ly):
    M=np.ones([Ly,Lx],np.int)*(2**8-1)# Une matrice Lx*Ly de 255
    return(M)
# Blanc(1200,1920)

def MotifCible(Lx,Ly,R1,R2,d): # Un motif de base pour test et alignement : un cercle barré
    #Lx largeurDMD,Ly longueur DMD, R1 et R2 diamètres extérieur et intérieur du cercle, d épaissseur des diagonales
    x,y=np.arange(Lx),np.arange(Ly)
    x,y=x-Lx/2+0.5,y-Ly/2+0.5
    X,Y=np.meshgrid(x,y)
    R,Theta=np.sqrt(X**2+Y**2),np.arctan2(Y,X)
    M=(R>R2)*(R<R1)+(X-Y>-d)*(X-Y<d)+(-X-Y>-d)*(-X-Y<d)
    return np.transpose(255*(M>=1))
    #M=MotifCible(1200,1920,500,420,30)

def MotifAntiCible(Lx,Ly,R1,R2,d): # Un motif de base pour test : un cercle en pointillé
    x,y=np.arange(Lx),np.arange(Ly)
    x,y=x-Lx/2+0.5,y-Ly/2+0.5
    X,Y=np.meshgrid(x,y)
    R,Theta=np.sqrt(X**2+Y**2),np.arctan2(Y,X)
    M=np.bitwise_and(((R>R2)*(R<R1)),((R>R2)*(R<R1))^((X-Y>-d)*(X-Y<d))^((-X-Y>-d)*(-X-Y<d)^((X>-d)*(X<d))^((Y>-d)*(Y<d))))
    return np.transpose(255*(M>=1))
    #exemple représentatif : M=MotifAntiCible(1200,1960,600,500,40)

def MotifCible4(Lx,Ly,R1,R2,d): #un motif constitué de 4 cibles dans différents quadrants pour évaluer les aberrations dans chaque quadrant et vérifier que
#kles alignements permettent de laisser passer de manière plus ou moins identique la lumière incidente dans chaque quadrants 
#Lx largeurDMD,Ly longueur DMD, R1 et R2 diamètres extérieur et intérieur des cercles de chaque quadrants, d épaissseur des diagonales
    v_trans=[[Lx//4,Ly//4],[Lx//4,-Ly//4],[-Lx//4,-Ly//4],[-Lx//4,Ly//4]] #le vecteur de translation
    M=MotifCible(Lx,Ly,R1,R2,d)*MotifCercle(Lx,Ly,R1)
    M=scim.shift(M,v_trans[0])+scim.shift(M,v_trans[1])+scim.shift(M,v_trans[2])+scim.shift(M,v_trans[3])
    return np.transpose(255*(M>=1))

#exemple représentatif : MotifCible4(1200,1920,400,350,30)

def MotifSecteur(Lx,Ly,Phi,Rmax,Rmin=0,Phi0=0):#Un secteur(="camembert") avec un rmin
    #angle en degré comprit entre 0 et 180 degrés
    Phi=Phi*np.pi/180
    x,y=np.arange(Lx),np.arange(Ly)
    x,y=x-Lx/2+0.5,y-Ly/2+0.5
    X,Y=np.meshgrid(x,y)
    R,Theta=np.sqrt(X**2+Y**2),np.arctan2(Y,X)
    M=(R>Rmin)*(R<Rmax)*(Theta<Phi+Phi0)*(Theta>Phi0)
    return np.transpose(255*(M>=1))
    #M=GeomDMD.MotifSecteur(1200,1920,90,600,200)

def MotifRectangle(Lx,Ly,lx,ly):
    x,y=np.arange(Lx),np.arange(Ly)
    x,y=x-Lx/2+0.5,y-Ly/2+0.5
    X,Y=np.meshgrid(x,y)
    M=(X>-lx/2)*(X<=lx/2)*(Y>-ly/2)*(Y<=ly/2)
    return 255*(M>=1)
    #M=GeomDMD.MotifRectangle(1920,1200,600,60)

def MotifAnneau(Lx,Ly,Rmax,Rmin):
    x,y=np.arange(Lx),np.arange(Ly)
    x,y=x-Lx/2+0.5,y-Ly/2+0.5
    X,Y=np.meshgrid(x,y)
    R,Theta=np.sqrt(X**2+Y**2),np.arctan2(Y,X)
    M=(R>Rmin)*(R<Rmax)
    return np.transpose(255*(M>=1))
    #M=GeomDMD.MotifAnneau(1200,1920,600,60)

def MotifCercle(Lx,Ly,R): #Permet de faire un anneau et dans le cas limite où Rmin=0 un cercle
    M=MotifAnneau(Lx,Ly,R,0)
    return(M)
    #M=GeomDMD.MotifCercle(1200,1920,600)

def MotifSpiraleLog(Lx,Ly,Rmax,A,B,angle): #spirale log equation r=a*b^theta et A=a*a et B=b*b
    M=np.zeros([Lx,Ly],np.int)
    for j in range(0,Ly):
        for i in range(0,Lx):
            r2=(j-Ly/2)**2+(i-Lx/2)**2
            theta=np.arctan2(j-Ly/2,i-Lx/2)
            if theta>((np.log(r2)-np.log(A))/np.log(B))%(2*np.pi)-np.pi and theta-angle<((np.log(r2)-np.log(A))/np.log(B))%(2*np.pi)-np.pi and r2<Rmax**2 :  # and np.arctan2(j,i)-angle>(np.log(r2)-np.log(A))/np.log(B): # and (j-Ly/2)**2+(i-Lx/2)**2>A*B**(np.arctan2(j,i)+angle) and (j-Ly/2)**2+(i-Lx/2)**2<Rmax**2:
                M[i,j]=255
#    plt.matshow(M)
#    plt.show()
    return(M)
    #M=GeomDMD.MotifSpiraleLog(1200,1920,600,10,10,1)

def MotifTriangleEqui(Lx,Ly,l):
    M=np.zeros([Lx,Ly],np.int)
    for j in range(0,Ly):
        for i in range(0,Lx):
            if i+Lx/2<+(-j+Ly/2+l)*np.sqrt(3) and i+Lx/2<+(j-Ly/2+l)*np.sqrt(3) and i-Lx/2>-l/2/np.sqrt(3) :# : i+Lx/2>+(j-Ly/2+l)*np.sqrt(3) and
                M[i,j]=255
    # plt.matshow(M)
    # plt.show()
    return(M)
    #M=GeomDMD.MotifTriangleEqui(1200,1920,1020)

def Motif8(Lx,Ly,l):
    M=MotifRectangle(Lx,Ly,4*l,5*l)
    M1,M2=MotifRectangle(Lx,Ly,2*l,1.5*l),MotifRectangle(Lx,Ly,2*l,1.5*l)
    M1,M2=scim.shift(M1,[l,0]),scim.shift(M1,[-l,0])
    return M-M1-M2

def SerieMire(Dim,rmax=1,rmin=0): #Valeur de tailles en pixels, coeff de réflexion  en amplitude
    [NyObj,NxObj,nyObj,nxObj]=Dim
    Rect=rmax*MotifRectangle(NyObj,NxObj,nyObj,nxObj)/255
    Obj=scim.shift(Rect,[-2*nyObj,0],mode='constant',cval=rmin)+scim.shift(Rect,[+2*nyObj,0],mode='constant',cval=rmin)+Rect
    #Obj=scim.shift(Obj,[0,-nxObj//3],mode='constant',cval=rmin)
    x0=np.nonzero(Obj<(rmin+0.1))
    Obj[x0]=rmin
    return(Obj)
    #plt.matshow(M)

def ElementMire(n_groupe,n_élement,dim,rmax=1,rmin=0):#numéro du groupe, numéro de l'élement, taille totale souhaitée en nb de px
    taille=1e-3/2/2**(n_groupe+(n_élement-1)/6) #largeur d'un élément du groupe = dim/15
    M=SerieMire([dim,dim,np.round(dim/15),np.round(dim/3)],rmax=rmax,rmin=rmin)
    M=scim.shift(M,[0,-dim/5])+scim.shift(scim.rotate(M,90),[0,dim/5])
    x,y=1e6*np.linspace(-15*taille/2,15*taille/2,dim),1e6*np.linspace(-15*taille/2,15*taille/2,dim) #vecteur taille en um
    return x,y,M

def GroupeMire(n_groupe): #numéro du groupe de la mire
    taille = [1e-3/2/2**(n_groupe+(i-1)/6) for i in range(0,7)] #les tailles des traits du groupe de la mire

def MireEtoile(Nx,Ny,pos_x,pos_y,a=4.5): #a : angle 
    x,y=np.linspace(-Nx/2+1/2,Nx/2-1/2,Nx),np.linspace(-Ny/2+1,Ny/2,Ny)
    XX,YY=np.meshgrid(x,y)
    RR,Theta=np.sqrt((XX+pos_x)**2+(YY+pos_y)**2),np.arctan2(YY+pos_y,XX+pos_x)
    M=np.zeros_like(XX)
    da=a/360*np.pi
    for i_x in range(len(x)) :
        for i_y in range(len(y)) :
            if (Theta[i_x,i_y]//da)%2==0 and (np.sqrt(Nx**2+Ny**2)/4.5>RR[i_x,i_y]>=np.sqrt(Nx**2+Ny**2)/10 or np.sqrt(Nx**2+Ny**2)/2.5>RR[i_x,i_y]>=np.sqrt(Nx**2+Ny**2)/4 or np.sqrt(Nx**2+Ny**2)/1.5>RR[i_x,i_y]>=np.sqrt(Nx**2+Ny**2)/2.34 or RR[i_x,i_y]>=np.sqrt(Nx**2+Ny**2)):
                M[i_x,i_y]=1
    return M
#M=MireEtoile(200,200,0,0)
#plt.matshow(M)

def Grille(Lx,Ly,D,Phi): #la taille Lx,Ly et la periode R et son
    #une grille ne contenant qu'une fréquence spatiale + échantillonage
    x,y=np.linspace(-Lx/2+1/2,Lx/2-1/2,Lx),np.linspace(-Ly/2+1,Ly/2,Ly)
    XX,YY=np.meshgrid(x,y)
    if np.sin(Phi)==0:
        fx=0
        fy=D/Ly
    elif np.cos(Phi)==0:
        fx=D/Ly
        fy=0
    else :
        fx,fy=D/Lx*np.cos(Phi),D/Ly*np.sin(Phi)
    M=np.sin(2*3.14*XX*fx+2*3.14*YY*fy)>0
    return(M)

"FONCTIONS DE PHASE"

def PhaseRadiale(Lx,Ly,Phi_max): #Phase dépendant de la position au centre
    x,y=np.linspace(-Lx/2+1/2,Lx/2-1/2,Lx),np.linspace(-Ly/2+1,Ly/2,Ly)
    XX,YY=np.meshgrid(x,y)
    M_phi=np.exp(1j*np.sqrt(XX**2+YY**2)*2*Phi_max/Ly)
    # plt.matshow(M)
    # plt.show()
    return(M_phi)
    #M_phi=GeomDMD.PhaseRadiale(1200,1920,2*np.pi)

def PentePhase(Lx,Ly,pente): #Pente de phase dans une direction  #pente=[pente_x,pente_y]
    x,y=np.linspace(-Lx/2+1/2,Lx/2-1/2,Lx),np.linspace(-Ly/2+1,Ly/2,Ly)
    XX,YY=np.meshgrid(x,y)
    M_phi=np.exp(1j*(XX*pente[0]+YY*pente[1]))
    # plt.matshow(M)
    # plt.show()
    return(M_phi)
    #M_phi=GeomDMD.PentePhase(1200,1920,[0,4*np.pi/1920])

def PentePhaseSeq(Lx,Ly,Phimax,Ntot,nom,ThetaCorr=None):
     # + petit angle approx entre 2 pixels du DMD (en bonne approx) 0.0005 rad on peut se contenter de 0.01 rad ici soit  environ 0.5deg
    dTheta=2*np.pi/Ntot # Pas d'angle pour faire un tour complet en Ntot images
    print('Ntot=',Ntot,' soit un pas d\'angle de', dTheta, ' rad' )#nombre d'images sur nombre total avec le pas d'angle choisi
                                                        # s'il n'y avait pas de saut d'images (à cause de la variable intervalle)
    Stack=[] # Création pile vide pour concaténation
    M=PentePhase(1200,1920,[0,2*Phimax/1920])
    imgSeq=np.asarray(Im2Bp(M)).ravel()
    get_ipython().run_line_magic('matplotlib', 'inline') # graphe en console pour affichage des images en console
    n_im=1
    bar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),progressbar.Bar()],max_value=Ntot).start()
    for i in range(0,Ntot):
        V=PentePhase(1200,1920,[np.sin(i*dTheta)*2*Phimax/1920,np.cos(i*dTheta)*2*Phimax/1920])#Création tableau tourné V=GeomDMD.RotationPoint(M,0,0,0.05*180/np.pi,12)
        Stack.append(V)
        imgSeq=np.concatenate([imgSeq,Im2Bp(V)])#imgSeq=np.concatenate(imgSeq,[V.ravel()]),np.asarray(GeomDMD.Im2Bp(V.astype(int)))
        print("Iteration numero ",i+1)
        plt.matshow(np.real(V))
        plt.show()
        plt.matshow(np.imag(V))
        plt.show()
        n_im=n_im+1
        bar += 1
    get_ipython().run_line_magic('matplotlib', 'qt5') #graphe en fenêtre
    # Allocate the onboard memory for the image sequence
    #imgSeq=imgSeq.astype(np.int32)
    #Sauvegarde du fichier
    outfile= open(nom+'Seq', 'wb')
    pickle.Pickler(outfile).dump(imgSeq)
    outfile.close()
    outfile= open(nom+'Stack', 'wb')
    pickle.Pickler(outfile).dump(Stack)
    outfile.close()
    print("Fichier sauvegarde")
    print("Nombre d'images=",n_im)
    # Send the image sequence as a 1D list/array/numpy array
    # GeomDMD.PentePhaseSeq(1200,1920,2*np.pi,250,'PentePhase',ThetaCorr=None)

"FONCTIONS DE CONVERSION/FORMATAGE"

def Im2Bp( imgArray, bitShift = 0): # de ALP4.src.ALP4 légéremment modifiée
    '''
    Creates a bit plane from the imgArray.
    The bit plane is an (nSizeX x nSizeY / 8) array containing only the bit values
    corresponding to the bit number bitShift.
    For a bit depth = 8, 8 bit planes can be extracted from the imgArray bu iterating ImgToBitPlane.

    WARNING: This function is slow. It is advised not to use it in a loop to convert a sequence
    of image arrays to bit planes. Use for tests only. It is recommended to directly generate images
    as bitplanes.

    Usage:

    ImgToBitPlane(imgArray,bitShift = 0)

    PARAMETERS
    ----------

    imgArray: 2D array or list corresponding yto the image
              An image of the same resolution as the DMD (nSizeX by nSizeY).

    bitShift: int, optional
              Bit plane to extract form the imgArray (0 to 8),
              Has to be <= bit depth.

    RETURNS
    -------

    bitPlane: list
              Array (nSizeX x nSizeY)/8


         '''
    [SizeY,SizeX]=np.shape(imgArray)
    bitPlane = [0]*(SizeX*SizeY//8) #self.nSizeX*self.nSizeY=1920*1200 ici
    for ind,value in enumerate(imgArray.ravel()): #on crée un  vecteur 1D avec .ravel()
        bitPlane[(ind-ind%8)//8] += (2**(7-ind%8))*((int(value)>>bitShift)%2)
    return bitPlane
 
"FONCTIONS D'AFFICHAGE"

#def AfficheurStack(Stack):

#M=MotifAntiCible(1200,1960,540,500,60)
#M=MotifCible(1200,1920,1000,960,20)
#M=MotifSecteur(1200,1920,600,1)
#M=MotifCible(1200,1920,900,700,100)
#M=MotifRectangle(1200,1920,1000,50);