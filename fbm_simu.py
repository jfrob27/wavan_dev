import numpy as np
from wavan import fan_trans
from noisegen import fbm2d
import matplotlib.pyplot as plt


def powerlawmod(wt, wtC, tab_k,  wherestart, slope,):   
    
    
    Wc=abs(wtC)
    wtmod=np.zeros((wt.shape[0],wt.shape[1],wt.shape[2]))
    x=np.log(tab_k)
    awt=wtC.copy()
    awt=abs(awt)
    wt=abs(wt)

    power=np.log(np.mean((abs(wt)**2.), axis=(1,2)))
    powernew=np.log(np.mean((abs(Wc)**2.), axis=(1,2)))

    end=wtmod.shape[0]
   
    for i in range(int(wherestart-6),end):
        test=0
        ctest=0
        wtfori=abs(Wc[i,:,:])
       

        difference = slope * ( x[i] - x[wherestart] ) - powernew[i] + power[wherestart]

        constant= np.sqrt(np.exp(difference))
        
        wtmod[i,:,:]=wtfori*constant
     
    return wtmod


def interceptmod(wt, wtC, tab_k,  incr,):   
    
    
    Wc=abs(wtC)
    wtmod=np.zeros((wt.shape[0],wt.shape[1],wt.shape[2]))
    x=np.log(tab_k)
    awt=wtC.copy()
    awt=abs(awt)
    wt=abs(wt)

    power=np.log(np.mean((abs(wt)**2.), axis=(1,2)))
    powernew=np.log(np.mean((abs(Wc)**2.), axis=(1,2)))

    end=wtmod.shape[0]
   
    for i in range(end):
        wtfori=abs(Wc[i,:,:])
       

        difference = incr

        constant= np.sqrt(np.exp(difference))
        
        wtmod[i,:,:]=wtfori*constant
     
    return wtmod
        
def all_images(wt, Wn, Wc, tab_k):
    

    #image = np.load('/home/aparker/pycodes/data/image.npy')
    #wt = np.load('/home/aparker/pycodes/data/wt.npy')
    #Wn = np.load('/home/aparker/pycodes/data/Wn.npy')
    #Wc = np.load('/home/aparker/pycodes/data/Wc.npy')
    #tab_k = np.load('/home/aparker/pycodes/data/tab_k.npy')
    #S1ac = np.load('/home/aparker/pycodes/data/S1ac.npy')
    #S1a = np.load('/home/aparker/pycodes/data/S1a.npy')
    #wt=np.complex128(wt)
    #Wn=np.complex128(Wn)
    #Wc=np.complex128(Wc)
    
    cphase=np.arctan2(Wc.imag, Wc.real)
    nphase=np.arctan2(Wn.imag,Wn.real)

    x=np.log(tab_k)
    power=np.log(np.mean((abs(wt)**2.), axis=(1,2), dtype=np.float64))
    powernew=np.log(np.mean((abs(Wc)**2.), axis=(1,2), dtype=np.float64))
    

    difference = abs(np.nanmean(powernew[15] - power[15]))
    
    incrementsize = difference / 25.
    wtnew=Wn.real.copy()
    Wc=powerlawmod(abs(wt),abs(Wc),tab_k, int(tab_k.shape[0]*1/2) , -3.3)
    Wc=interceptmod(abs(wt), abs(Wc), tab_k, -difference)

    

#Raising Power#
    for i in range(25):
        plt.close()
        f_fig, f_ax = plt.subplots(2, 1, figsize=(9,9))
        f_ax[0].set_xlim(-6,0)
        f_ax[0].set_ylim(-6,14)
        
        
        wtnew=0
        wCmod=0
        wCmod=interceptmod(abs(wt), abs(Wc), tab_k, incrementsize*(i))
        wCmod[np.isnan(wCmod)]=0
        wtnew=Wn.real+abs(wCmod)*np.cos(cphase)
        rec_image= np.sum(wtnew, axis=0)
        y_c=np.log(np.mean((abs(wCmod)**2.), axis=(1,2)))
        
        f_ax[0].plot(x, y_c)
        f_ax[0].plot(x,power)
        f_ax[1].imshow(rec_image.real, interpolation='none', cmap= 'Greys_r',)
        f_ax[0].set_xlim(-6,0)
        f_ax[0].set_ylim(-6,14) 
    
        
       
        f_fig.savefig('/Users/robitaij/postdoc/fil2star/parker/increment_'+str(i)+'.png')
        


#Changing Slopes
    for i in range(25):
        imagecount=i+24
        
        f_fig, f_ax = plt.subplots(2, 1, figsize=(9,9))
        
        frac_i= (i)/25.
        wCmod=0
        wCmod=powerlawmod(abs(wt),abs(Wc),tab_k, int(tab_k.shape[0]*1/2) , -3.3+frac_i)
    
        wCmod[np.isnan(wCmod)]=0
        wtnew=Wn.real+abs(wCmod)*np.cos(cphase)
        rec_image=np.sum(wtnew, axis=0)
        y_c=np.log(np.mean((abs(wCmod)**2.), axis=(1,2)))
        
    
        f_ax[0].plot(x, y_c)
        f_ax[0].plot(x,power)
        f_ax[1].imshow(rec_image.real, interpolation='none', cmap= 'Greys_r')
        f_ax[0].set_xlim(-6,0)
        f_ax[0].set_ylim(-6,14)
       
        f_fig.savefig('/Users/robitaij/postdoc/fil2star/parker/increment_'+str(imagecount)+'.png')
       
'''  
image=fbm2d(-2.8,512,512)
#X0=np.std(image)
#M=1.1
#L=((np.log(1.+0.5*(M**2.)))**0.5)/X0
L=1.5
image=np.exp(L*image)
image=image-image.min()

q = [2.5] * 25

wt, S1a, tab_k, S1ac, q = fan_trans(image, q=q, reso=1, qdyn=True)

M = tab_k.shape[0]

all_images(wt[0:M,:,:], wt[M:2*M,:,:], wt[2*M:3*M,:,:], tab_k)
'''
