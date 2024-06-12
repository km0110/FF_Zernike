#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from adjoint import create_starting_optic
font = {'size'   : 26}

plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (16,9)
plt.rcParams['lines.linewidth'] = 3


# In[26]:


def spikes(mirror):
    ddfmr=np.diff(mirror,n=2)
    spikes=np.where(ddfmr<-4.5e-7)[0]+1
    return spikes


# In[27]:


def shift(mirror,spikes):
    new_mir=np.copy(mirror)
    n=len(spikes) # number of shifts
    spikes=np.append(spikes,len(mirror))
    for i in range(n):
        start=spikes[i]
        end=spikes[i+1]
        new_mir[start:end]-=((i+1)*5e-7)
    return new_mir


# # Total Change in mirror

# In[2]:


def plot_total_change(path,r,w0,gr,lr,wl,steps,xlim,spikes=None):
    plt.scatter(r/w0,-1*np.sum(gr,axis=0)*lr/wl,label='After %d Steps' %(steps))
    # plt.xlim((0,2*w0))
    plt.title('Total change in the mirror')
    plt.xlabel(r'$r/w_0$')
    plt.ylabel(r'$\sum\delta z/\lambda$')
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    # plt.axvline(w0,label='$w_0$',linestyle='--',c='k')
    plt.axhline(0,c='k',linewidth=1)
    plt.xlim(xlim)
    if spikes != None:
        for s in r[spikes]:
            plt.axvline(s/w0,linestyle='--',c='k',linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path+'/total_change')
    plt.close()
    # plt.savefig('f-0_01_total_change_%d_steps_full'%(i))


# In[3]:
def plot_fom_mu(path,fom,mu):
    # Sample data
    
    # Create the first plot with the left y-axis
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel(r'$\mu$', color=color)
    ax1.plot(mu, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create the second plot with the right y-axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Figure of Merit', color=color)  
    ax2.plot(fom/fom[0], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Show the plot
    plt.savefig(path+'/fom_mu')
    plt.close()

def plot_integrals(path,prsq,prpt,ptsq):
    plt.plot(prsq*ptsq,label=r'$\langle\phi_R^2\rangle \langle\phi_T^2\rangle$')
    plt.plot(prpt*prpt,label=r'$\langle\phi_R\phi_T\rangle^2$')
    plt.xlabel('Iterations')
    plt.legend()
    plt.savefig(path+'/prpt')
    plt.close()

def plot_max_grad(path,gr):
    plt.plot(np.max(np.abs(gr),axis=1))
    # print(np.min(np.max(np.abs(gr),axis=1)))
    plt.xlabel('Iteration')
    plt.ylabel(r'$\max [\frac{dF}{dz_m}]$')
    # plt.axvline(30,color='k',linestyle='--',linewidth=2,label='Step: 30')
    # plt.axvline(300,color='k',linestyle='--',linewidth=2)
    plt.savefig(path+"/max_gr")
    plt.close()
    # plt.savefig('max_grad_0-30')

    # plt.show()


# # Compare $\phi_T$ to $\phi_R$ for each step

# In[4]:


def plot_phi_t_phi_r_f(path,phr_f,r,w0,ftf_t,fl):
    for i,ref in enumerate(phr_f):
        if i%50==0: print('PRPT_F step: %d'%i)
        plt.figure(figsize=(11.52,6.56),dpi=100)
        plt.plot(r/w0,np.abs(ftf_t[:,fl]),label=r'$\phi_T(z=f)$')

        plt.plot(r/w0,np.abs(ref),label=r'$\phi_R(z=f)$')
        plt.title('Step: %d' %i)
        plt.xlim((0,1))
        plt.ylabel(r'$\phi$')
        plt.xlabel(r'$r/w_0$')
        plt.legend()
    #     plt.show()
        plt.tight_layout()
        plt.savefig(path+'/phr_f/phr_f_%d'%i,dpi=100)
        plt.close()

# In[6]:


def plot_phi_t_phi_r_OA(path,phr_oa,z,f,ftf_t):
    for i,ref in enumerate(phr_oa):
        if i%50==0: print('PRPT_OA step: %d'%i)
        plt.figure(figsize=(11.52,6.56),dpi=100)
        plt.plot(z/f,np.abs(ftf_t[0,:]),label=r'$\phi_T$')

        plt.plot(z/f,np.abs(ref),label=r'$\phi_R$')
        plt.title('Step: %d' %i)
        plt.ylabel('On-axis amplitude')
        plt.xlabel(r'$z/f$')
        plt.legend()
    #     plt.show()
        plt.tight_layout()
        plt.savefig(path+'/pr_oa/pr_oa_%d'%i,dpi=100)
        plt.close()


# # Show each step size 

# In[37]:

def plot_all(path,lr,gr,r,w0,wl):
    

def plot_dz(path,lr,gr,r,w0,wl):
    delta_zm=-1*lr*np.array(gr)
    for i,dzm in enumerate(delta_zm):
        if i%50==0: print('G step: %d'%i)

        plt.figure(figsize=(11.52,6.56),dpi=100)
        plt.plot(r/w0,dzm/wl,label='step %d'%i)
        #plt.plot(r,np.real(ref),label='real')
        plt.title('Step: %d' %i)

        plt.ylabel(r'$\delta z/\lambda$ ')
        plt.xlabel(r'$r/w_0$')
        plt.axhline(0,linestyle='--',c='k')
        plt.xlim((0,2))
        plt.ylim((np.min(delta_zm)/wl,np.max(delta_zm)/wl))
    #     plt.show()
        plt.savefig(path+'/g/g_%d'%(i),dpi=100)
        plt.close()

def plot_s(path,s,z,f):
    for i,ref in enumerate(s):
        if i%50==0: print('S step: %d'%i)
        plt.figure(figsize=(11.52,6.56),dpi=100)
        plt.plot(z/f,np.abs(ref))
        plt.title('Step: %d' %i)
        plt.ylabel(r'$S=\phi_R-\mu\phi_T$')
        plt.xlabel(r'$z/f$')
    #     plt.show()
        plt.tight_layout()
        plt.savefig(path+'/s/s_%d'%i,dpi=100)
        plt.close()

# # Show Target, Initial, and Final Mirror

# In[8]:


def plot_mirror_comp(path,r,w0,z_m,mr,steps,conic,xlim=(0,2),ylim=(0,3e-5),do_shift=False):
    plt.plot(r/w0,z_m,label='Target Mirror')
    plt.plot(r/w0,mr[0],label='Initial Mirror')
    # me=np.loadtxt('exp_mr.csv')
    # plt.plot(r,me,label='exp',linestyle='--',linewidth=3)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.xlabel(r'$r/w_0$')
    plt.ylabel('z (m)')
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))

    sp=spikes(mr[-1])
    if do_shift:
        smr=shift(mr[-1],sp)
        name='/mirror_comp_shifted'
    else:
        smr=mr[-1]
        name='/mirror_comp'
    plt.plot(r/w0,smr,label='After %d Steps' % (steps),linestyle='--',linewidth=4)

    for s in r[sp]:
        plt.axvline(s/w0,linestyle='--',c='k',linewidth=2)

    plt.tight_layout()
    plt.savefig(path+name)
    plt.close()


# In[10]:


def plot_mirror_be_af(path,r,w0,mr,xlim=(0,2),ylim=(0,3e-5)):
    plt.plot(r/w0,mr[-1],label='Before shift', linewidth=3)
    sp=spikes(mr[-1])
    smr=shift(mr[-1],sp)
    plt.plot(r/w0,smr,label='After shift',linestyle='--',linewidth=3)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.xlabel(r'$r/w_0$')
    plt.ylabel('z (m)')
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))

    sp=spikes(mr[-1])
    for s in r[sp]:
        plt.axvline(s/w0,linestyle='--',c='k',linewidth=2)

    plt.tight_layout()
    # plt.show()
    plt.savefig(path+'/mirror_be_af')
    plt.close()

