# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:54:11 2018

@author: mulla
"""

import numpy as np
import pandas as pd
import copy
import time


#If you run a single instance then you can use e.g. 
#simulation=local_loads([0.5],0.7,1000,0.1,max_seconds=3600) 
#to simulate 1000 linkers at k=0.5, a force sensitivity of 0.7 and stress of 0.1. max_seconds is handy to stop the simulation if it doesn't rupture within a practical timeframe (1h in this case). 
#Afterward, you can find the most important output in simulation.output.

class local_loads():
    def __init__(self,K,Fe,binding_sites,stress,mobile=False,mode='constant',max_steps=np.inf,max_t=np.inf,max_seconds=np.inf,memory=True,memorize_all_bonds=False):
        assert np.size(K)==np.size(Fe)
        try:
            K[0]
        except IndexError or TypeError:
            K=[K]
        try:
            Fe[0]
        except TypeError:
            Fe=[Fe]
        self.K=np.array(K,dtype=float)
        self.Fe=np.array(Fe,dtype=float) 
        self.binding_sites=int(binding_sites)
        if mode=='ramp':
            self.rate=copy.copy(stress)
            self.stress=0.0
        else:
            self.rate=False
            self.stress=float(stress)
            assert mode=='constant'
        if mobile==False:
            self.N=copy.copy(self.binding_sites)
            self.mobile=mobile
        else:
            self.N=copy.copy(mobile)
            self.mobile=int(mobile)
        self.mode=mode
        self.bonds,self.ruptured=self.initialize_bonds()
        self.distances,self.rates=self.initialize_distances(self.bonds)
        L=np.nanmax(self.distances)
        self.output=self.output=pd.DataFrame({'n':[sum(self.bonds)],'t':[0],'L':[L],'Mean rate':np.mean(self.rates[self.bonds]) })
        self.step=0
        self.run_simulation(max_steps=max_steps,max_t=max_t,max_seconds=max_seconds,memory=memory,memorize_all_bonds=memorize_all_bonds)

    def initialize_bonds(self):
        off_i=self.BellEvans(0)
        if self.mobile:
            bonds=np.zeros(self.binding_sites,dtype=bool)
            empty=np.where(np.logical_not(bonds))[0]
            on_eff=1.0
            for i in range(self.mobile):
                if np.random.rand()<on_eff/(on_eff+off_i):
                    test=np.random.randint(np.size(empty))
                    index=empty[test]
                    bonds[index]=True
                    empty=np.where(np.logical_not(bonds))[0]
                    on_eff=float(np.size(empty))/float(self.binding_sites)
        else:
            bonds=np.random.rand(self.binding_sites)<1.0/(1.0+off_i)
        if sum(bonds)==0:
            ruptured=True
        else:
            ruptured=False
        return bonds,ruptured
        
    def initialize_distances(self,bonds):
        n=sum(bonds)
        if n==0:
            distances=np.array([])
        elif n==1:
            distances=np.array(self.binding_sites,dtype=float)
        elif n==2:
            distances=np.array([0.5*self.binding_sites,0.5*self.binding_sites])
        else:
            positions=np.where(bonds)[0]
            length=np.diff(positions)
            distances=np.zeros(n)
            distances[1:-1]=length[0:-1]+length[1:]
            distances[-1]=self.binding_sites-positions[-2]+positions[0]
            distances[0]=self.binding_sites-positions[-1]+positions[1]
            distances=distances/2.0
        distances_vector=np.zeros(self.binding_sites)*np.nan
        distances_vector[bonds]=distances
        rates=self.BellEvans(distances_vector*self.stress*float(self.N)/float(self.binding_sites))
        return distances_vector,rates
     
    def update_distances(self,indices,firstcall=True):
        filled=np.where(self.bonds)[0]
        for index in indices:
            lower=filled<index
            higher=filled>index
            if sum(lower)==0:
                low_index=max(filled[higher])
                low=low_index-self.binding_sites
            else:
                low_index=max(filled[lower])
                low=copy.copy(low_index)
            if sum(higher)==0:
                high_index=min(filled[lower])
                high=high_index+self.binding_sites
            else:
                high_index=min(filled[higher])
                high=copy.copy(high_index)
            if self.bonds[index]:
                self.distances[index]=float(high-low)/2.0
                self.rates[index]=self.BellEvans(self.distances[index]*self.stress*float(self.N)/float(self.binding_sites))
            
            else:
                self.distances[index]=np.nan
                self.rates[index]=np.nan
            if firstcall:
                self.update_distances([low_index,high_index],firstcall=False)

    def run_simulation(self,max_steps=np.inf,max_t=np.inf,max_seconds=np.inf,memory=True,memorize_all_bonds=False):
        assert not(max_steps==np.inf and max_t==np.inf and max_seconds==np.inf)
        start_t=time.clock()
        if max_steps==np.inf:
            steps=int(1E4)
        else:
            steps=int(copy.copy(max_steps))
        step=0
        ns=np.zeros(steps+1)*np.nan
        ts=np.zeros(steps+1)*np.nan
        Ls=np.zeros(steps+1)*np.nan
        rs=np.zeros(steps+1)*np.nan
        ns[0]=self.output['n'].values[-1]
        ts[0]=self.output['t'].values[-1]
        Ls[0]=self.output['L'].values[-1]
        rs[0]=self.output['Mean rate'].values[-1]
        if memorize_all_bonds:
            self.bond_history=np.zeros([len(self.bonds),steps],dtype=bool)
        start_t_simu=copy.copy(self.output['t'].values[-1])
        counter=0
        while memory and step<max_steps and ts[~np.isnan(ts)][-1]<max_t+start_t_simu and not(self.ruptured) and time.clock()-start_t<max_seconds:
            step+=1
            if step-counter>len(ns)-1:
                if memory:
                        
                    try:
                        ns=np.append(ns,np.zeros(steps+1)*np.nan)
                        ts=np.append(ts,np.zeros(steps+1)*np.nan)
                        Ls=np.append(Ls,np.zeros(steps+1)*np.nan)
                        rs=np.append(rs,np.zeros(steps+1)*np.nan)
                    except MemoryError:
                        print('Ran out of memory')
                        memory=False
                else:
                    counter+=len(ns)
                    ns=np.zeros(steps+1)*np.nan
                    ts=np.zeros(steps+1)*np.nan
                    Ls=np.zeros(steps+1)*np.nan
                    rs=np.zeros(steps+1)*np.nan
                    ns[0]=ns[-1]
                    ts[0]=ts[-1]
                    Ls[0]=Ls[-1]
                    rs[0]=rs[-1]
                      
            ns[step-counter],ts[step-counter],Ls[step-counter],rs[step-counter]=self.KMC_step(ns[step-1-counter],ts[step-1-counter])
            if memorize_all_bonds:
                effective_step=int(step-(np.floor(step/steps)*steps))
                self.bond_history[self.bonds,effective_step-1]=True
                self.bond_history[np.logical_not(self.bonds),effective_step-1]=False
        if time.clock()-start_t>max_seconds:
            print('Time block executed')
        if memorize_all_bonds:
            if step>steps:
                print('Too long to track track all bonds')
                first_part=self.bond_history[int(step-(np.floor(step/steps)*steps)):]
                second_part=self.bond_history[:int(step-(np.floor(step/steps)*steps))]
                self.bond_history[int(step-(np.floor(step/steps)*steps)):]=first_part
                self.bond_history[:int(step-(np.floor(step/steps)*steps))]=second_part
            self.bond_history=self.bond_history[:,0:int(step)]
        self.output=self.output.append(pd.DataFrame({'n':ns[~np.isnan(ns)][1:],'t':ts[~np.isnan(ts)][1:],'L':Ls[~np.isnan(Ls)][1:],'Mean rate':rs[~np.isnan(rs)][1:]}),ignore_index=True)
        self.step+=step
        
        
    def KMC_step(self,n,t):
        if self.mobile:
            self.rates[np.logical_not(self.bonds)]=float(self.N-n)/float(self.binding_sites)
        else:
            self.rates[np.logical_not(self.bonds)]=1.0
        
        if any(self.rates==np.inf):
            singularities=np.where(self.rates==np.inf)[0]
            event=singularities[np.random.randint(0,high=len(singularities))]
        else:
            dt=np.random.exponential(1.0/np.sum(self.rates))
            probs=np.cumsum(self.rates/np.sum(self.rates))
            random_number=np.random.rand()
            event=np.where(probs>random_number)[0][0]
            t+=dt
            if self.rate:
                self.stress=t*self.rate
        self.bonds[event]=not(self.bonds[event])
        n+=np.sign(self.bonds[event]-0.5)
        if n==0:
            self.distances=np.array(np.inf)
            self.ruptured=True
            self.rates[self.bonds]=np.inf
            r=np.inf
        elif n==1:
            self.distances[self.bonds]=self.binding_sites
            self.rates[self.bonds]=self.BellEvans(self.stress*self.binding_sites*float(self.N)/float(self.binding_sites))
            r=np.mean(self.rates[self.bonds])
        elif n==2:
            self.distances[self.bonds]=self.binding_sites*0.5
            self.rates[self.bonds]=self.BellEvans(self.stress*self.binding_sites*0.5*float(self.N)/float(self.binding_sites))
            r=np.mean(self.rates[self.bonds])
        else:
            self.update_distances([event])
            r=np.mean(self.rates[self.bonds])
        L=np.nanmax(self.distances)        
        return n,t,L,r

    def BellEvans(self,force):
        Koffs=np.subtract(np.divide(1.0,self.K),1)
        Koff=0
        for i in range(np.size(Koffs)):
            Koff+=Koffs[i]*np.exp(force/(self.Fe[i]))
        return Koff

def ablation(Ks,Fes,binding_sites,stresses,ablations,pre_ablation_t=3,after_ablation_t=3,repeats=1,mobile=False,relative_stress=False):
    output=pd.DataFrame(columns=['K','Fe','binding_sites','stress','Ablation size','Ruptured','repeat','Pre-ruptured','Rupture time'])
    for K in Ks:
        for Fe in Fes:
            if np.size(K)==np.size(Fe):
                for stress in stresses:
                    if relative_stress:
                        stress=stress*np.max(K)
                    for ablation in ablations:
                        for repeat in range(repeats):
                            ab=local_loads(K,Fe,binding_sites,stress,mobile=mobile,max_t=pre_ablation_t)
                            if ab.ruptured:
                                pre_ruptured=True
                                ruptured=np.nan
                                rupture_time=np.nan
                            else:
                                pre_ruptured=False
                                ab.bonds[0:ablation]=False
                                if sum(ab.bonds)==0:
                                    pre_ruptured=True
                                    ruptured=np.nan
                                    rupture_time=np.nan
                                else:
                                    init_time=copy.copy(ab.output['t'].values[-1])
                                    pre_ruptured=False
                                    ab.distances,ab.rates=ab.initialize_distances(ab.bonds)
                                    ab.output=ab.output.append({'L':np.nanmax(ab.distances),'Mean rate':np.mean(ab.rates[ab.bonds]),'n':sum(ab.bonds),'t':ab.output['t'].values[-1]},ignore_index=True)
                                    ab.run_simulation(max_t=after_ablation_t)
                                    if ab.ruptured:
                                        ruptured=True
                                        rupture_time=ab.output['t'].values[-1]-init_time
                                    else:
                                        rupture_time=np.nan
                                        ruptured=False
                            output=output.append({'K':K,'Fe':Fe,'binding_sites':binding_sites,'stress':stress,'Ablation size':ablation,'Ruptured':ruptured,'repeat':repeat,'Pre-ruptured':pre_ruptured,'Rupture time':rupture_time,'mobile':mobile},ignore_index=True)
    return output
        
def vary_locals(Ks,Fes,stresses,ratio_binding_sites,mobiles,repeats=1,max_steps=np.inf,max_t=np.inf,max_seconds_total=np.inf,max_seconds_step=np.inf,deterministic=False,mode='constant'):
    assert not(max_steps==np.inf and max_t==np.inf and max_seconds_step==np.inf and max_seconds_total==np.inf)
    stresses=np.flipud(np.sort(stresses))
    start_t=time.clock()
    if max_seconds_total<max_seconds_step:
        max_seconds_step=max_seconds_total
    output=pd.DataFrame(columns=['K','Fe','binding_site','mobile','stress','repeat','rupture time','mean rate','mean max L','steps','Ruptured','Mean bound number',',min K','max K','min Fe'])
    for K in Ks:
        for Fe in Fes:
            if np.size(K)==np.size(Fe):
                for ratio_binding_site in ratio_binding_sites:
                    for mobile in mobiles:
                        ruptured=True
                        if mobile:
                            binding_site=ratio_binding_site*mobile
                        else:
                            binding_site=copy.copy(ratio_binding_site)
                        for stress in stresses:
                            for repeat in range(repeats):
                                if (ruptured or deterministic) and time.clock()-start_t<max_seconds_total:
                                    sim=local_loads(K,Fe,binding_site,stress,mobile=mobile,mode=mode,max_steps=max_steps,max_t=max_t,max_seconds=max_seconds_step)
                                    steps=copy.copy(sim.step)
                                    ruptured=copy.copy(sim.ruptured)
                                    if sim.ruptured:
                                        tmax=-20
                                        t=sim.output['t'].values[-1]
                                    else:
                                        t=np.nan
                                        tmax=0
                                    elements=np.logical_and(sim.output['t'].values>2,sim.output['t'].values<sim.output['t'].values[-1]+tmax)
                                    if sum(elements)==0:
                                        r=np.nan
                                        L=np.nan
                                        n=np.nan
                                    else:
                                        r=np.mean(sim.output['Mean rate'].values[elements])
                                        L=np.mean(sim.output['L'].values[elements])
                                        n=np.mean(sim.output['n'].values[elements])
                                    output=output.append({'K':K,'Fe':Fe,'binding_site':binding_site,'mobile':mobile,'stress':stress,'repeat':repeat,'rupture time':t,'mean rate':r,'mean max L':L,'steps':steps,'Ruptured':ruptured,'Mean bound number':n,',min K':np.min(K),'max K':np.max(K),'min Fe':np.min(Fe)},ignore_index=True)
    return output
    
def submatsum(data,n,m):
    # return a matrix of shape (n,m)
    bs = data.shape[0]//n,data.shape[1]//m  # blocksize averaged over
    return np.reshape(np.array([np.sum(data[k1*bs[0]:(k1+1)*bs[0],k2*bs[1]:(k2+1)*bs[1]]) for k1 in range(n) for k2 in range(m)]),(n,m))

    