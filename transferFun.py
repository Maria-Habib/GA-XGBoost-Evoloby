# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:50:06 2019

@author: Maria
"""
import math
import random

def transferFun(pos,vel,fid):
    if fid==1:
        s=1./(1+math.exp(-2.*vel))             #S1
        
    if fid==2:
        s=1/(1+math.exp(-vel))                 #S2
        
    if fid==3:
       s=1/(1+math.exp(-vel/2))                #S3
    
    if fid==4:
        s=1/(1+math.exp(-vel/3))               #S4 
    
    if fid==5:
        s=math.fabs(math.erf(((math.sqrt(math.pi)/2)*vel)))          #V1
        
    if fid==6:
        s=math.fabs(math.tanh(vel))                                 #V2
          
    if fid==7:
        s=math.fabs(vel/math.sqrt((1+math.pow(vel,2))))                       #V3
        
    if fid==8:
        s=math.fabs((2/math.pi)*math.atan((math.pi/2)*vel))        #V4
        
       
    if fid<=4 and fid>=1 :
        r=random.randint(0,1)
        if r<s:
            posOut=1;
        else:
            posOut=0;
            
    if fid<=8 and fid>4 :
        r=random.randint(0,1)
        if r<s:
            posOut=not pos;
        else:
            posOut=pos;
            
    return posOut
                
            
     