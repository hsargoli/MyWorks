#three or tree
import os
import cv2
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from xml.dom import minidom, Node
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------

class annot_sg():
    '''SG for annotate image SG 
    please call RUN function with your image path
    ==obj[0:3]=xmin,ymin,xmax,ymax \n---obj[:][shape]'''
    def __init__(self):
        global ix,iy,drawing,xx,yy
        self.xmin=[]
        self.xmax=[]
        self.ymin=[]
        self.ymax=[]
        drawing=False
    def click_event(self,event,x,y,flags,params):
        global ix,iy,drawing,xx,yy
        if event==cv2.EVENT_LBUTTONDOWN:
            drawing = True
            print('x min: {}  |  y min: {} '.format(x,y))
            self.xmin.append(x)
            self.ymin.append(y)
            ix,iy = x,y
            cv2.imshow('press any button to close', self.img)
        elif event==cv2.EVENT_LBUTTONUP:
            drawing=False
            print('x max: {}  |  y max: {} '.format(x,y))    
            self.xmax.append(x)
            self.ymax.append(y)
            c1,c2,c3=random.randint(0,255),random.randint(0,255),random.randint(0,200)
            cv2.rectangle(self.img,(ix,iy),(x,y),(c1,c2,c3),thickness=2)
            cv2.imshow('press any button to close',self.img)
    def RUN(self,image_path):
        self.image_path=image_path
        self.img=cv2.imread(self.image_path)                        #   Image PATH
        cv2.imshow('press any button to close', self.img)
        cv2.setMouseCallback('press any button to close', self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        self.xy=np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        print('shape of output: {}'.format(self.xy.shape))
        return self.xy
# -----------------------------------------------------------------    
    def kind(self,object_name,roott,inside):
        '''
        obj_name is the objact to create 
        root : where to obj_name connect to
        inside is the text inside the obj_name
        '''
        root=minidom.Document()
        obj=root.createElement(str(object_name))
        if inside:
            obj.appendChild(root.createTextNode(inside))
        roott.appendChild(obj)
        return obj
# -----------------------------------------------------------------    
    def create_xml(self,save_path):                   
        w,h,c=self.img.shape
        root=minidom.Document()
        image_name=self.image_path[:-4]
        unn       =self.kind('annotion',root,None)
        folder    =self.kind('path',unn,self.image_path)
        file_n    =self.kind('filename',unn,'{}'.format(image_name))
        pic_size  =self.kind('size',unn,None)
        pic_width =self.kind('width',pic_size,'{}'.format(h))
        pic_height=self.kind('height',pic_size,'{}'.format(w))
        dep       =self.kind('depth',pic_size,'{}'.format(c))
        seg       =self.kind('segmented',unn,'0')
        Nobject   =self.kind('objec_number',unn,str(len(self.xmin)))
        cluster_name='cutter'
        for i in np.arange(0,len(self.xmin)):
                obj       =self.kind('object',unn,None)
                name      =self.kind('name',obj,cluster_name)
                pos       =self.kind('pos',obj,'Unspecified')
                truncated =self.kind('truncated',obj,'0')
                difficult =self.kind('difficult',obj,'0')
                bndbox    =self.kind('bndbox',obj,None)
                xmi       =self.kind('xmin',bndbox,str(self.xmin[i]))
                ymi       =self.kind('ymin',bndbox,str(self.ymin[i]))
                xma       =self.kind('xmax',bndbox,str(self.xmax[i]))
                yma       =self.kind('ymax',bndbox,str(self.ymax[i]))
        xml_=root.toprettyxml()
        print(xml_)
        os.chdir(save_path)                                              # save_path
        save_path_file = save_path+image_name+'.xml'                            
        with open(save_path_file, "w") as f: 
                f.write(xml_)  
        print ('process is done')
