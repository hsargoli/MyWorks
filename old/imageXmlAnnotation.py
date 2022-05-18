import matplotlib.pyplot as plt
import numpy as np
import cv2
import xml.etree.ElementTree as et
import glob
from matplotlib import patches

class dataset():
    def __init__(self,path):
        import os
        os.chdir(path)
    def read_images(self):
        import numpy as np
        import glob        
        self.images=[]
        self.idd=[]
        for i in glob.glob('img//*.jpg',recursive=True):
            self.idd.append(int(i[4:-4]))
            a=plt.imread(i)
            self.images.append(np.array(a))
        return self.images
    def read_annot(self):
        self.annot=[]
        self.shape=[]
        self.objects=[]
        self.claz=[]
        self.truncatez=[]
        self.xminz=[]
        self.yminz=[]
        self.xmaxz=[]
        self.ymaxz=[]
        self.bbox=[]
        for i in glob.glob('ann//*.xml',recursive=True):
            parsing=et.parse(i)
            root=parsing.getroot()
            path=root[0].text
            width=root[2][0].text
            height=root[2][1].text
            level=root[2][2].text
            shhh=np.array([int(width),int(height),int(level)])
            self.shape.append(shhh)
            self.annot.append(root)
            obnum=int(root[4].text)
            self.objects.append(int(obnum))
            
            for j in range(5,5+obnum):
                classs=root[j][0].text
                self.claz.append(classs)
                self.truncatez.append(root[j][2].text)
                xm1=root[j][4][0].text
                ym1=root[j][4][1].text
                xm0=root[j][4][2].text
                ym0=root[j][4][3].text
                self.xminz.append(int(xm1))
                self.yminz.append(int(ym1))
                self.xmaxz.append(int(xm0))
                self.ymaxz.append(int(ym0))
                ooo=[j-5,[xm1,ym1,xm0,ym0]]
                self.bbox.append(ooo)
        return self.annot
    def make_dataset(self):
        self.ii=self.read_images()
        self.rawann=self.read_annot()
        self.dicty={'xml':self.rawann,
               'shape':self.shape,
               'objects':self.objects,
               'ob_detail':{'classez':self.claz,'truncated':self.truncatez,
                           'boundbox':{'xmin':self.xminz,'ymin':self.yminz,'xmax':self.xmaxz,'ymax':self.ymaxz}},
               'images':self.ii}    
        #bb=np.concatenate((ii,aa),axis=0)
        return self.dicty
    def bound_boxes(self):
        dic=self.dicty
        self.BX=[]
        for i in range(len(self.xminz)):    
            xmi=data['ob_detail']['boundbox']['xmin'][i]
            ymi=data['ob_detail']['boundbox']['ymin'][i]
            xma=data['ob_detail']['boundbox']['xmax'][i]
            yma=data['ob_detail']['boundbox']['ymax'][i]
            bx=[xmi,ymi,xma,yma]
            self.BX.append(bx)
        return self.bbox
    def bound(self,idd):
        '''__________________________________________________________________'''
        dic=self.dicty
        cer_ann=self.rawann[idd]
        self.cer_obnum=int(cer_ann[4].text)
        self.BX1=[]
        self.pltheight=[]
        self.pltwidth=[]
        self.pltxy=[]
        for j in range(self.cer_obnum):  
            xmi=cer_ann[j+5][4][0].text
            ymi=cer_ann[j+5][4][1].text
            xma=cer_ann[j+5][4][2].text
            yma=cer_ann[j+5][4][3].text
            bx=[xmi,ymi,xma,yma]
            txy=[int(xmi),int(yma)]
            twidth=[int(xma)-int(xmi)]
            theight=[int(yma)-int(ymi)]
            self.pltheight.append(theight)
            self.pltwidth.append(twidth)
            self.pltxy.append(txy)
            self.BX1.append(bx)
        return self.BX1
    def sgx0(self,xml_path,child):
        parsing=et.parse(xml_path)
        root=parsing.getroot()
        for i in range(len(root)):
            if root[i].tag==child:
                tagg=root[i].text
        return tagg            
    def status(self,idd):
        print('\n|--------status--------|\n')
        print('number of images:\t{}'.format(len(self.ii)))
        print('number of instance in each images objects:\t{}'.format(self.objects))
        print('shape of choosed image: \t{}'.format(self.shape[idd]))
        xyminmax=self.bound(idd)
        #self.ii[idd]
        noi=self.objects[idd]
        print('number of instance in image :\t{}'.format(noi))
        print('bound boxes instance:{}'.format(xyminmax))
        cer_image=np.copy(self.ii[idd])
        #plt.imshow(cer_image)  
        #a12=[]
        fig,ax=plt.subplots()
        ax.imshow(cer_image)
        #ax.add_patch(rect)
        plt.show()
        return xyminmax
    def mask(self,idm):
        
        mask_image=np.copy(self.ii[idm])
        aw=self.bound(idm)
        self.xyminmax=self.bound(idm)
        for i in range(len(self.pltxy)):
            rect = patches.Rectangle((int(self.xyminmax[i][0]),int(self.xyminmax[i][1])),
                                     int(self.pltwidth[i][0]),int(self.pltheight[i][0]),
                                     linewidth=2, edgecolor='r',facecolor='none')
            fig,ax=plt.subplots()
            ax.imshow(mask_image)
            ax.add_patch(rect)
            plt.show()
        return self.pltxy,self.pltheight,self.pltwidth
    
