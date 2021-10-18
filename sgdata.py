import matplotlib.pyplot as plt
import numpy as np
import cv2
import xml.etree.ElementTree as et
import glob
from matplotlib import patches

class dataset():
    def __init__(self,path):
        ''' change current direction'''
        import os
        os.chdir(path)
        self.make_dataset()
    def read_images(self):
        '''
        read images from img folder and return list of it 
        '''
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
        '''
        read annot from ann folder of directory
        '''
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
    def xret(self):
        return self.xminz,self.yminz,self.xmaxz,self.ymaxz
    def make_dataset(self):
        '''
        make dictionary of annot & shape & classez of images and annotes
        '''
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
        self.show_mask(idd)
        return xyminmax
    def show_mask(self,idsh):
        self.xyminmax=self.bound(idsh)
        mask_image=np.copy(self.ii[idsh])
        for i in range(len(self.pltxy)):
                rect = patches.Rectangle((int(self.xyminmax[i][0]),int(self.xyminmax[i][1])),
                                         int(self.pltwidth[i][0]),int(self.pltheight[i][0]),
                                         linewidth=2, edgecolor='r',facecolor='none')
                fig,ax=plt.subplots()
                ax.imshow(mask_image)
                ax.add_patch(rect)
                plt.show()  
    def mask(self,idm):
        '''
        return xmin ymin for each objects
        and plot bounding box for each instance in image
        
        '''
        self.xyminmax=self.bound(idm)
        mask_image=np.copy(self.ii[idm])
        aw=self.bound(idm)
        
        return self.pltxy,self.pltheight,self.pltwidth
    def get_trains(self):
        return self.ii
    def get_targets(self):
        targets=[]
        for i in range(len(self.pltxy)):
            bbx=[self.pltxy[i],self.pltheight[i],self.pltwidth[i]]
            targets.append(bbx)
        return targets

#_____________
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
    def just_mask(self):
        xy=self.get_allbound()
        imgs=self.ii
        for i in range(len(imgs)):
            cutted=self.cutter(imgs[i],xy)
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
            bx=[int(xmi),int(ymi),int(xma),int(yma)]
            txy=[int(xmi),int(ymi)]
            twidth=[int(xma)-int(xmi)]
            theight=[int(yma)-int(ymi)]
            self.pltheight.append(theight)
            self.pltwidth.append(twidth)
            self.pltxy.append(txy)
            self.BX1.append(bx)
        return self.BX1
    def get_allbound(self):
        self.allbnd=[]
        for i in range(len(self.ii)):
            ww=self.bound(i)
            self.allbnd.append(ww)
        return self.allbnd    
    def get_masks(self):
        allb=self.allbnd
        imgs=self.ii
        masks=[]
        for i in range(len(allb)):
            bndbx=allb[i]
            cerimage=imgs[i]
            for k in bndbx:
                cropp=cerimage[k[1]:k[3],k[0]:k[2],:]
                masks.append(cropp)
        return masks
    
    def expanding(self,pil_img,osize):
        from PIL import Image
        width,height=pil_img.size
        left=(osize-width)/2
        right=(osize-width)/2
        top=(osize-height)/2
        bot=(osize-height)/2
        new_width=int(width+left+right)
        new_height=int(height+top+bot)
        result=Image.new(pil_img.mode,(new_width,new_height),(255,255,255))
        result.paste(pil_img,(int(left),int(top)))
        return result
    def do_expanding(self,size):
        from PIL import Image
        extended=[]
        images0=self.ii
        pil_images=[]
        pad_pil_images=[]
        pad_arr_images=[]
        for i in images0:
            o=Image.fromarray(i)
            pil_images.append(o)
        for i in pil_images:
            z=self.expanding(i,size)
            pad_pil_images.append(z)
        for i in pad_pil_images:
            T=np.array(i)
            pad_arr_images.append(T)
        return pad_arr_images