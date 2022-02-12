def crop_rois(img,result,which):
 rois=result[0]['rois']
 ymin,xmin,ymax,xmax=rois[which]
 cutter=img[ymin:ymax,xmin:xmax]
 return cutter,xmin,ymin,xmax,ymax
 
def show_rois(img,result):
                import random
                import cv2
                import numpy as np
                #self.img_bgr=cv2.imread(path)
                #self.img_rgb=cv2.cvtColor(self.img_bgr,cv2.COLOR_BGR2RGB)
                rois = result[0]['rois']
                num_instance=len(rois)
                for i in np.arange(0,num_instance):
                    a=rois[i]
                    ymin,xmin,ymax,xmax=a
                    c1,c2,c3=random.randint(50,255),random.randint(100,255),random.randint(0,160)
                    aa=cv2.rectangle(img,(xmin+15,ymin+15),(xmax-15,ymax-15),color=(c1,c2,c3),thickness=2)
                return aa
                
def save_array(self,arr_img):
                from PIL import Image
                im=Image.fromarray(arr_img)  # convert array to image object
                self.save_rois_path='mrcnn.jpg'
                im.save(self.save_rois_path)
                
def gradian_maker(org_image):
    from scipy.signal import convolve2d
    import numpy as np

    Hx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])#, dtype=np.float32)
    Hy = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])#, dtype=np.float32)

    gx=convolve2d(org_image,Hx)
    gy=convolve2d(org_image,Hy)
    gradian=np.sqrt(gx*gx+gy*gy)

    h,v= gradian.shape
    test_image=gradian.reshape(h,v,1)
    alll=np.concatenate((test_image,test_image,test_image),axis=2)
    return alll
   
   
   
def kernel_saaz(path,save_path,kernel):
    names=[]
    for i in glob.glob(path,recursive=True):   #   key
        names.append(i)
    a=cv2.imread(names[0],0)
#2 read images file with their name
    imagess=[]
    for i in names:
        imagess.append(cv2.imread(i,0))
#3 kernels
    
    Hx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Hy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float32)
    sharpen_kernel=np.array([[0,-1,0],
                             [-1,5,-1],
                             [0,-1,0]],np.float32)

    outline_kernel = np.array([[-1,-1,-1],
                             [-1,8,-1],
                             [-1,-1,-1]],np.float32)    
#4 apply the Kernel
    kerneli=[]
    if kernel == 'sharp':        
        for i in np.arange(0,len(imagess)):
            g=convolve2d(imagess[i],sharpen_kernel)
            kerneli.append(g)

    elif  kernel == 'gradian':
        for i in np.arange(0,len(imagess)):
            gx=convolve2d(imagess[i],Hx)
            gy=convolve2d(imagess[i],Hy)
            g=np.sqrt(gx*gx+gy*gy)
            kerneli.append(g)
    elif kernel == 'outline':
        for i in np.arange(0,len(imagess)):
            g=convolve2d(imagess[i],outline_kernel)
            kerneli.append(g)
#5 save the images
    os.chdir(save_path)
    for i in np.arange(0,len(kerneli)):
        cv2.imwrite('%d.jpg'%(i+1),kerneli[i])
        
