import cv2
import numpy as np
import math

file_coordonner=[]
file_coordonner1=[]

box_coordinates = []
box2_coordinates = []

# Lecture des images et convertion en niveau de gris 

img1 = cv2.imread('image092.png')
img2 = cv2.imread('image072.png')

grayImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
grayImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# creation de l'image Residu
Residu=np.zeros(grayImg1.shape)

# fonction MSE

def MSE(block1, block2):
   err = np.sum((block1- block2)**2)
   err = err / (block1.shape[0]*block1.shape[1])
   return err

# recherche dicotomique sur un block
def Search_Dict_Block(block_1,img,i,j,step):

    Neighbours=[(i,j),(i-step,j),(i+step,j),(i,j-step),(i,j+step),(i-step,j-step),(i+step,j+step),(i+step,j-step),(i-step,j+step)]
    min=math.inf
    for k in range(9):
        (x,y)=Neighbours[k]
        block_2 = img[int(x):int(x)+16, int(y):int(y)+16]
        if (len(block_1) == len(block_2) and len(block_1[0]) == len(block_2[0])):
            mse = MSE(block_1, block_2)
            if mse < min:
              min=mse
              x=int(x)
              y=int(y)
              block_min=img[x:x+16,y:y+16]
              position_min=(x,y)
    return (min,block_min,position_min)

# recherche dicotomique sur toute l'image
def Recherche_Dict(block,img,i,j):
    k=32
    while k>=1:
        (min,bl,position)=Search_Dict_Block(block,img,i,j,k)
        i=int(position[0])
        j=int(position[1])
        k=k/2
    return min,bl

for i in range (0,grayImg2.shape[0],16):
    for j in range (0,grayImg2.shape[1],16):
        block1 = grayImg1[i:i + 16,j:j + 16]
        min,block_sim=Recherche_Dict(block1,grayImg2,i,j)
        if min>50 :         
            file_coordonner.append((j,i))
            file_coordonner1.append((j,i))
            Residu[i:i+16,j:j+16]=(block1-block_sim)
        
       
    
for i in range (len(file_coordonner)) :
    cv2.rectangle(img2, (file_coordonner[i][0], file_coordonner[i][1]),
                      (file_coordonner[i][0]+16,file_coordonner[i][1]+16), (0, 0, 255), 2) 

    cv2.rectangle(img1, (file_coordonner1[i][0], file_coordonner1[i][1]),
                      (file_coordonner1[i][0]+16,file_coordonner1[i][1]+16), (0, 255, 0), 2)                   
cv2.imwrite('out.png', Residu)
ResiduImg=cv2.imread('out.png')


# affichage
cv2.imshow("Frame_72", img1)
cv2.waitKey(0)
cv2.imshow("Frame_92", img2)
cv2.waitKey(0)
cv2.imshow('Residu',ResiduImg)
cv2.waitKey(0)