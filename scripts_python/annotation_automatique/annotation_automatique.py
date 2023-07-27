import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt 
from math import *
from skimage import color
import os
from shapely.geometry import Point, Polygon #pip install shapely
from multiprocessing import Process, Value, Array, Pool

#path = 'C:/Users/edou1/Desktop/shipmark/annotation_automatique' #chemin du projet : A MODIFIER POUR CHAQUE PERSONNE
path = os.path.dirname(__file__)

masks = pd.read_csv(path + "/input/segmentation_csv/train_ship_segmentations.csv")

ID = masks['ImageId']
mask_without_duplicate = pd.read_csv(path + "/input/segmentation_csv/train_ship_segmentations.csv")
mask_without_duplicate.drop_duplicates(subset=['ImageId'], keep="first", inplace=True)
mask_without_duplicate.dropna(inplace=True)

serie = list(range(0,len(mask_without_duplicate),1))
mask_without_duplicate['index'] = serie
mask_without_duplicate.set_index('index', inplace=True)
ID = mask_without_duplicate['ImageId']


EncodedPixels = mask_without_duplicate['EncodedPixels']
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
#---------------------------------------------------------------#
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape).T  # Needed to align to RLE direction
#---------------------------------------------------------------#


#--------------------------trouver les diagonales du rectangle et les enlever de la lsite-------------------------------#
def get_cotes(Param_1, Param_2, Param_3, Param_4, DEBUG):
    A = Param_1
    B = Param_2
    C = Param_3
    D = Param_4
    #Top = A   Bottom = B    Right = C    Left = D
        #On va tester si les droites se croisent pour obtenir les diagonales du rectangle "bateau"
    #Equation Top,Bottom    y=mx+p
    m1 = 0
    if((B[0]-A[0])!=0):
        m1 = (B[1]-A[1])/(B[0]-A[0]) #m=(yB−yA)/(xB−xA)
    p1 = A[1] - m1*A[0] #p=yA−mxA
    #Equation Right,Left
    m2 = 0
    if((D[0]-C[0])!=0):
        m2 = (D[1]-C[1])/(D[0]-C[0]) #m=(yB−yA)/(xB−xA)
    p2 = C[1] - m2*C[0] #p=yA−mxA
    #Les droites se croisent-elles ?
    x1 = 0
    if((m1 - m2)!=0):
        x1 = (p2 - p1) / (m1 - m2)
    #print("x1 = ", x1)

    #Equation Top,Right
    m1 = 0
    if((C[0]-A[0])!=0):
        m1 = (C[1]-A[1])/(C[0]-A[0]) #m=(yB−yA)/(xB−xA)
    p1 = A[1] - m2*A[0] #p=yA−mxA
    #Equation Bottom,Left
    m2 = 0
    if((D[0]-B[0])!=0):
        m2 = (D[1]-B[1])/(D[0]-B[0]) #m=(yB−yA)/(xB−xA)
    p2 = B[1] - m2*B[0] #p=yA−mxA
    #Les droites se croisent-elles ?
    x2 = 0
    if((m1 - m2)!=0):
        x2 = (p2 - p1) / (m1 - m2)
    #print("x2 = ", x2)

    #Equation Top,Left
    m1 = 0
    if((D[0]-A[0])!=0):
        m1 = (D[1]-A[1])/(D[0]-A[0]) #m=(yB−yA)/(xB−xA)
    p1 = A[1] - m2*A[0] #p=yA−mxA
    #Equation Bottom,Right
    m2 = 0
    if((C[0]-B[0])!=0):
        m2 = (C[1]-B[1])/(C[0]-B[0]) #m=(yB−yA)/(xB−xA)
    p2 = B[1] - m2*B[0] #p=yA−mxA
    #Les droites se croisent-elles ?
    x3 = 0
    if((m1 - m2)!=0):
        x3 = (p2 - p1) / (m1 - m2)
    #print("x3 = ", x3)

    liste_Contours_coord = [A,C, A,D, B,C, B,D]
    if((x1 >= A[0]) & (x1 <= B[0])):
        liste_Contours_coord = [A,C, A,D, B,C, B,D]
    if((x2 >= A[0]) & (x2 <= B[0])):
        liste_Contours_coord = [A,B, A,D, B,C, C,D]
    if((x3 >= A[0]) & (x3 <= B[0])):
        liste_Contours_coord = [A,B, A,C, B,C, C,D]
        #PB ICI, il manque des cas

    return liste_Contours_coord
#-----------------------------------------------------------#

#-------------------------trouver les 2 segments les plus longs--------------------------------#
def get_long(liste_Contours_coord):
    long_1 = [0,0]
    long_2 = [0,0]

    D_1 = sqrt((abs(liste_Contours_coord[1][0]-liste_Contours_coord[0][0]))**2 + (abs(liste_Contours_coord[1][1]-liste_Contours_coord[0][1]))**2)
    D_2 = sqrt((abs(liste_Contours_coord[3][0]-liste_Contours_coord[2][0]))**2 + (abs(liste_Contours_coord[3][1]-liste_Contours_coord[2][1]))**2)
    D_3 = sqrt((abs(liste_Contours_coord[5][0]-liste_Contours_coord[4][0]))**2 + (abs(liste_Contours_coord[5][1]-liste_Contours_coord[4][1]))**2)
    D_4 = sqrt((abs(liste_Contours_coord[7][0]-liste_Contours_coord[6][0]))**2 + (abs(liste_Contours_coord[7][1]-liste_Contours_coord[6][1]))**2)
    listeContours = [D_1,D_2,D_3,D_4]

    #Trouve les 2 cotés les plus longs
    length=len(listeContours)
    max1 = 0
    indice_max1 = 0
    i = 0
    #coté 1
    for i in range(length):
        if listeContours[i] >= max1:
            max1 = listeContours[i]
            indice_max1 = i
    long_1[0] = liste_Contours_coord[indice_max1*2]
    long_1[1] = liste_Contours_coord[indice_max1*2+1]
    del listeContours[indice_max1]
    del liste_Contours_coord[indice_max1*2]
    del liste_Contours_coord[indice_max1*2]
    #coté 2
    length=len(listeContours)
    max2 = 0
    indice_max2 = 0
    i = 0
    for i in range(length):
        if listeContours[i] >= max2:
            max2 = listeContours[i]
            indice_max2 = i
    long_2[0] = liste_Contours_coord[indice_max2*2]
    long_2[1] = liste_Contours_coord[indice_max2*2+1]
    del listeContours[indice_max2]
    del liste_Contours_coord[indice_max2*2]
    del liste_Contours_coord[indice_max2*2]

    liste_long = [long_1[0], long_1[1], long_2[0], long_2[1]]

    return liste_long
#-----------------------------------------------------------#

#-------------------------trouver les 2 segments les plus courts---------------------------#
def get_court(liste_Contours_coord_2):
    court_1 = [0,0]
    court_2 = [0,0]

    D_1 = sqrt((abs(liste_Contours_coord_2[1][0]-liste_Contours_coord_2[0][0]))**2 + (abs(liste_Contours_coord_2[1][1]-liste_Contours_coord_2[0][1]))**2)
    D_2 = sqrt((abs(liste_Contours_coord_2[3][0]-liste_Contours_coord_2[2][0]))**2 + (abs(liste_Contours_coord_2[3][1]-liste_Contours_coord_2[2][1]))**2)
    D_3 = sqrt((abs(liste_Contours_coord_2[5][0]-liste_Contours_coord_2[4][0]))**2 + (abs(liste_Contours_coord_2[5][1]-liste_Contours_coord_2[4][1]))**2)
    D_4 = sqrt((abs(liste_Contours_coord_2[7][0]-liste_Contours_coord_2[6][0]))**2 + (abs(liste_Contours_coord_2[7][1]-liste_Contours_coord_2[6][1]))**2)
    listeContours = [D_1,D_2,D_3,D_4]

    #Trouve les 2 cotés les plus longs
    length=len(listeContours)
    min1 = 1000
    indice_min1 = 0
    i = 0
    #coté 1
    for i in range(length):
        if listeContours[i] < min1:
            min1 = listeContours[i]
            indice_min1 = i
    court_1[0] = liste_Contours_coord_2[indice_min1*2]
    court_1[1] = liste_Contours_coord_2[indice_min1*2+1]
    del listeContours[indice_min1]
    del liste_Contours_coord_2[indice_min1*2]
    del liste_Contours_coord_2[indice_min1*2]
    #coté 2
    length=len(listeContours)
    min2 = 0
    indice_min2 = 0
    i = 0
    for i in range(length):
        if listeContours[i] < min2:
            min2 = listeContours[i]
            indice_min2 = i
    court_2[0] = liste_Contours_coord_2[indice_min2*2]
    court_2[1] = liste_Contours_coord_2[indice_min2*2+1]
    del listeContours[indice_min2]
    del liste_Contours_coord_2[indice_min2*2]
    del liste_Contours_coord_2[indice_min2*2]

    liste_court = [court_1[0], court_1[1], court_2[0], court_2[1]]

    return liste_court
#-----------------------------------------------------------#

#----------------------limiter les bordures de l'image------------------------------#
def limit(X_min, Y_min,X_max, Y_max, A):
    if(A[0] < X_min):
        A[0] - X_min
        A[0] = X_min
    if(A[1] < Y_min):
        A[1] = Y_min
    if(A[0] > X_max):
        A[0] = X_max
    if(A[1] > Y_max):
        A[1] = Y_max
    return A
#-----------------------------------------------------------#

#-----------------------verifier q'un coord de pixel est dans une zone-----------------------------#
#--------------------------          renvoi 0 ou 1                ---------------------------------#
def is_within(X, A, B, C, D):
    coords = [A, B, C, D]
    poly = Polygon(coords)

    p1 = Point(X[0], X[1])
    p1.within(poly)     
    
    return p1.within(poly)

#Test exemple 1
X1 = [24.82, 60.24]
Test1 = is_within(X1,[24.89, 60.06], [24.75, 60.06], [24.75, 60.30], [24.89, 60.30] )

#Test exemple 2
X2 = [24.895, 60.05]
Test2 = is_within(X2,[24.89, 60.06], [24.75, 60.06], [24.75, 60.30], [24.89, 60.30] )
def get_pair(A, B, cote_1_C, cote_1_D, cote_2_C, cote_2_D):

    milieu_AB = [((A[0] + B[0])/2), ((A[1] + B[1])/2)]
    milieu_C1 = [((cote_1_C[0] + cote_1_D[0])/2), ((cote_1_C[1] + cote_1_D[1])/2)]
    milieu_C2 = [((cote_2_C[0] + cote_2_D[0])/2), ((cote_2_C[1] + cote_2_D[1])/2)]

    D_1 = sqrt((abs(milieu_AB[0]-milieu_C1[0]))**2 + (abs(milieu_AB[1]-milieu_C1[1]))**2)
    D_2 = sqrt((abs(milieu_AB[0]-milieu_C2[0]))**2 + (abs(milieu_AB[1]-milieu_C2[1]))**2)

    test = 0
    if(D_1 > D_2):
        test = 1
    
    return test
#-----------------------------------------------------------#

#----------------------pour les limites de l'image-----------------------#
def get_MinMax(A, B, C, D):
    list = [A, B, C, D]
    max_X = 0
    min_X = 0
    max_Y = 1000
    min_Y = 1000
    i = 0
    for i in range(len(list)):
        if(list[i][0] > max_X):
            max_X = list[i][0]
        if(list[i][1] > max_Y):
            max_Y = list[i][1]
        if(list[i][0] < min_X):
            min_X = list[i][0]
        if(list[i][1] < min_Y):
            min_Y = list[i][1]

    list_final = [min_X, min_Y, max_X, max_Y]

    return list_final
#-----------------------------------------------------------#

#-----------------------------------------------------------#
def get_Top_Bottom_Right_Left(mask_test, DEBUG):
    left_list = []
    left_list.append([0,999])
    right_list = []
    right_list.append([0,0])
    top_list = []
    top_list.append([999,0])
    bottom_list = []
    bottom_list.append([0,0])
    for i in range(768):
        for j in range(768):
            if (mask_test[i][j]==1):
                if(j<left_list[0][1]):
                    left_list = []
                    left_list.append([i,j])
                else:
                    if(j==left_list[0][1]):
                        left_list.append([i,j])
                if (j>right_list[0][1]):
                    right_list = []
                    right_list.append([i,j])
                else:
                    if(j==right_list[0][1]):
                        right_list.append([i,j])
                if(i>bottom_list[0][0]):
                    bottom_list = []
                    bottom_list.append([i,j])
                else:
                    if(i==bottom_list[0][0]):
                        bottom_list.append([i,j])
                if(i<top_list[0][0]):
                    top_list = []
                    top_list.append([i,j])
                else: 
                    if(i==top_list[0][0]):
                        top_list.append([i,j])

    #top et bottom sont toujours opposé donc si bottom a droite top plus a guache et inversement
    topsum=0
    mintop=[999,999]
    maxtop=[0,0]
    bottomsum=0
    minbottom=[999,999]
    maxbottom=[0,0]
    leftsum=0
    minleft=[999,999]
    maxleft=[0,0]
    rightsum=0
    minright=[999,999]
    maxright=[0,0]
    for i in range(len(top_list)):
        topsum+=top_list[i][1]
        if top_list[i][1]>maxtop[1]:
            maxtop= top_list[i]
        if top_list[i][1]<mintop[1]:
            mintop= top_list[i]
    for i in range(len(bottom_list)):
        bottomsum+=bottom_list[i][1]
        if bottom_list[i][1]>maxbottom[1]:
            maxbottom= bottom_list[i]
        if bottom_list[i][1]<minbottom[1]:
            minbottom= bottom_list[i]
    for i in range(len(left_list)):
        rightsum+=left_list[i][0]
        if left_list[i][0]>maxleft[0]:
            maxleft= left_list[i]
        if left_list[i][0]<minleft[0]:
            minleft= left_list[i]
    for i in range(len(right_list)):
        rightsum+=right_list[i][0]
        if right_list[i][0]>maxright[0]:
            maxright= right_list[i]
        if right_list[i][0]<minright[0]:
            minright= right_list[i]

    topsum=topsum/len(top_list)
    bottomsum=bottomsum/len(bottom_list)
    leftsum=leftsum/len(left_list)
    rightsum=rightsum/len(right_list)

    top=[]
    bottom = []
    right = []
    left =[]

    if topsum-bottomsum<0:
        bottom=maxbottom
        top=mintop
    else: 
        bottom=minbottom
        top=maxtop

    if leftsum-rightsum<0:
        right=maxright
        left=minleft
    else:
        right=minright
        left=maxleft

    return [top, bottom, right, left]
#-----------------------------------------------------------#

#--------------------------- dessine le triangle de direction ------------------------#
def get_triangle(top, bottom, right, left, img, img3, mask_T2, mask_T3, imgTest, ImageId, average, DEBUG):

    liste_Contours_coord = get_cotes(top, bottom, right, left, DEBUG) #On trouve les 4 cotés (on élimine les diagonales du carré)
    liste_long = get_long(liste_Contours_coord) #On trouve les 2 cotés les plus long
    long_1 = [liste_long[0],liste_long[1]] 
    long_2 = [liste_long[2],liste_long[3]]
    court_1 = [liste_Contours_coord[0],liste_Contours_coord[1]]
    court_2 = [liste_Contours_coord[2],liste_Contours_coord[3]]

    #On crée les 4 marqueurs autour du bateau
    constant = 1.5
    A = [(int)(constant*(long_1[0][0]-long_1[1][0]))+long_1[0][0],(int)(constant*(long_1[0][1]-long_1[1][1]))+long_1[0][1]]
    C = [(int)(constant*(long_1[1][0]-long_1[0][0]))+long_1[1][0],(int)(constant*(long_1[1][1]-long_1[0][1]))+long_1[1][1]]
    D = [(int)(constant*(long_2[0][0]-long_2[1][0]))+long_2[0][0],(int)(constant*(long_2[0][1]-long_2[1][1]))+long_2[0][1]]
    B = [(int)(constant*(long_2[1][0]-long_2[0][0]))+long_2[1][0],(int)(constant*(long_2[1][1]-long_2[0][1]))+long_2[1][1]]

    A1 = B
    B1 = A

    A2 = C
    B2 = D

    T1 = court_1[0]
    T2 = court_1[1]
    T3 = court_2[0]
    T4 = court_2[1]
    A1_p = A1
    B1_p = B1

    test_val = get_pair(A1_p,B1_p,T1,T2,T3,T4)

    if(test_val == 0):
        C1 = court_1[0]
        D1 = court_1[1]

        C2 = court_2[0]
        D2 = court_2[1]
    else:
        C1 = court_2[0]
        D1 = court_2[1]

        C2 = court_1[0]
        D2 = court_1[1]

    color1 = [0,0,0]
    color2 = [0,0,0]
    liste_MinMax = get_MinMax(A1,B1,C1,D1)
    Min = [liste_MinMax[0],liste_MinMax[1]]
    Max = [liste_MinMax[2],liste_MinMax[3]]
    i = 0
    cpt_px_vert = 0
    cpt_px_rouge = 0
    for i in range(Min[0], Max[0]):
        for j in range(Min[1], Max[1]):
            if((is_within([i,j],A1,B1,C1,D1) == True) and (i>=0) and (j>=0) and (i<=767) and (j<=767)): #zone verte
                color1 += img3[i,j]
                mask_T3[i,j] = 1
                cpt_px_vert+=1
                imgTest[i,j] = [0, 255, 0]

    liste_MinMax = get_MinMax(A2,B2,C2,D2)
    Min = [liste_MinMax[0],liste_MinMax[1]]
    Max = [liste_MinMax[2],liste_MinMax[3]]           
    for i in range(Min[0], Max[0]):
        for j in range(Min[1], Max[1]):
            if((is_within([i,j],A2,B2,C2,D2) == True) and (i>=0) and (j>=0) and (i<=767) and (j<=767)): #zone rouge
                color2 += img3[i,j]
                mask_T3[i,j] = 1
                cpt_px_rouge+=1
                imgTest[i,j] = [255, 0, 0]

    #On regarde la dose de Bleu + Rouge = mauve (+de mauve = + de trainée)
    couleur_total_1 = 0
    if(cpt_px_vert>0):
        couleur_total_1 = (color1[2]+(color1[0]/4))/cpt_px_vert
    couleur_total_2 = 0
    if(cpt_px_rouge>0):
        couleur_total_2 = (color2[2]+(color2[0]/4))/cpt_px_rouge

    arriere_1 = [0,0]
    arriere_2 = [0,0]


    ####################################################################
    couleur_total_1 = abs(couleur_total_1)
    couleur_total_2 = abs(couleur_total_2)
    valid = False

    diference = 0
    threshold = 1.1 # Robustesse = 64%
    threshold_blue_color_average = 80 # Robustesse = 75%

    if(couleur_total_1 >= couleur_total_2):
        if(couleur_total_2 > 0):
            diference = couleur_total_1 / couleur_total_2
    else:
        if(couleur_total_1 > 0):
            diference = couleur_total_2 / couleur_total_1

    average_np = np.average(img3, axis=(0,1))

    average[0] = average_np.item(0)
    average[1] = average_np.item(1)
    average[2] = average_np.item(2)
    average[0] = int(average[0]*255)
    average[1] = int(average[1]*255)
    average[2] = int(average[2]*255)

    if(diference >= threshold):
        valid = True


    if(average[2] > threshold_blue_color_average):
        valid = False

    ####################################################################



    if(couleur_total_1 < couleur_total_2):
        arriere_1 = C2
        arriere_2 = D2
        liste_MinMax = get_MinMax(C1,D1,C2,D2)
        front = [((C1[0] + D1[0])/2), ((C1[1] + D1[1])/2)]
        Min = [liste_MinMax[0],liste_MinMax[1]]
        Max = [liste_MinMax[2],liste_MinMax[3]]
        i = 0
        for i in range(Min[0], Max[0]):
            for j in range(Min[1], Max[1]):
                if(is_within([i,j],front,front,C2,D2) == True): #zone verte
                    if(valid == True):
                        mask_T2[i,j] = 1
    else:
        arriere_1 = C1
        arriere_2 = D1
        liste_MinMax = get_MinMax(C1,D1,C2,D2)
        front = [((C2[0] + D2[0])/2), ((C2[1] + D2[1])/2)]
        Min = [liste_MinMax[0],liste_MinMax[1]]
        Max = [liste_MinMax[2],liste_MinMax[3]]
        i = 0
        for i in range(Min[0], Max[0]):
            for j in range(Min[1], Max[1]):
                if(is_within([i,j],front,front,C1,D1) == True): #zone verte
                    if(valid == True):
                        mask_T2[i,j] = 1

    
    x = 1
    y = 0

    Bateau_center = [0,0]
    Bateau_center[y] = (top[y] + bottom[y])/2
    Bateau_center[x] = (left[x] + right[x])/2
    Bateau_lx = right[x]-left[x]
    Bateau_ly = bottom[y]-top[y]


    avant = front

    if(avant[y] > (arriere_1[y] and arriere_2[y])):
        if(arriere_1[x] < arriere_2[x]):
            arriere_d = arriere_1
            arriere_g = arriere_2
        else:
            arriere_d = arriere_2
            arriere_g = arriere_1
    elif(avant[y] < (arriere_1[y] and arriere_2[y])): 
        if(arriere_1[x] > arriere_2[x]):
            arriere_d = arriere_1
            arriere_g = arriere_2
        else:
            arriere_d = arriere_2
            arriere_g = arriere_1
    elif(avant[x] > arriere_1[x]): 
        if(arriere_1[y] > arriere_2[y]):
            arriere_d = arriere_1
            arriere_g = arriere_2
        else:
            arriere_d = arriere_2
            arriere_g = arriere_1
    else: 
        if(arriere_1[y] < arriere_2[y]):
            arriere_d = arriere_1
            arriere_g = arriere_2
        else:
            arriere_d = arriere_2
            arriere_g = arriere_1


    # Determine si les résultats sont valides ou non, grâce au seuil de confiance
    ##################################################################################
    if(valid == True):
        write_file(Bateau_center, Bateau_lx, Bateau_ly, avant, arriere_g, arriere_d, imgTest, ImageId, DEBUG)
        print("File : " + ImageId + " Valid")
    else:
        print("File : " + ImageId + " NOT Valid")
    
    return valid
    #################################################################################

#-----------------------------------------------------------#

#-------------------------- algo -------------------------#
def algo(ImageId, DEBUG):

    img = imread(path + "/input/dataset/" + ImageId)
    all_m = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels']
    number_of_boats = all_m.size
    print("File : "+ ImageId +" |  Number of Boats : ", number_of_boats)

    if(number_of_boats > 0):

        test = np.array(img)

        imgTest = imread(path + "/input/dataset/" + ImageId)
        img1 = color.rgb2gray( img )
        img2 = color.rgb2lab( img )
        img3 = color.rgb2hsv( img )

        # Take the individual ship masks and create a single mask array for Only one ship
        mask_T2 = np.zeros((768, 768))
        mask_T3 = np.zeros((768, 768))
        mask_test = np.zeros((768, 768))
        mask_test += rle_decode(all_m.iloc[0])

        valid = False
        average = [0,0,0]

        for i in range(number_of_boats):
            mask_test = np.zeros((768, 768))
            mask_test += rle_decode(all_m.iloc[i])
            new_list = get_Top_Bottom_Right_Left(mask_test, DEBUG)
            valid_Function = get_triangle(new_list[0], new_list[1], new_list[2], new_list[3], img, img3, mask_T2, mask_T3, imgTest, ImageId, average, DEBUG)
            if(valid_Function == True):
                valid = True

        all_masks = np.zeros((768, 768))

        d = np.angle(all_masks)

        fig, axarr = plt.subplots(2, 2, figsize=(11, 11))
        axarr[0][0].axis('off')
        axarr[0][1].axis('off')
        axarr[1][0].axis('off')
        axarr[1][1].axis('off')
        axarr[0][0].imshow(test)
        axarr[0][1].imshow(test)
        axarr[0][1].imshow(mask_T2, alpha=0.8)
        axarr[1][0].imshow(img3)
        axarr[1][1].imshow(imgTest)

        plt.tight_layout(h_pad=0.1, w_pad=0.15)

        #string_val = "_" + str(average[0]) + "_" + str(average[1]) + "_" + str(average[2]) + ".jpg"
        ImageId = ImageId[0:-4]

        if(DEBUG == 1):
            if(valid == True):
                fig.savefig(path + "/output/images_VALID/" + ImageId)
            else:
                fig.savefig(path + "/output/images_NOT_VALID/" + ImageId)

    return 0
#-----------------------------------------------------------#

#----------------------------- Only draw the final DEBUG image and the final .txt annotation file -----------------------------#
def write_file(bateau, bateau_lx, bateau_ly, avant, arriere_g, arriere_d, imgTest, ImageId, DEBUG):
    marge = 6
    bateau_lx += marge
    bateau_ly += marge
    x = 1
    y = 0

    coin_Top_Left = [bateau[x]-(bateau_lx/2), bateau[y]-(bateau_ly/2)]
    coin_Bottom_Right = [bateau[x]+(bateau_lx/2), bateau[y]+(bateau_ly/2)]

    x=0
    y=1
    #imgTest[y,x] : (l'image marche à l'envers)

    #Trace Rectangle LX
    for i in range(bateau_lx):
        if((coin_Top_Left[1]) >= 0 and (coin_Top_Left[1]) <= 767 and (coin_Top_Left[0]+i) >= 0 and (coin_Top_Left[0]+i) <= 767):
            imgTest[int(coin_Top_Left[1]), int(coin_Top_Left[0]+i)] = [255, 0, 255]
        if((coin_Bottom_Right[1]) >= 0 and (coin_Bottom_Right[1]) <= 767 and (coin_Bottom_Right[0]-i) >= 0 and (coin_Bottom_Right[0]-i) <= 767):
            imgTest[int(coin_Bottom_Right[1]), int(coin_Bottom_Right[0]-i)] = [255, 0, 255]

    #Trace Rectangle LY
    for i in range(bateau_ly):
        if((coin_Top_Left[1]+i) >= 0 and (coin_Top_Left[1]+i) <= 767 and (coin_Top_Left[0]) >= 0 and (coin_Top_Left[0]) <= 767):
            imgTest[int(coin_Top_Left[1]+i), int(coin_Top_Left[0])] = [255, 0, 255]
        if((coin_Bottom_Right[1]-i) >= 0 and (coin_Bottom_Right[1]-i) <= 767 and (coin_Bottom_Right[0]) >= 0 and (coin_Bottom_Right[0]) <= 767):
            imgTest[int(coin_Bottom_Right[1]-i), int(coin_Bottom_Right[0])] = [255, 0, 255]


    #Trace les points
    for i in range(10):
        #Tracer centre
        if((bateau[x]+i-5) >= 0 and (bateau[x]+i-5) <= 767):
            imgTest[int(bateau[x]+i-5), int(bateau[y])] = [255, 0, 255]
        if((bateau[y]+i-5) >= 0 and (bateau[y]+i-5) <= 767):
            imgTest[int(bateau[x]), int(bateau[y]+i-5)] = [255, 0, 255]

        #Tracer coin carré Haut
        if((coin_Top_Left[1]+i-5) >= 0 and (coin_Top_Left[1]+i-5) <= 767 and (coin_Top_Left[0]) >= 0 and (coin_Top_Left[0]) <= 767):
            imgTest[int(coin_Top_Left[1]+i-5), int(coin_Top_Left[0])] = [255, 0, 255]
        if((coin_Top_Left[0]+i-5) >= 0 and (coin_Top_Left[0]+i-5) <= 767 and (coin_Top_Left[1]) >= 0 and (coin_Top_Left[1]) <= 767):
            imgTest[int(coin_Top_Left[1]), int(coin_Top_Left[0]+i-5)] = [255, 0, 255]

        #Tracer coin carré Bas
        if((coin_Bottom_Right[1]+i-5) >= 0 and (coin_Bottom_Right[1]+i-5) <= 767 and (coin_Bottom_Right[0]) >= 0 and (coin_Bottom_Right[0]) <= 767):
            imgTest[int(coin_Bottom_Right[1]+i-5), int(coin_Bottom_Right[0])] = [255, 0, 255]
        if((coin_Bottom_Right[0]+i-5) >= 0 and (coin_Bottom_Right[0]+i-5) <= 767 and (coin_Bottom_Right[1]) >= 0 and (coin_Bottom_Right[1]) <= 767):
            imgTest[int(coin_Bottom_Right[1]), int(coin_Bottom_Right[0]+i-5)] = [255, 0, 255]

        #Tracer proue
        if((avant[x]+i-5) >= 0 and (avant[x]+i-5) <= 767):
            imgTest[int(avant[x]+i-5), int(avant[y])] = [255, 0, 255]
        if((avant[y]+i-5) >= 0 and (avant[y]+i-5) <= 767):
            imgTest[int(avant[x]), int(avant[y]+i-5)] = [255, 0, 255]

        #Tracer poupe droit
        if((arriere_d[x]+i-5) >= 0 and (arriere_d[x]+i-5) <= 767):
            imgTest[int(arriere_d[x]+i-5), int(arriere_d[y])] = [255, 255, 0]
        if((arriere_d[y]+i-5) >= 0 and (arriere_d[y]+i-5) <= 767):
            imgTest[int(arriere_d[x]), int(arriere_d[y]+i-5)] = [255, 255, 0]

        #Tracer poupe gauche
        if((arriere_g[x]+i-5) >= 0 and (arriere_g[x]+i-5) <= 767):
            imgTest[int(arriere_g[x]+i-5), int(arriere_g[y])] = [0, 255, 255]
        if((arriere_g[y]+i-5) >= 0 and (arriere_g[y]+i-5) <= 767):
            imgTest[int(arriere_g[x]), int(arriere_g[y]+i-5)] = [0, 255, 255]

    x = 1
    y = 0

    str_ImageId = ImageId[0:-4]

    f = open(path + "/output/annotations/" + str_ImageId + ".txt", "a")

    #Carré de bateau (centre + lx + ly)
    #N_x = bateau[x]/767
    #N_y = bateau[y]/767
    #N_lx = bateau_lx/767
    #N_ly = bateau_ly/767
    #f.write("1 " + str(N_x) + " " + str(N_y) + " " + str(N_lx) + " " + str(N_ly))
    #f.write("\n")

    #Avant
    N_x = avant[x]/767
    N_y = avant[y]/767
    f.write("0 " + str(N_x) + " " + str(N_y))
    f.write("\n")

    #Arrière Gauche
    N_x = arriere_g[x]/767
    N_y = arriere_g[y]/767
    f.write("3 " + str(N_x) + " " + str(N_y))
    f.write("\n")

    #Arrière Droit
    N_x = arriere_d[x]/767
    N_y = arriere_d[y]/767
    f.write("2 " + str(N_x) + " " + str(N_y))
    f.write("\n")

    f.close()

    return 0
#-------------------------------------------------------------------------------------------------------------------------#


#----------- run the algorithme for each image and verify if the fil exist in the directory ------------------#
def run(file):
    
    # DEBUG = 0 doesn't print the output image | DEBUG = 1 print the output image in the folder (RAM memory consuming)
    DEBUG = 1

    if(os.path.exists(path + "/output/annotations/" + file[0:-4] + ".txt")):
        os.remove(path + "/output/annotations/" + file[0:-4] + ".txt")
    if(os.path.exists(path + "/output/images_VALID/" + file[0:-4] + ".jpg")):
        os.remove(path + "/output/images_VALID/" + file[0:-4] + ".jpg")
    if(os.path.exists(path + "/output/images_NOT_VALID/" + file[0:-4] + ".jpg")):
        os.remove(path + "/output/images_NOT_VALID/" + file[0:-4] + ".jpg")
 
    if(os.path.exists(path + "/input/dataset/" + file[0:-4] + ".jpg")):
        algo(file, DEBUG)

    return 0
#-----------------------------------------------------------------------------------------------------------#


#--------------------- return all the files from a directory ----------------------------#
def for_all_directory():
    dir = 'run'
    total_files = []

    all_files = os.listdir(path  + "/input/" + dir + "/")

    for file in all_files:
        if(os.path.exists(path  + "/input/dataset/" + file[0:-4] + ".jpg")):
            total_files.append(file[0:-4] + ".jpg")

    return total_files
#---------------------------------------------------------------------------------------#




# Run the program for all the images in the directory named "run"
total_files = for_all_directory()

# Number of processes used in parallel
Number_Of_Processes = 10

if __name__ == '__main__':
    pool = Pool(Number_Of_Processes)
    with pool:
        print("Path = "+ path)
        print("\nAlgorithme lancé pour",len(total_files), "fichiers\n")
        pool.map(run,total_files) # Run the algorithm for each image at a time
        pool.close()
        pool.join()
        print("\nFINISHED Successfully\n")
