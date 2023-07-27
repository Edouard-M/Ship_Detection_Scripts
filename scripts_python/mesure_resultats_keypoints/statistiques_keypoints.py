from math import *
from struct import *
import copy
import os
import statistics

pair_list = []
pair_list_correct = []

class coord:
    x  = 0.0
    y  = 0.0

class Bateau:
    avant     = coord()
    arriere_g = coord()
    arriere_d = coord()

    def show(self):
        print("avant :  x = ", self.avant.x, " y = ", self.avant.y)
        print("arriere_g :  x = ", self.arriere_g.x, " y = ", self.arriere_g.y)
        print("arriere_d :  x = ", self.arriere_d.x, " y = ", self.arriere_d.y)

class BateauPair:
    Bateau_True = Bateau()
    Bateau_Test = Bateau()



#--------------------- lit tout un fichier ----------------------------#
def read_file(dir, file):
    f = open(path + "/" + dir + "/" + file + ".txt", "r")
    str_file = f.read()
    f.close()

    return str_file
#---------------------------------------------------------------------#

#--------------------- regarde si un bateau est présent en double dans une liste ----------------------------#
def boat_index(boat, bateau_list):
    value = []

    for i in range(len(bateau_list)):
        if(float(boat.avant.x)     == float(bateau_list[i].avant.x)     and float(boat.avant.y)     == float(bateau_list[i].avant.y)     and
           float(boat.arriere_g.x) == float(bateau_list[i].arriere_g.x) and float(boat.arriere_g.y) == float(bateau_list[i].arriere_g.y) and
           float(boat.arriere_d.x) == float(bateau_list[i].arriere_d.x) and float(boat.arriere_d.y) == float(bateau_list[i].arriere_d.y)):
            value.append(i)

    return value
#------------------------------------------------------------------------------------------------------------#

#----------- Remet les bateaux du tableau "test" dans l'ordre des bateaux du tableau "true" -----------#
def bateau_reOrder(Bateaux_true, Bateaux_test):
    NEW_Bateaux_test = []
    D_bateaux = []

    for j in range(len(Bateaux_true)):
        index=0
        D = 0
        D_min = 10000

        for i in range(len(Bateaux_test)):
            #D = sqrt((abs(float(Bateaux_test[i].rectangle.x)-float(Bateaux_true[j].rectangle.x)))**2 + (abs(float(Bateaux_test[i].rectangle.y)-float(Bateaux_true[j].rectangle.y)))**2)
            arrière_x_test = (float(Bateaux_test[i].arriere_g.x)+float(Bateaux_test[i].arriere_d.x))/2     # Milieu Arrière X =   ((x1 + x2)/2) ; ((y1 + y2)/2)
            arrière_y_test = (float(Bateaux_test[i].arriere_g.y)+float(Bateaux_test[i].arriere_d.y))/2     # Milieu Arrière Y =   ((x1 + x2)/2) ; ((y1 + y2)/2)
            milieu_x_test = (float(Bateaux_test[i].avant.x)+arrière_x_test)/2
            milieu_y_test = (float(Bateaux_test[i].avant.y)+arrière_y_test)/2
            arrière_x_true = (float(Bateaux_true[j].arriere_g.x)+float(Bateaux_true[j].arriere_d.x))/2     # Milieu Arrière X =   ((x1 + x2)/2) ; ((y1 + y2)/2)
            arrière_y_true = (float(Bateaux_true[j].arriere_g.y)+float(Bateaux_true[j].arriere_d.y))/2     # Milieu Arrière Y =   ((x1 + x2)/2) ; ((y1 + y2)/2)
            milieu_x_true = (float(Bateaux_true[j].avant.x)+arrière_x_true)/2
            milieu_y_true = (float(Bateaux_true[j].avant.y)+arrière_y_true)/2

            D = sqrt((abs(milieu_x_test-milieu_x_true))**2 + (abs(milieu_y_test-milieu_y_true))**2) #Milieu

            if(D < D_min):
                D_min = D
                index = i

        NEW_Bateaux_test.append(copy.copy(Bateaux_test[index]))

        D_bateaux.append(D_min)


    tab_index = []
    tab_bannis = []
    for i in range(len(NEW_Bateaux_test)):


        if(D_bateaux[i] < limit):
            pair = BateauPair()
            pair.Bateau_Test = NEW_Bateaux_test[i]
            pair.Bateau_True = Bateaux_true[i]
            pair_list.append(pair)

        """
        tab_index = boat_index(NEW_Bateaux_test[i], NEW_Bateaux_test)
        if(len(tab_index) > 1):
            print("DOUBLE")
            D_min = 10000
            index = 0

            for j in range(len(tab_index)):
                D = D_bateaux[tab_index[j]]
                if(D < D_min):
                    D_min = D
                    index = tab_index[j]

            for j in range(len(tab_index)):
                if((index != tab_index[j]) and (tab_bannis.__contains__(tab_index[j]) == False)):
                    tab_bannis.append(tab_index[j])

            if(tab_bannis.__contains__(i) == False):
                #print("Pair : Index", i, "=", NEW_Bateaux_test[i].avant.x)
                pair = BateauPair()
                pair.Bateau_Test = NEW_Bateaux_test[i]
                pair.Bateau_True = Bateaux_true[i]
                pair_list.append(pair)

        else:
            #print("Pair : Index", i, "=", NEW_Bateaux_test[i].avant.x)
            pair = BateauPair()
            pair.Bateau_Test = NEW_Bateaux_test[i]
            pair.Bateau_True = Bateaux_true[i]
            pair_list.append(pair)
        """


    return NEW_Bateaux_test
#----------------------------------------------------------------------------------------------------#

#----------- Erreur en nombre de bateaux -----------#
def image_erreur_bateaux(file):

    bateaux_true = find_bateaux(read_file(dir_true, file[0:-4]))
    bateaux_test = find_bateaux(read_file(dir_true, file[0:-4]))

    erreur = len(bateaux_test)-len(bateaux_true)

    return erreur
#--------------------------------------------------#

#--------------------- lit les bateaux d'un fichier ----------------------------#
def find_bateaux(string):
    bateaux = []
    string_tab = string.split("\n")

    for j in range(int(len(string_tab)/3)):
        bateau = Bateau()
        #bateau.rectangle = coord()
        bateau.avant     = coord()
        bateau.arriere_d = coord()
        bateau.arriere_g = coord()
        for i in range(j*3, ((j*3)+3)):
            string_values = string_tab[i].split(" ")
            if(int(string_values[0]) == 0):
                bateau.avant.x  = string_values[1]
                bateau.avant.y  = string_values[2]
            if(int(string_values[0]) == 2):
                bateau.arriere_d.x  = string_values[1]
                bateau.arriere_d.y  = string_values[2]
            if(int(string_values[0]) == 3):
                bateau.arriere_g.x  = string_values[1]
                bateau.arriere_g.y  = string_values[2]

        if(float(bateau.avant.x) > 0 and float(bateau.avant.x) < 1):
            bateau.avant.x = int(float(bateau.avant.x) * image_size)
        if(float(bateau.avant.y) > 0 and float(bateau.avant.y) < 1):
            bateau.avant.y = int(float(bateau.avant.y) * image_size)
        if(float(bateau.arriere_d.x) > 0 and float(bateau.arriere_d.x) < 1):
            bateau.arriere_d.x = int(float(bateau.arriere_d.x) * image_size)
        if(float(bateau.arriere_d.y) > 0 and float(bateau.arriere_d.y) < 1):
            bateau.arriere_d.y = int(float(bateau.arriere_d.y) * image_size)
        if(float(bateau.arriere_g.x) > 0 and float(bateau.arriere_g.x) < 1):
            bateau.arriere_g.x = int(float(bateau.arriere_g.x) * image_size)
        if(float(bateau.arriere_g.y) > 0 and float(bateau.arriere_g.y) < 1):
            bateau.arriere_g.y = int(float(bateau.arriere_g.y) * image_size)

        if(float(bateau.avant.x) < 0):
            bateau.avant.x = 0
        if(float(bateau.avant.y) < 0):
            bateau.avant.y = 0
        if(float(bateau.arriere_d.x) < 0):
            bateau.arriere_d.x = 0
        if(float(bateau.arriere_d.y) < 0):
            bateau.arriere_d.y = 0
        if(float(bateau.arriere_g.x) < 0):
            bateau.arriere_g.x = 0
        if(float(bateau.arriere_g.y) < 0):
            bateau.arriere_g.y = 0

        if(float(bateau.avant.x) > image_size):
            bateau.avant.x = image_size
        if(float(bateau.avant.y) > image_size):
            bateau.avant.y = image_size
        if(float(bateau.arriere_d.x) > image_size):
            bateau.arriere_d.x = image_size
        if(float(bateau.arriere_d.y) > image_size):
            bateau.arriere_d.y = image_size
        if(float(bateau.arriere_g.x) > image_size):
            bateau.arriere_g.x = image_size
        if(float(bateau.arriere_g.y) > image_size):
            bateau.arriere_g.y = image_size
        bateaux.append(bateau)





    return bateaux
#-------------------------------------------------------------------------------#

#--------------------- lit les tous les fichiers ----------------------------#
def read_all_files():

    all_files = os.listdir(path  + "/" + dir_true + "/")
    erreur_bateaux = 0
    bateaux_true = []
    bateaux_test = []
    total = 0
    nb_file = 0
    nb_empty_file = 0

    #print("\nDossier True =", len(all_files), "fichier(s)")
    for file in all_files:
        if(os.path.exists(path  + "/" + dir_test + "/" + file[0:-4] + ".txt")):
            #print("EXIST")
            bateaux_true = find_bateaux(read_file(dir_true, file[0:-4]))
            bateaux_test = find_bateaux(read_file(dir_test, file[0:-4]))
            total += len(bateaux_true)
            if(len(bateaux_true) <= 0 or len(bateaux_test) <= 0):
                nb_empty_file+=1
            nb_file += 1

            bateaux_test = bateau_reOrder(bateaux_true, bateaux_test)

            if(len(bateaux_true) != len(bateaux_test)):
                print("PAS LE MEME NOMBRE DE BATEAUX DANS FICHIER TRUE ET FICHIER TEST")
                print("Length True = ", len(bateaux_true))
                print("Length Test = ", len(bateaux_test))
                print("\n")

       # for i in range(len(bateaux_true)):
            #bateau_pair = BateauPair()
            #bateau_pair.Bateau_True = bateaux_true[i]
            #bateau_pair.Bateau_Test = bateaux_test[i]
            #pair_list.append(bateau_pair)

    if(len(pair_list) > 0):
        erreur_bateaux = total - len(pair_list)
        taux_erreur_bateaux = len(pair_list)/total*100

        print("Test effectué sur",(nb_file-nb_empty_file), "fichier(s) (+", nb_empty_file, "fichier(s) vide)\n")
        print("Test effectué avec", len(pair_list), "bateaux trouvés, sur", total, "initialement annotés.", "Soit un taux de", taux_erreur_bateaux, "% de bateaux trouvés")
        if(erreur_bateaux > 0):
            print("(Soit une erreur de", erreur_bateaux, "bateaux manquants)")
        elif(erreur_bateaux < 0):
            print("(Soit une erreur de", abs(erreur_bateaux), "bateaux en trop)")
        else:
            print("(Soit une erreur de bateaux = Aucun bateau manquant)")
        print("(Le milieu de chaque bateau est comparé avec la véritée terrain, limite :", limit, "pixels d'erreur)")
        print("")

    return len(pair_list)
#--------------------------------------------------------------------------#


#--------------------- réduit le nom de chaque fichier à 9 carractères ----------------------------#
def rename_dir(dir):
    all_files = os.listdir(path  + "/" + dir + "/")
    for file in all_files:
        if(os.path.exists(path  + "/" + dir + "/" + file)):
            os.rename(path  + "/" + dir + "/" + file, path  + "/" + dir + "/" + file[0:9] + ".txt")

    return 0
#--------------------------------------------------------------------------------------------------#

#--------------------- Si le bateau a une tros grosse erreur, il n'est pas compté ----------------------------#
def is_boat_valid(Bateau_true, Bateau_test, limit):
    valid = False

    #D = sqrt((abs(float(Bateau_test.rectangle.x)-float(Bateau_true.rectangle.x)))**2 + (abs(float(Bateau_test.rectangle.y)-float(Bateau_true.rectangle.y)))**2)
    D = sqrt((abs(float(Bateau_test.avant.x)-float(Bateau_true.avant.x)))**2 + (abs(float(Bateau_test.avant.y)-float(Bateau_true.avant.y)))**2)

    if(D < limit):
        valid = True

    return valid
#-------------------------------------------------------------------------------------------------------------#

#--------------------- La direction du bateau trouvée est-elle juste ----------------------------#
def is_direction_correct(Bateau_true, Bateau_test):
    correct = False
    milieu_poupe_true = coord()
    milieu_poupe_true.x = (float(Bateau_true.arriere_g.x) + float(Bateau_true.arriere_d.x))/2
    milieu_poupe_true.y = (float(Bateau_true.arriere_g.y) + float(Bateau_true.arriere_d.y))/2

    D_proue = sqrt((abs(float(Bateau_test.avant.x)-float(Bateau_true.avant.x)))**2 + (abs(float(Bateau_test.avant.y)-float(Bateau_true.avant.y)))**2)
    D_poupe = sqrt((abs(float(Bateau_test.avant.x)-float(milieu_poupe_true.x)))**2 + (abs(float(Bateau_test.avant.y)-float(milieu_poupe_true.y)))**2)

    if(D_proue < D_poupe):
        correct = True

    return correct
#-------------------------------------------------------------------------------------------------#

#--------------------- Calcul la robustesse du programme (le raux d'erreurs) ----------------------------#
def robustesse():
    taux = 0
    nb_bateaux = 0

    for i in range(len(pair_list)):
        #if(is_boat_valid(pair_list[i].Bateau_True, pair_list[i].Bateau_Test, limit)):
        nb_bateaux += 1
        if(is_direction_correct(pair_list[i].Bateau_True, pair_list[i].Bateau_Test)):
            pair_list_correct.append(pair_list[i])

    print("Parmi", nb_bateaux, "bateaux trouvés et valide,", len(pair_list_correct), "trouvés dans le bon sens")

    taux = len(pair_list_correct)/nb_bateaux*100

    return taux
#--------------------------------------------------------------------------------------------------------#

#--------------------- Calcul l'erreur de distance en pixels entre 2 points ----------------------------#
def get_erreur_pixels(true, test):
    erreur = [0,0]
    erreur[0] = float(true.x) - float(test.x)
    erreur[1] = float(true.y)  - float(test.y)
    return erreur
#-------------------------------------------------------------------------------------------------------#

#--------------------- Calcul l'erreur de distance en pixels entre 2 Bateaux ----------------------------#
def erreur_pixels(limit, DEBUG):

    tab_erreurs_pixels_X = []
    tab_erreurs_pixels_Y = []

    erreur_total = [0,0]
    erreur_moyenne = [0,0]

    erreur_moyenne_avant = [0,0]
    erreur_moyenne_arriere = [0,0]

    nb_bateaux = 0

    for i in range(len(pair_list_correct)):
        erreur_avant = get_erreur_pixels(pair_list_correct[i].Bateau_True.avant, pair_list_correct[i].Bateau_Test.avant)
        erreur_arriere_g = get_erreur_pixels(pair_list_correct[i].Bateau_True.arriere_g, pair_list_correct[i].Bateau_Test.arriere_g)
        erreur_arriere_d = get_erreur_pixels(pair_list_correct[i].Bateau_True.arriere_d, pair_list_correct[i].Bateau_Test.arriere_d) 

        erreur = [erreur_avant[0] + erreur_arriere_g[0] + erreur_arriere_d[0], erreur_avant[1] + erreur_arriere_g[1] + erreur_arriere_d[1]]

        if((abs(float(erreur[0])) < limit) and (abs(float(erreur[1])) < limit)):
            nb_bateaux += 1
            erreur_total[0] +=  erreur_avant[0] + erreur_arriere_g[0] + erreur_arriere_d[0]
            erreur_total[1] +=  erreur_avant[1] + erreur_arriere_g[1] + erreur_arriere_d[1]
            erreur_moyenne[0] +=  (abs(erreur_avant[0]) + abs(erreur_arriere_g[0]) + abs(erreur_arriere_d[0]))/3
            erreur_moyenne[1] +=  (abs(erreur_avant[1]) + abs(erreur_arriere_g[1]) + abs(erreur_arriere_d[1]))/3
            erreur_moyenne_avant[0] += abs(erreur_avant[0])
            erreur_moyenne_avant[1] += abs(erreur_avant[1])
            erreur_moyenne_arriere[0] +=  (abs(erreur_arriere_g[0]) + abs(erreur_arriere_d[0]))/2
            erreur_moyenne_arriere[1] +=  (abs(erreur_arriere_g[1]) + abs(erreur_arriere_d[1]))/2
            
            if(DEBUG > 1):
                print()
                print("Erreur Proue : ", erreur_avant)
                print("Erreur Poupe Gauche : ", erreur_arriere_g)
                print("Erreur Poupe Droit  : ", erreur_arriere_d)

            tab_erreurs_pixels_X.append(erreur_avant[0])
            tab_erreurs_pixels_X.append(erreur_arriere_g[0])
            tab_erreurs_pixels_X.append(erreur_arriere_d[0])
            tab_erreurs_pixels_Y.append(erreur_avant[1])
            tab_erreurs_pixels_Y.append(erreur_arriere_g[1])
            tab_erreurs_pixels_Y.append(erreur_arriere_d[1])

    erreur_moyenne[0] = erreur_moyenne[0]/len(pair_list_correct)
    erreur_moyenne[1] = erreur_moyenne[1]/len(pair_list_correct)
    erreur_moyenne_avant[0] = erreur_moyenne_avant[0]/len(pair_list_correct)
    erreur_moyenne_avant[1] = erreur_moyenne_avant[1]/len(pair_list_correct)
    erreur_moyenne_arriere[0] = erreur_moyenne_arriere[0]/len(pair_list_correct)
    erreur_moyenne_arriere[1] = erreur_moyenne_arriere[1]/len(pair_list_correct)




    ecart_type_X = statistics.pstdev(tab_erreurs_pixels_X)
    ecart_type_Y = statistics.pstdev(tab_erreurs_pixels_Y)

    print("Précision mesurée sur", nb_bateaux, "bateaux parmi ceux trouvés dans le bon sens (limite d'erreur totale de", limit, "pixels)")
    #print(tab_erreurs_pixels_X)

    value = [erreur_total[0], erreur_total[1], erreur_moyenne[0], erreur_moyenne[1], ecart_type_X, ecart_type_Y, erreur_moyenne_avant[0], erreur_moyenne_avant[1], erreur_moyenne_arriere[0], erreur_moyenne_arriere[1]]

    return value
#-----------------------------------------------------------------------------------------------------#

#--------------------- lit les tous les fichiers ----------------------------#
def tranform_all_files_to_keypoints(dir_to_transform):

    all_files = os.listdir(path  + "/" + dir_to_transform + "/")

    for file in all_files:
        print("File = ", file)
        f = open(path + "/" + dir_to_transform + "/" + file, "r")
        str_file = f.read()
        f.close()

        bateaux = []
        string_tab = str_file.split("\n")

        for j in range(int(len(string_tab)/4)):
            bateau = Bateau()
            bateau.avant     = coord()
            bateau.arriere_d = coord()
            bateau.arriere_g = coord()
            for i in range(j*4, ((j*4)+4)):
                string_values = string_tab[i].split(" ")
                if(int(string_values[0]) == 0):
                    bateau.avant.x  = string_values[1]
                    bateau.avant.y  = string_values[2]
                if(int(string_values[0]) == 2):
                    bateau.arriere_d.x  = string_values[1]
                    bateau.arriere_d.y  = string_values[2]
                if(int(string_values[0]) == 3):
                    bateau.arriere_g.x  = string_values[1]
                    bateau.arriere_g.y  = string_values[2]
            bateaux.append(bateau)

        fmode = "x"
        if(os.path.exists(path  + "/" + "true_AI_2" + "/" + file)):
            fmode = "w"
        else:
            fmode = "x"

        f = open(path  + "/" + "true_AI_2" + "/" + file, fmode)

        for i in range(len(bateaux)):
            f.write("0 " + str(bateaux[i].avant.x) + " " + str(bateaux[i].avant.y) + "\n")
            f.write("2 " + str(bateaux[i].arriere_d.x) + " " + str(bateaux[i].arriere_d.y) + "\n")
            f.write("3 " + str(bateaux[i].arriere_g.x) + " " + str(bateaux[i].arriere_g.y))
            if(i < (len(bateaux)-1)):
                f.write("\n")

        f.close()


    return 0
#--------------------------------------------------------------------------#

#------------- main -------------#
DEBUG = 0

path = os.path.dirname(__file__)

dir_true = 'verite_terrain'
dir_test = 'input'

image_size = 768 #Nb de pixels en taille d'image
limit = 30
limit_erreur = 100

if(read_all_files() > 0):

    taux = 0
    taux = robustesse()
    print("Robustesse =", taux, "%\n")

    erreur_pixel = (0,0,0,0,0,0,0,0,0,0)
    erreur_pixel = erreur_pixels(limit_erreur, DEBUG)
    precision_pixels_X = erreur_pixel[0]
    print("Erreur Total X =", precision_pixels_X,"pixels", "                               (pixels d'erreur total entre les keypoints)")
    precision_pixels_Y = erreur_pixel[1]
    print("Erreur Total Y =", precision_pixels_Y,"pixels", "                               (pixels d'erreur total entre les keypoints)")
    precision_pixels_Moyenne_X = erreur_pixel[2]
    print("Précision Moyenne X =", precision_pixels_Moyenne_X,"pixels", "             (pixels d'erreur en moyenne entre les keypoints)")
    precision_pixels_Moyenne_Y = erreur_pixel[3]
    print("Précision Moyenne Y =", precision_pixels_Moyenne_Y,"pixels", "             (pixels d'erreur en moyenne entre les keypoints)\n")
    precision_pixels_Moyenne_avant_X = erreur_pixel[4]
    print("Précision Moyenne Avant X =", precision_pixels_Moyenne_avant_X,"pixels")
    precision_pixels_Moyenne_avant_Y = erreur_pixel[5]
    print("Précision Moyenne Avant Y =", precision_pixels_Moyenne_avant_Y,"pixels")
    precision_pixels_Moyenne_arriere_X = erreur_pixel[6]
    print("Précision Moyenne Arrière X =", precision_pixels_Moyenne_arriere_X,"pixels")
    precision_pixels_Moyenne_arriere_Y = erreur_pixel[7]
    print("Précision Moyenne Arrière Y =", precision_pixels_Moyenne_arriere_Y,"pixels\n")


    ecart_type = [erreur_pixel[8], erreur_pixel[9]]
    print("Ecart Type X =", erreur_pixel[8], "Pixels")
    print("Ecart Type Y =", erreur_pixel[9], "Pixels \n")
else:
    print("\nAucune correspondance trouvée entre la véritée terrain et l'input de test\n")