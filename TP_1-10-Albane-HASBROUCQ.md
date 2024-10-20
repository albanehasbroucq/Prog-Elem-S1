---
jupytext:
  cell_metadata_json: true
  encoding: '# -*- coding: utf-8 -*-'
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
language_info:
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
nbhosting:
  title: suite du TP simple avec des images
---

Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

+++

# TP images (2/2)

merci à Wikipedia et à stackoverflow

**le but de ce TP n'est pas d'apprendre le traitement d'image  
on se sert d'images pour égayer des exercices avec `numpy`  
(et parce que quand on se trompe ça se voit)**

+++

Albane Hasbroucq

```{code-cell} ipython3
import numpy as np
from matplotlib import pyplot as plt
```

+++ {"tags": ["framed_cell"]}

````{admonition} → **notions intervenant dans ce TP**

* sur les tableaux `numpy.ndarray`
  * `reshape()`, masques booléens, *ufunc*, agrégation, opérations linéaires
  * pour l'exercice `patchwork`:  
    on peut le traiter sans, mais l'exercice se prête bien à l'utilisation d'une [indexation d'un tableau par un tableau - voyez par exemple ceci](https://ue12-p24-numerique.readthedocs.io/en/main/1-14-numpy-optional-indexing-nb.html)
  * pour l'exercice `sepia`:  
    ici aussi on peut le faire "naivement" mais l'utilisation de `np.dot()` peut rendre le code beaucoup plus court
* pour la lecture, l'écriture et l'affichage d'images
  * utilisez `plt.imread()`, `plt.imshow()`
  * utilisez `plt.show()` entre deux `plt.imshow()` si vous affichez plusieurs images dans une même cellule

  ```{admonition} **note à propos de l'affichage**
  :class: seealso dropdown admonition-small

  * nous utilisons les fonctions d'affichage d'images de `pyplot` par souci de simplicité
  * nous ne signifions pas là du tout que ce sont les meilleures!  
    par exemple `matplotlib.pyplot.imsave` ne vous permet pas de donner la qualité de la compression  
    alors que la fonction `save` de `PIL` le permet
  * vous êtes libres d'utiliser une autre librairie comme `opencv`  
    si vous la connaissez assez pour vous débrouiller (et l'installer), les images ne sont qu'un prétexte...
  ```
````

+++

## Création d'un patchwork

+++

1. Le fichier `data/rgb-codes.txt` contient une table de couleurs:
```
AliceBlue 240 248 255
AntiqueWhite 250 235 215
Aqua 0 255 255
.../...
YellowGreen 154 205 50
```
Le nom de la couleur est suivi des 3 valeurs de ses codes `R`, `G` et `B`  
Lisez cette table en `Python` et rangez-la dans la structure qui vous semble adéquate.

```{code-cell} ipython3
:scrolled: true

#On utilise une structure de dictionnaire, avec des arrays pour contenir les valeurs RGB 
colors=dict()
filename= 'data/rgb-codes.txt'
with open(filename, 'r') as file:
    for line in file:
        colname, *l = line.split()
        colors[colname]= np.array([int(e) for e in l], dtype=np.uint8)
print(colors)

#on aurait ausssi pu faire avec une structure de dictionnaire et mettre les valeurs RGB dans un tuple:
#colors=dict()
#fichier = 'data/rgb-codes.txt'
#with open(fichier, 'r') as f:
#    for ligne in f:
#        donnees=[str(mot) for mot in ligne.split()]
#        colors[str(donnees[0])]=(donnees[1], donnees[2], donnees[3])
#    f.close()
```

2. Affichez, à partir de votre structure, les valeurs rgb entières des couleurs suivantes  
`'Red'`, `'Lime'`, `'Blue'`

```{code-cell} ipython3
#On affiche les valeurs RGB des couleurs en les cherchant dans le dictionnaire, le nom de la couleur étant la clé:
print(colors['Red'], colors['Lime'], colors['Blue'])
```

3. Faites une fonction `patchwork` qui  

   * prend une liste de couleurs et la structure donnant le code des couleurs RGB
   * et retourne un tableau `numpy` avec un patchwork de ces couleurs  
   * (pas trop petits les patchs - on doit voir clairement les taches de couleurs  
   si besoin de compléter l'image mettez du blanc

+++

````{admonition} indices
:class: dropdown
  
* sont potentiellement utiles pour cet exo:
  * la fonction `np.indices()`
  * [l'indexation d'un tableau par un tableau](https://ue12-p24-numerique.readthedocs.io/en/main/1-14-numpy-optional-indexing-nb.html)
* aussi, ça peut être habile de couper le problème en deux, et de commencer par écrire une fonction `rectangle_size(n)` qui vous donne la taille du patchwork en fonction du nombre de couleurs  
  ```{admonition} et pour calculer la taille au plus juste
  :class: tip dropdown

  en version un peu brute, on pourrait utiliser juste la racine carrée;
  par exemple avec 5 couleurs créer un carré 3x3 - mais 3x2 c'est quand même mieux !

  voici pour vous aider à calculer le rectangle qui contient n couleurs

  n | rect | n | rect | n | rect | n | rect |
  -|-|-|-|-|-|-|-|
  1 | 1x1 | 5 | 2x3 | 9 | 3x3 | 14 | 4x4 |
  2 | 1x2 | 6 | 2x3 | 10 | 3x4 | 15 | 4x4 |
  3 | 2x2 | 7 | 3x3 | 11 | 3x4 | 16 | 4x4 |
  4 | 2x2 | 8 | 3x3 | 12 | 3x4 | 17 | 4x5 |
  ```
````

```{code-cell} ipython3
#On commence par faire une fonction qui détermine la taille idéale du patchwork en focntion du nombre de couoleur qu'on doit y
#mettre. On utilise i la partie entière et j la partie entière + 1. Si i*j est assez grand, on choisit ces dimensions, sinon,
#on majore en prenant j*j.

import math
def rectangle_size(n): 

    racine = np.sqrt(n)
    i = math.floor(racine)
    j = math.ceil(racine)
    if i * j >= n:
        return [i, j]
    else :
        return [j,j]
```

```{code-cell} ipython3
:scrolled: true

#Maintenant que l'on peut vite obtenir la bonne taille, on définit la fonction patchwork.
#(j'ai essayé avec la méthodes d'indexation mais mes fonctions ne marchaient pas donc j'ai fini par mettre des boucles for ...)
def patchwork(liste_couleurs, colors):
    n = len(liste_couleurs)
    taille = rectangle_size(n)
    patch=np.ones((taille[0],taille[1],3))*255 #on crée un tableau blanc de la bonne dimmension
    RGB=[] #on crée une liste vide dans laquelle on mettre les valeurs RGB de notre liste de couleurs, que l'on retrouve par le dictionnaire
    for c in liste_couleurs:
        RGB.append(colors[c]/255) #on normalise pour avoir le bon type

    i=0
    for j in range(taille[0]):
        for k in range(taille[1]):
            if i<n:
                patch[j,k]=RGB[i] #on ajoute chaque carré de couleur à notre tableau patch, en prenant un par un dans la liste RGB
                i+=1

    plt.axis('off') 
    plt.imshow(patch)
    
            
```

4. Tirez aléatoirement une liste de couleurs et appliquez votre fonction à ces couleurs.

```{code-cell} ipython3
nb=np.random.randint(0,200) #on choisit un nombre de couleurs au hasard
liste_couleurs= [np.random.choice(list(colors.keys())) for _ in range(nb)] #on crée une liste de couleurs prises aléatoirement dans l'ensemble des couelurs du dictionnaire
patchwork(liste_couleurs, colors) #on applique la focntion patchwork à cette liste
```

5. Sélectionnez toutes les couleurs à base de blanc et affichez leur patchwork  
même chose pour des jaunes

```{code-cell} ipython3
#Pour sélectionner les couleurs blanches, le blanc étant de code RGB (255,255,255), on choisit ici les couleurs dont la 
#moyenne des valeurs RGB dépasse 240, et ce avec un écart type inférieur à 10
blancs=[]
for x in colors.keys():
    y=colors[x]
    if np.mean(y) >= 240 and np.std(y)<10:
        blancs.append(x)
        
patchwork(blancs, colors) #on applique le patchwork à cette liste de blancs
#mais on aurait pu faire en sélectionnant les couleurs contant le mon 'white'
```

```{code-cell} ipython3
#De meme, le jaune étant de code RGB (255,255,0), on choisit que la moyenne des deux premieres valeurs RGB doit dépasser 220
#et la 3ème valeur ne doit pas excéder 20 pour etre considéré comme jaune.
jaunes=[]
for x in colors.keys():
    y=colors[x]
    y_jaune=[y[0],y[1]]
    if np.mean(y_jaune) >= 220 and y[2]<20:
        jaunes.append(x)
        
patchwork(jaunes, colors) #on applique le patchwork à cette liste de jaunes
```

6. Appliquez la fonction à toutes les couleurs du fichier  
et sauver ce patchwork dans le fichier `patchwork.png` avec `plt.imsave`

```{code-cell} ipython3
#Pour l'appliquer à toutes les couleurs du fichier, on prend comme liste l'ensemble des clés du dictionnaire:
mon_patch=patchwork(colors.keys(), colors)
plt.savefig('patchwork.png') #j'utilise savefig pour sauver ce patchwork 
```

7. Relisez et affichez votre fichier  
   attention si votre image vous semble floue c'est juste que l'affichage grossit vos pixels

vous devriez obtenir quelque chose comme ceci

```{image} media/patchwork-all.jpg
:align: center
```

```{code-cell} ipython3
#Pour relire le fichier, on fait appel à imread, et on affiche grace à imshow (ici encore, on enlève les axes par esthétisme)
patch_à_lire='patchwork.png'
patch=plt.imread(patch_à_lire)
plt.imshow(patch)
plt.axis('off')
```

## Somme dans une image & overflow

+++

0. Lisez l'image `data/les-mines.jpg`

```{code-cell} ipython3
im=plt.imread('data/les-mines.jpg') #lecture de l'image
im.shape #j'ai décidé d'afficher la taille de l'image
```

1. Créez un nouveau tableau `numpy.ndarray` en sommant **avec l'opérateur `+`** les valeurs RGB des pixels de votre image

```{code-cell} ipython3
#Pour sommer les valeurs RGB, je les récupère d'abord une par une dans des tableaux R, G et B (par slicing), qui sont donc
#chacun de dim (533,800), et je somme ensuite ces 3 tableaux dans un nouveau tableau, appelé tab_somme:
R=im[:,:,0]
G=im[:,:,1]
B=im[:,:,2]
tab_somme=R+G+B
```

2. Regardez le type de cette image-somme, et son maximum; que remarquez-vous?  
   Affichez cette image-somme; comme elle ne contient qu'un canal il est habile de l'afficher en "niveaux de gris" (normalement le résultat n'est pas terrible ...)


   ```{admonition} niveaux de gris ?
   :class: dropdown tip

   cherchez sur google `pyplot imshow cmap gray`
   ```

```{code-cell} ipython3
print(tab_somme.dtype) #on affiche le type de cette image 
print(tab_somme.max()) #et son maximum
```

On remarque que le max (255) est le max possible: en effet, le type étant uint8, les pixels sont codés de 0 à 255.

```{code-cell} ipython3
plt.imshow(tab_somme, cmap='gray') #on ajoute l'argument cmap pour afficher les niveaux de gris
```

3. Créez un nouveau tableau `numpy.ndarray` en sommant mais cette fois **avec la fonction d'agrégation `np.sum`** les valeurs RGB des pixels de votre image

```{code-cell} ipython3
#Nous allons cette fois utiliser la fonction d'agrégation directemnt disponible avec numpy.
tab_somme2=np.sum(im, axis=2) # on fait la somme sur le 3ème axe (axis =2 parce que ca démarre à 0)
print(tab_somme2==tab_somme) #On remarque tout de suite que, par ces deux méthodes, on n'obtient pas le meme résultat.
```

4. Comme dans le 2., regardez son maximum et son type, et affichez la

```{code-cell} ipython3
print(tab_somme2.dtype)
print(tab_somme2.max())
```

Le type est cette fois uint64, donc de 64 bits, contrairement à tout à l'heure où on avait un octet.
De même, le max (765) excède largement 255 pusiqu'on code avec plus de bits.
Ceci explique alors la différence obtenue précédemment ; le nombre de bits n'est pas le même donc les pixels ne sont pas 
distribués sur le même spectre de valeurs.

```{code-cell} ipython3
plt.imshow(tab_somme2, cmap='gray')
```

5. Les deux images sont de qualité très différente, pourquoi cette différence ? Utilisez le help `np.sum?`

+++

Elles ne sont pas du tout de même qualité, puisqu'elles ne sont pas codées avec le même nombre de bits; il y a un fort gain 
de précision avec la fonction d'agrégation qui se base sur uint64.
En effet, c'est np.sum qui utilise de lui meme plus de bits, par défaut il fonctionne avec 64 bits, si on ne lui donne pas 
d'indication contraire.

+++

6. Passez l'image en niveaux de gris de type entiers non-signés 8 bits  
(de la manière que vous préférez)

```{code-cell} ipython3
#Il faut convertir de uint64 vers uint8. Donc il faut normaliser, parce que avce uint8, on doit avoir des valuers comprises
#entre 0 et 255 avant de changer le type pour ne pas perdre toutes les valeurs supérieures à 255.
tab_somme2_normal= (tab_somme2-tab_somme2.min())*255 / (tab_somme2.max()-tab_somme2.min()) #normalisation
tab_somme2_uint8 = tab_somme2_normal.astype(np.uint8) #maintenant, on peut convertire en entiers non signés 8 bits
#vérifications:
print(tab_somme2_uint8.dtype)
print(tab_somme2_uint8.shape)
plt.imshow(tab_somme2_uint8, cmap='gray')
```

7. Remplacez dans l'image en niveaux de gris,  
les valeurs >= à 127 par 255 et celles inférieures par 0  
Affichez l'image avec une carte des couleurs des niveaux de gris  
vous pouvez utilisez la fonction `numpy.where`

```{code-cell} ipython3
#On remplace les valeurs comme demandé :
tab_somme2_uint8[tab_somme2_uint8 >=127]=255
tab_somme2_uint8[tab_somme2_uint8 <127]=0
print(tab_somme2_uint8) #on vérifie que les changements ont bien été faits
plt.imshow(tab_somme2_uint8,cmap='gray') # on affiche en niveaux de gris
```

8. avec la fonction `numpy.unique`  
regardez les valeurs différentes que vous avez dans votre image en noir et blanc

```{code-cell} ipython3
np.unique(tab_somme2_uint8)
#c'est logique, on a du noir et du blanc : 0 et 255
```

## Image en sépia

+++

Pour passer en sépia les valeurs R, G et B d'un pixel  
(encodées ici sur un entier non-signé 8 bits)  

1. on transforme les valeurs `R`, `G` et `B` par la transformation  
`0.393 * R + 0.769 * G + 0.189 * B`  
`0.349 * R + 0.686 * G + 0.168 * B`  
`0.272 * R + 0.534 * G + 0.131 * B`  
(attention les calculs doivent se faire en flottants pas en uint8  
pour ne pas avoir, par exemple, 256 devenant 0)  
1. puis on seuille les valeurs qui sont plus grandes que `255` à `255`
1. naturellement l'image doit être ensuite remise dans un format correct  
(uint8 ou float entre 0 et 1)

+++

````{tip}
jetez un coup d'oeil à la fonction `np.dot` 
qui est si on veut une généralisation du produit matriciel

dont voici un exemple d'utilisation:
````

```{code-cell} ipython3
:scrolled: true

# exemple de produit de matrices avec `numpy.dot`
# le help(np.dot) dit: dot(A, B)[i,j,k,m] = sum(A[i,j,:] * B[k,:,m])

i, j, k, m, n = 2, 3, 4, 5, 6
A = np.arange(i*j*k).reshape(i, j, k)
B = np.arange(m*k*n).reshape(m, k, n)

C = A.dot(B)
# or C = np.dot(A, B)

print(f"en partant des dimensions {A.shape} et {B.shape}")
print(f"on obtient un résultat de dimension {C.shape}")
print(f"et le nombre de termes dans chaque `sum()` est {A.shape[-1]} == {B.shape[-2]}")
```

**Exercice**

+++

1. Faites une fonction qui prend en argument une image RGB et rend une image RGB sépia  
la fonction `numpy.dot` peut être utilisée si besoin, voir l'exemple ci-dessus

```{code-cell} ipython3
# On définit la focntion sépia qui prend en argument l'image qu'on veut modifier:
def sepia(im):
    im_float=im.astype(np.float32) #on commence par convertir puisque les calculs doivent se faire en flottants 
    #on définit la matrice avec les coefficients qui permettent de transformer en sepia:
    mat_sep=np.array([[0.393, 0.349, 0.272],
                      [0.769, 0.686, 0.534],
                      [0.189, 0.168, 0.131]])
    im_sep=np.dot(im_float, mat_sep) #on réalise le produit matriciel pour réaliser la transformation des valeurs RGB
    im_sep[im_sep>255]=255 #on seuille
    im_sep_uint8=im_sep.astype(np.uint8) #puis on repasse en type uint8
    return(im_sep_uint8)                     
```

2. Passez votre patchwork de couleurs en sépia  
Lisez le fichier `patchwork-all.jpg` si vous n'avez pas de fichier perso

```{code-cell} ipython3
patch=plt.imread('patchwork.png')
patch=patch[:,:,0:3] #on restreint le tableau aux valeurs RGB, puisqu'on avait ici du RGBA (valeur de transparence)
patch_uint8=(patch*255).astype(np.uint8) #on passe des valeurs de 0 à  à celles de 0 à 255 et on convertit en tableau basé du 1 octet
patch_sepia=sepia(patch_uint8) # on applique la fonction sepia 

plt.imshow(patch_sepia)
plt.axis('off')
```

3. Passez l'image `data/les-mines.jpg` en sépia

```{code-cell} ipython3
im=plt.imread('data/les-mines.jpg')#on lit l'image
im_sepia=sepia(im)#on applique la fonction sepia
plt.imshow(im_sepia)#on affiche
```

## Exemple de qualité de compression

+++

1. Importez la librairie `Image`de `PIL` (pillow)  
(vous devez peut être installer PIL dans votre environnement)

```{code-cell} ipython3
from PIL import Image
```

2. Quelle est la taille du fichier `data/les-mines.jpg` sur disque ?

```{code-cell} ipython3
file = "data/les-mines.jpg"
```

```{code-cell} ipython3
#Pour avoir la taille sur disque, je fais appel au module os:
import os 
os.path.getsize(file)
```

3. Lisez le fichier 'data/les-mines.jpg' avec `Image.open` et avec `plt.imread`

```{code-cell} ipython3
im_pil=Image.open('data/les-mines.jpg') # on lit le fichier avec la méthode open de Image
im_pil_tableau=np.array(im_pil) #on convertit en array
im_plt=plt.imread('data/les-mines.jpg') # on lit cette fois avec matplotlib 
```

4. Vérifiez que les valeurs contenues dans les deux objets sont proches

```{code-cell} ipython3
print(im_pil_tableau.shape, im_plt.shape)
proche=np.allclose(im_pil_tableau,im_plt) #on regarde si les valeurs sont les memes (on aurait pu avec un ==)
print(im_pil_tableau==im_plt) #on peut aussi faire comme ca

print(proche)
```

5. Sauvez (toujours avec de nouveaux noms de fichiers)  
l'image lue par `imread` avec `plt.imsave`  
l'image lue par `Image.open` avec `save` et une `quality=100`  
(`save` s'applique à l'objet créé par `Image.open`)

```{code-cell} ipython3
#sauvegarde:
plt.imsave('data/les-mines-imread-de-plt.jpg',im_plt)
im_pil.save('data/les-mines-save-de-pil.jpg', quality=100)
```

6. Quelles sont les tailles de ces deux fichiers sur votre disque ?  
Que constatez-vous ?

```{code-cell} ipython3
#on affiche encore la taille des fichiers à l'aide de os:
file_imread='data/les-mines-imread-de-plt.jpg'
file_save='data/les-mines-save-de-pil.jpg'
os.path.getsize(file_imread), os.path.getsize(file_save)
```

7. Relisez les deux fichiers créés et affichez avec `plt.imshow` leur différence

```{code-cell} ipython3
image_imread=plt.imread('data/les-mines-imread-de-plt.jpg')
image_save=plt.imread('data/les-mines-save-de-pil.jpg')
diff=np.abs(image_imread-image_save) #on calcule la différence entre nos deux images
plt.imshow(diff) #puis on l'affiche 
#je suis vraiment pas sue de moi là
```
