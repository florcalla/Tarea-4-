# -*-coding:utf-8 -*

import os
from bitstring import BitArray
import hashlib
import ast
import time
import random
import pickle
import codecs
import Crypto

cstarray = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]


CONST_PATHFILE = os.path.realpath("__file__")
def decompose(e): # Write a number as a sum of powers of 2
    binaryList = []
    powerList = []

    q = 1 # Giving q a value, so it can bound the first time

    while q != 0:
        # Quotient and remainder
        q = e / 2
        r = e % 2

        binaryList.append(r)

        e = e / 2

    # Now populating the power list
    bitPos = 0
    for bit in binaryList:
        if bit == 1:
            powerList.append(2**bitPos)
        bitPos += 1

    return powerList

def remove2EOL(blockarray):
    for c in blockarray:
        if c == "\n":
            ind = blockarray.index(c)
            if blockarray[ind+1]=="\n":
                blockarray.pop(ind)
    return blockarray
def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m

def tobits(s): # Converts characters into a list of bits
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def frombits(bits): # Converts a list of bits into characters
    chars = []
    for b in range(len(bits) // 8):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def isGenerator(a,p):
    k = p-1
    while k%2==0:
        testvalue = computeModuleWithExpoBySquaring(a,k//2,p)
        if testvalue == 1:
            return False
        k = k//2
    return True    
def computeModuleWithExpoBySquaring(a, e, m): # Computing a^(e) mod m
    modList = []
    currentPower = 1
    actualPower = 1
    currentMod = a
    # First step : writing e as the sum of powers of 2
    powerList = decompose(e)
    maxPower = max(powerList)
    while currentPower <= maxPower:
        currentMod = currentMod ** actualPower % m
        if currentPower in powerList:
            modList.append(currentMod)
        currentPower = currentPower * 2
        actualPower = 2

    listProduct=1
    for mod in modList:
        listProduct = listProduct*mod

    return listProduct % m


def getscriptpath():
    print("Original pathfile : {0}".format(CONST_PATHFILE))
    filename = os.path.basename("__file__")
    print("Filename used : {0}".format(filename))
    filepath = CONST_PATHFILE.replace(filename, "")
    filepath = filepath[:-1]
    print("Filepath returned : {0}".format(filepath))
    print(filepath)
    return filepath

def restorepublickey(filename): # Récupération de la clé publique avec le nom du fichier
    currentpath = getscriptpath()

    os.chdir(currentpath + "/cramershoup/publickeys_cs")
    listparam = []
    with open(filename, 'rb') as key:
        depickler = pickle.Unpickler(key)
        key = depickler.load()
        for value in key.values():
            #print(value)
            listparam.append(value)
    return listparam
def restoreprivatekey(filename): # On récupère la clé privée de la même façon
    currentpath = getscriptpath()
    os.chdir(currentpath + "/cramershoup/privatekeys_cs")
    listparam = []
    with open(filename, 'rb') as key:
        depickler = pickle.Unpickler(key)
        key = depickler.load()
        for value in key.values():
            #print(value)
            listparam.append(value)
    return listparam

def decipher_cramershoup(msgfilename):
    msgfinal = ""
    currentpath = getscriptpath()
    #On isole le timestamp pour retrouver les fichiers des clés
    os.chdir(currentpath + "/cramershoup/cipheredmsgs")
    timestampinter = msgfilename.split("_")
    timestamp = timestampinter[1].split(".")[0]
    filetoopen = open(msgfilename)
    data = []
    for line in filetoopen:
        data.append(line)

    u1 = data[0]
    u2 = data[1]
    e = data[2]
    v = data[3]

    privatekeyname = "privatekey_"+timestamp+".prvk"
    publickeyname = "publickey_"+timestamp+".pblk"

    #On récupère les clés
    key = restoreprivatekey(privatekeyname)
    keypb = restorepublickey(publickeyname)

    x1 = key[0]
    x2 = key[1]
    y1 = key[2]
    y2 = key[3]
    z = key[4]

    h = int(keypb[2])
    g1 = int(keypb[3])
    g2 = int(keypb[4])

    q = int(keypb[5])

    #on avait séparé les blocs dans le message stocké en ajoutant des //
    arraye = e.split("//")
    arraye.pop()
    arrayv = v.split("//")
    arrayv.pop()

    i = 0

    numberu1 = int(u1,16)
    numberu2 = int(u2,16)

    numberx1 = int(x1)
    numberx2 = int(x2)

    numbery1 = int(y1)
    numbery2 = int(y2)

    numberz = int(z)

    stringu1 = bytes(codecs.encode(u1[:-1]))
    stringu2 = bytes(codecs.encode(u2[:-1]))

    booleanverif = True

    for ei in arraye:
        print("ei = ",ei)
    for ei in arraye:
        numbere = int(ei,16)
        numberv = int(arrayv[i],16)
        stringciphered = bytes(codecs.encode(ei[2:]))

        #print("STRE = ",ei[2:], "; STRU1 = ",u1," ; STRU2 = ",u2)

        #print("STRINGcipherED = ",stringciphered, " -- STRINGU1 = ",stringu1," -- STRINGU2 = ",stringu2)

        #on calcule le hash pour chaque bloc et on vérifie qu'il est égal au hash stocké
        hashmsg2 = hashlib._hashlib.openssl_sha256(stringciphered+stringu1+stringu2).hexdigest()
        
        #hashmsg2 = sha2(ei[2:]+u1+u2)
        alpha = int(hashmsg2,16)

        #print("\n ALPHA = ",alpha,"\n")

        verif2int = (computeModuleWithExpoBySquaring(numberu1,numbery1,q)*computeModuleWithExpoBySquaring(numberu2,numbery2,q))%q
        verifpt1 = (computeModuleWithExpoBySquaring(numberu1,numberx1,q)*computeModuleWithExpoBySquaring(numberu2,numberx2,q))
        verifpt2 = computeModuleWithExpoBySquaring(verif2int,alpha,q)
        #print("VERIF PT1 = ",(computeModuleWithExpoBySquaring(numberu1,numberx1,q)*computeModuleWithExpoBySquaring(numberu2,numberx2,q)) % q)
        #print("VERIF PT2 = ", computeModuleWithExpoBySquaring(verif2int,alpha,q))

        verif = (verifpt1 * verifpt2) %q
        #print("verif = ",verif)
        #print("numberv = ",numberv)
        if verif != numberv:
            booleanverif = False
            break
        i+=1

    if not booleanverif:
        #print("La vérification est fausse : déchiffrement annulé")
        return("NULL")
    else:

        #si le hash est valide : on déchiffre chaque bloc puis on retourne le tableau des blocs déchiffrés
        arraymsg = []
        u1z = computeModuleWithExpoBySquaring(numberu1,numberz,q)
        #print("u1z = ",u1z)

        msgtotal = ""

        invu1z = modinv(u1z,q)
        for ei in arraye:
            msgblck = []
            numbere = int(ei,16)

            #print("HEX MSG = ",hex(numbere))
            m = numbere*invu1z % q

            strm = str(hex(m))[2:]

            #print("MESSAGE  = ",strm)
            for i in range(len(strm)//2):
                charinhex = strm[2*i]+strm[2*i+1]
                character = chr(int(charinhex,16))
                msgblck.append(character)
            msgblck2 = remove2EOL(msgblck)
            arraymsg.append(msgblck)

            print(msgblck)
            for c in msgblck:
                msgtotal = msgtotal + c
        
        print("Deciphered message : \n",msgtotal)
        return msgtotal

# ------------------------------------------------------------------------------

def decipher_cramershoup(msgfilename):
    msgfinal = ""
    currentpath = getscriptpath()
    #On isole le timestamp pour retrouver les fichiers des clés
    os.chdir(currentpath + "/cramershoup/cipheredmsgs")
    timestampinter = msgfilename.split("_")
    timestamp = timestampinter[1].split(".")[0]
    filetoopen = open(msgfilename)
    data = []
    for line in filetoopen:
        data.append(line)

    u1 = data[0]
    u2 = data[1]
    e = data[2]
    v = data[3]

    privatekeyname = "privatekey_"+timestamp+".prvk"
    publickeyname = "publickey_"+timestamp+".pblk"

    #On récupère les clés
    key = restoreprivatekey(privatekeyname)
    keypb = restorepublickey(publickeyname)

    x1 = key[0]
    x2 = key[1]
    y1 = key[2]
    y2 = key[3]
    z = key[4]

    h = int(keypb[2])
    g1 = int(keypb[3])
    g2 = int(keypb[4])

    q = int(keypb[5])

    #on avait séparé les blocs dans le message stocké en ajoutant des //
    arraye = e.split("//")
    arraye.pop()
    arrayv = v.split("//")
    arrayv.pop()

    i = 0

    numberu1 = int(u1,16)
    numberu2 = int(u2,16)

    numberx1 = int(x1)
    numberx2 = int(x2)

    numbery1 = int(y1)
    numbery2 = int(y2)

    numberz = int(z)

    stringu1 = bytes(codecs.encode(u1[:-1]))
    stringu2 = bytes(codecs.encode(u2[:-1]))

    booleanverif = True

    for ei in arraye:
        print("ei = ",ei)
    for ei in arraye:
        numbere = int(ei,16)
        numberv = int(arrayv[i],16)
        stringciphered = bytes(codecs.encode(ei[2:]))

        #print("STRE = ",ei[2:], "; STRU1 = ",u1," ; STRU2 = ",u2)

        #print("STRINGcipherED = ",stringciphered, " -- STRINGU1 = ",stringu1," -- STRINGU2 = ",stringu2)

        #on calcule le hash pour chaque bloc et on vérifie qu'il est égal au hash stocké
        hashmsg2 = hashlib._hashlib.openssl_sha256(stringciphered+stringu1+stringu2).hexdigest()
        
        #hashmsg2 = sha2(ei[2:]+u1+u2)
        alpha = int(hashmsg2,16)

        #print("\n ALPHA = ",alpha,"\n")

        verif2int = (computeModuleWithExpoBySquaring(numberu1,numbery1,q)*computeModuleWithExpoBySquaring(numberu2,numbery2,q))%q
        verifpt1 = (computeModuleWithExpoBySquaring(numberu1,numberx1,q)*computeModuleWithExpoBySquaring(numberu2,numberx2,q))
        verifpt2 = computeModuleWithExpoBySquaring(verif2int,alpha,q)
        #print("VERIF PT1 = ",(computeModuleWithExpoBySquaring(numberu1,numberx1,q)*computeModuleWithExpoBySquaring(numberu2,numberx2,q)) % q)
        #print("VERIF PT2 = ", computeModuleWithExpoBySquaring(verif2int,alpha,q))

        verif = (verifpt1 * verifpt2) %q
        #print("verif = ",verif)
        #print("numberv = ",numberv)
        if verif != numberv:
            booleanverif = False
            break
        i+=1

    if not booleanverif:
        #print("La vérification est fausse : déchiffrement annulé")
        return("NULL")
    else:

        #si le hash est valide : on déchiffre chaque bloc puis on retourne le tableau des blocs déchiffrés
        arraymsg = []
        u1z = computeModuleWithExpoBySquaring(numberu1,numberz,q)
        #print("u1z = ",u1z)

        msgtotal = ""

        invu1z = modinv(u1z,q)
        for ei in arraye:
            msgblck = []
            numbere = int(ei,16)

            #print("HEX MSG = ",hex(numbere))
            m = numbere*invu1z % q

            strm = str(hex(m))[2:]

            #print("MESSAGE  = ",strm)
            for i in range(len(strm)//2):
                charinhex = strm[2*i]+strm[2*i+1]
                character = chr(int(charinhex,16))
                msgblck.append(character)
            msgblck2 = remove2EOL(msgblck)
            arraymsg.append(msgblck)

            print(msgblck)
            for c in msgblck:
                msgtotal = msgtotal + c
        
        print("Deciphered message : \n",msgtotal)
        return msgtotal
decipher_cramershoup("cipheredmsg_1604862528.cimsg")
getscriptpath()