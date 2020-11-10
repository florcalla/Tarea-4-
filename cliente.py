from pathlib import Path
from tinyec import registry
from Crypto.Cipher import AES
import hashlib, secrets, binascii , time, bcrypt , os, sys, socket, json
curve = registry.get_curve('brainpoolP256r1')


def decrypt_AES_GCM(ciphertext, nonce, authTag, secretKey):
    aesCipher = AES.new(secretKey, AES.MODE_GCM, nonce)
    plaintext = aesCipher.decrypt_and_verify(ciphertext, authTag)
    return plaintext

def ecc_point_to_256_bit_key(point):
    sha = hashlib.sha256(int.to_bytes(point.x, 32, 'big'))
    sha.update(int.to_bytes(point.y, 32, 'big'))
    return sha.digest()


def decrypt_ECC(encryptedMsg, privKey):
    (ciphertext, nonce, authTag, ciphertextPubKey) = encryptedMsg
    sharedECCKey = privKey * ciphertextPubKey
    secretKey = ecc_point_to_256_bit_key(sharedECCKey)
    plaintext = decrypt_AES_GCM(ciphertext, nonce, authTag, secretKey)
    return plaintext
def crear_llaves() :
    nup = secrets.randbelow(curve.field.n)
    ogf = [curve.g , nup ] 
    print( " llave publica : ",nup * curve.g)
    return (ogf , nup ) 
# se crear socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#conecta socket con el puerto 
server_address = ('localhost', 10000)
sock.connect(server_address)
print('..Generando llaves..')

keys = crear_llaves() 
publickey= keys[0]
privatekey = keys[1]

sock.sendall(bytes(str(publickey),'utf-8'))
print('..llave enviada..')
#direccion = sock.recv(4096).decode('utf-8')
#print('Direccion recibida :' , direccion)

sock.close()
 
