from pathlib import Path
from tinyec import registry
from Crypto.Cipher import AES
import hashlib, secrets, binascii , time, bcrypt , os, sys, socket, json
CONST_PATHFILE = os.path.realpath("__file__")

curve = registry.get_curve('brainpoolP256r1')

def encrypt_AES_GCM(msg, secretKey):
    aesCipher = AES.new(secretKey, AES.MODE_GCM)
    ciphertext, authTag = aesCipher.encrypt_and_digest(msg)
    return (ciphertext, aesCipher.nonce, authTag)
def encrypt_ECC(msg, pubKey):
    ciphertextPrivKey = secrets.randbelow(curve.field.n)
    sharedECCKey = ciphertextPrivKey * pubKey
    secretKey = ecc_point_to_256_bit_key(sharedECCKey)
    ciphertext, nonce, authTag = encrypt_AES_GCM(msg, secretKey)
    ciphertextPubKey = ciphertextPrivKey * curve.g
    return (ciphertext, nonce, authTag, ciphertextPubKey)

def ecc_point_to_256_bit_key(point):
    sha = hashlib.sha256(int.to_bytes(point.x, 32, 'big'))
    sha.update(int.to_bytes(point.y, 32, 'big'))
    return sha.digest()    
def getscriptpath():

    filename = os.path.basename("__file__")
    filepath = CONST_PATHFILE.replace(filename, "")
    filepath = filepath[:-1]
    return filepath


def cracker(archivo, diccionario, output, hash_type) :
    #Direcciones : 
    currentpath =  getscriptpath()
    archivo_path = currentpath + "/archivos/" + archivo
    diccionario_path = currentpath + "/diccionarios/" + diccionario
    output_path = currentpath +"/textoPlano/" + output

    #sentencia del comando 
    command =' hashcat -a 0 -m ' +  str(hash_type) + ' --outfile='+ output_path+' '+ archivo_path + ' ' +diccionario_path + ' --force '  
    #command =' hashcat -a 0 -m ' +  str(hash_type)+' '+ archivo_path + ' ' +diccionario_path + ' --force '  
    
    os.chdir(currentpath + "/hashcat")
    
    #se ejecuta el comando 
    start = time.time()
    os.system(command)
    final = time.time()
    print("Tiempo en crackear" , archivo , ":",final-start)

def output() : 

    currentpath =  getscriptpath()
    output_path =  currentpath + "/textoPlano/"
    texto_plano = open(currentpath + "/outputs/lista", 'w') 
    with os.scandir(output_path) as files : 
        files = [archivo.name for archivo in files if archivo.is_file()]    
    
    for file in files : 
        file_path = output_path + file
        file_= open(file_path , 'r')
        for line in file_ :
            line = line.strip()
            pc = list(line.split(':'))[-1]
            texto_plano.write(pc+'\n')
        file_.close()
    texto_plano.close()
    

def hash(): 
    salt = bcrypt.gensalt()
    currentpath = getscriptpath()
    texto_plano_path =currentpath + "/outputs/lista"
    hash_path = currentpath + "/outputs/hash_lista"
    start = time.time()
    if texto_plano_path:
        lista = open(texto_plano_path , 'r') 
        hash_list = open(hash_path , 'w')
        for line in lista : 
            line = line.strip()
            hash_p = bcrypt.hashpw(line.encode(encoding='UTF-8'), salt)
            hash_list.write(hash_p.decode('UTF-8')+'\n')
        lista.close()    
    final = time.time() 
    print(" Tiempo de hasheo : ", final -start)   
    
def public_key(pbkey): 
    a = pbkey.split(',')
    ellie = int(a[2][:-1])
    return ellie * curve.g   

def conectar(): 
    # Crea el socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # vincula socket con el puerto
    server_address = ('localhost', 10000)
    sock.bind(server_address)
    print("...Esperando llave...")
    # escucha las conecciones
    sock.listen(1) 
    connection, client_address = sock.accept()
    data = connection.recv(4096).decode('utf-8')

    print('Llave recibida : ', data )
    por_cifrar= b'straykids'
    pbkey = public_key(data)

    currentpath = getscriptpath()
    os.chdir(currentpath + "/outputs")
    file = "salida"
    file_open= open( file, "w")
    encryptedMsg = encrypt_ECC(por_cifrar, pbkey)
    print(encryptedMsg)


    
    connection.close()
    sock.close()     

def opciones():
    while True : 
        print('''
            
        ->1<- Crackear archivos    
        ->2<- Hashear contraseñas en texto plano
        ->3<- Encriptar y desencriptar mediante sockets
        ->4<-  Salir
        ->5<-- Generando llave
            ''')
        
        op = input()

        if op == '1' :

            #cracker('archivo_1', 'diccionario_2.dict','Archivo_1',0)    
            #cracker('archivo_2', 'diccionario_2.dict','Archivo_2',10)
            #cracker('archivo_3', 'diccionario_2.dict','Archivo_3',10)
            #cracker('archivo_4', 'diccionario_2.dict','Archivo_4',1000)
            #cracker('archivo_5', 'diccionario_2.dict','Archivo_5',1800)
            output()
        elif op == '2':  
            hash()
        elif op =='3' : 
            conectar()
        elif op == '4' :
            exit()
        elif op == '5' :
            print('opcion de prueba')

if __name__ == "__main__":
    opciones()

