from pymongo import MongoClient
from pymongo import errors
import pandas as pd
import numpy as np

# Connessione a MongoDB
try:
    # Crea la connessione
    client = MongoClient(
        "mongodb://diamond:eXpr1viAKCMct@52.20.211.97:27117/diamond?tls=false&authSource=diamond", 
        serverSelectionTimeoutMS=5000)
        
    # Accedi al database e alla collezione
    db = client["diamond"]
    collection = db["telemetries"]
    
    # Estrai tutti i documenti
    data = list(collection.find())
    
    # Converti in DataFrame Pandas
    df = pd.DataFrame(data)
    
    # (Opzionale) Rimuovi la colonna '_id' se non ti serve
    df.drop(columns=["_id"], inplace=True, errors="ignore")

    # Chiudi la connessione
    client.close()

except errors.ServerSelectionTimeoutError as e:
    print(f"Errore di timeout nella connessione: {e}")
    print("Verifica che l'indirizzo e la porta siano corretti e che il server sia raggiungibile.")
    
except Exception as e:
    print(f"Errore generico: {e}")