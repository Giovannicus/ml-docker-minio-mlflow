import pymongo

# Connessione a MongoDB
try:
    # Crea la connessione
    client = pymongo.MongoClient("mongodb://52.20.211.97:27117/", serverSelectionTimeoutMS=5000)
    
    # Verifica che la connessione funzioni
    client.admin.command('ping')
    
    # Accedi al database e alla collezione
    db = client["diamond"]
    collection = db["telemetries"]
    
    # Conta i documenti
    count = collection.count_documents({})
    print(f"Connessione riuscita! Numero di documenti nella collezione: {count}")
    
    # Prova a ottenere un documento di esempio
    sample = collection.find_one({})
    if sample:
        print("\nEsempio di documento:")
        for key, value in sample.items():
            print(f"{key}: {value}")
    
    # Chiudi la connessione
    client.close()

except pymongo.errors.ServerSelectionTimeoutError as e:
    print(f"Errore di timeout nella connessione: {e}")
    print("Verifica che l'indirizzo e la porta siano corretti e che il server sia raggiungibile.")
    
except Exception as e:
    print(f"Errore generico: {e}")