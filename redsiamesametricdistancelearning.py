"""Autenticación de Evidencia de Audio y Detección de Empalmes entre Dispositivos mediante Aprendizaje de Métricas de Distancia sobre Segmentos de Voz No Vocalizados. Implementar un prototipo de software para la autenticación de evidencia de audio mediante la detección de empalmes cruzados entre dispositivos a partir de segmentos de voz no vocalizados bajo condiciones acústicas controladas basada en un modelo de aprendizaje de métricas de distancia basado en Red Neuronal Siamesa."""


####################################
"""Carga Desde La Base De Datos"""
####################################

#Librerias 
import librosa
import torch
import torchaudio
import os
import random
import numpy as np
from IPython.display import Audio, display
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
from training import Trainer # Importa la clase que pegaste
from sklearn.manifold import TSNE
from pytorch_metric_learning import losses, miners
import matplotlib.gridspec as gridspec
from datetime import datetime
device_to_label = {
    "huawei_nova_9": 0,
    "poco_m4_pro": 1,
    "redmi_note_11": 2,
}
#revisar como se crean las carpetas
base_path = r"C:\BasededatosAutenticaciondeAudios"
normalized_16k_path = os.path.join(base_path, "normalized_data_16k")
#dispositivos que queremos cargar
dispositivos = ["huawei_nova_9", "poco_m4_pro", "redmi_note_11"]

dataset_audios = []

for dispositivo in dispositivos:
    folder_path = os.path.join(normalized_16k_path, dispositivo, "speech_clean")#Se une la ruta de cada dispositvio
    # Verificamos que la carpeta exista para evitar errores
    if os.path.exists(folder_path):
        print(f"Cargando audios de: {dispositivo}...")

        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                file_full_path = os.path.join(folder_path, filename)

                # Cargamos el audio
                audio, sr = librosa.load(file_full_path, sr=None)



                dataset_audios.append({
                    "audio": audio,
                    "sr": sr,
                    "filename": filename,
                    "device": dispositivo,
                    "label": device_to_label[dispositivo]   # etiqueta numérica
                })
    else:
        print(f"No se encontró la carpeta para {dispositivo}")

print(f"\n Total de audios cargados: {len(dataset_audios)}")


#################################
"""Funcion Splicing"""
#################################



def generar_par_splicing(audio_info_a, audio_info_b, top_db=30):
    y1 = audio_info_a['audio']
    y2 = audio_info_b['audio']
    sr = audio_info_a['sr']

    # 1. Detectar intervalos de actividad (voz)
    intervals = librosa.effects.split(y1, top_db=top_db)

    if len(intervals) < 2:
        return None

    # 2. Evitar el primer intervalo si es puro silencio
    # O mejor: elegir un índice de corte que NO sea el extremo inicial.

    # Elegimos un índice entre el primer y el penúltimo intervalo de voz detectado
    # Así garantizamos que ya hubo voz de la persona A antes del corte.
    idx_corte = random.randint(0, len(intervals) - 2)
    punto_corte = intervals[idx_corte][1] # El final de un segmento de habla

    # 3. Crear el empalme
    y_spliced = np.concatenate([y1[:punto_corte], y2[punto_corte:]])

    return {
        "audio": y_spliced,
        "label": 0,
        "source_a": audio_info_a['device'],
        "source_b": audio_info_b['device'],
        "cut_point": punto_corte,
        "cut_time": punto_corte / sr
    }
    

#####################################
"""ejemplo splicing"""
#####################################


# Seleccionamos dos audios al azar
audio_a = dataset_audios[0]  # Un Huawei
audio_b = dataset_audios[50] # Un Poco M4

# Creamos el audio corrupto
resultado = generar_par_splicing(audio_a, audio_b)

if resultado:
    audio_corrupto = resultado['audio']
    punto_corte = resultado['cut_point']

    # Extraer los dos segmentos que la Red Siamesa va a comparar
    # 8000 muestras equivalen a 0.5 segundos (si sr=16000)
    segmento_antes = audio_corrupto[punto_corte - 8000 : punto_corte]
    segmento_despues = audio_corrupto[punto_corte : punto_corte + 8000]

    # Estos dos segmentos son los que entran a la Red Neuronal
    
    

###########################################
"""generar dataset de splicing+original"""
###########################################


def preparar_dataset_splicing_completo(dataset_audios, num_por_dispositivo=15):
    dataset_spliced_completo = []

    # Agrupar audios por dispositivo
    dispositivos = list(set([a['device'] for a in dataset_audios]))
    audios_por_dev = {dev: [a for a in dataset_audios if a['device'] == dev] for dev in dispositivos}

    for dev_a in dispositivos:
        # 1. Seleccionamos 10 audios aleatorios de este dispositivo para corromper
        audios_para_corromper = random.sample(audios_por_dev[dev_a], num_por_dispositivo)

        for audio_info_a in audios_para_corromper:
            # Elegir un dispositivo B diferente
            dev_b = random.choice([d for d in dispositivos if d != dev_a])
            audio_info_b = random.choice(audios_por_dev[dev_b])

            # Aplicamos tu función original
            resultado = generar_par_splicing(audio_info_a, audio_info_b)

            if resultado:
                # Guardamos el diccionario con el audio completo ("y_spliced")
                dataset_spliced_completo.append({
                    "audio": resultado['audio'], # Audio completo empalmado
                    "label": 0,                  # Spliced
                    "device_a": resultado['source_a'],
                    "device_b": resultado['source_b'],
                    "cut_point": resultado['cut_point'],
                    "is_spliced": True
                })

        # 2. También agregamos 10 audios originales (Genuinos) para comparar
        audios_genuinos = random.sample(audios_por_dev[dev_a], num_por_dispositivo)
        for audio_info in audios_genuinos:
            dataset_spliced_completo.append({
                "audio": audio_info['audio'],    # Audio original sin tocar
                "label": 1,                      # Genuino
                "device_a": audio_info['device'],
                "device_b": audio_info['device'],
                "cut_point": None,               # No hay punto de corte real
                "is_spliced": False
            })

    random.shuffle(dataset_spliced_completo)
    return dataset_spliced_completo

# Uso
dataset_audios_completos = preparar_dataset_splicing_completo(dataset_audios)

#from IPython.display import Audio, display

def oir_audio_completo(dataset, idx):
    item = dataset[idx]
    print(f"Index: {idx} | Label: {item['label']} | Dev_A: {item['device_a']} | Dev_B: {item['device_b']}")
    if item['cut_point']:
        print(f"Punto de empalme en muestra: {item['cut_point']}")
        print(item['cut_point']/sr)
    display(Audio(item['audio'], rate=16000))

# Prueba con el primero
oir_audio_completo(dataset_audios_completos, 0)
print(len(dataset_audios_completos))

#####################################
"""Visualizar data set nuevo"""
#####################################

print(dataset_audios_completos[0])
print(dataset_audios_completos[1])


def visualizar_splicing(audio_original, audio_spliced, sr=16000):
    plt.figure(figsize=(12, 8))

    # Espectrograma del Original
    plt.subplot(2, 1, 1)
    S_orig = librosa.feature.melspectrogram(y=audio_original, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S_orig, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
    plt.title('Audio Original (Un solo dispositivo)')
    plt.colorbar(format='%+2.0f dB')

    # Espectrograma del Spliced (Corrupto)
    plt.subplot(2, 1, 2)
    S_splice = librosa.feature.melspectrogram(y=audio_spliced, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S_splice, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
    plt.title('Audio con Splicing (Cambio de Dispositivo)')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.show()


visualizar_splicing(audio_a['audio'], resultado['audio'])


###########################################
"""Enviar a python"""
###########################################


# 1. Guardar el array como archivo .wav en el disco virtual de Colab
nombre_archivo = "audio_corrupto_test.wav"
sf.write(nombre_archivo, resultado['audio'], 16000)

# 2. Descargarlo a tu computadora
#files.download(nombre_archivo) #descargar desde colab


###########################################
"""Sliding windows"""
###########################################


def segmentar_en_ventanas(dataset_completo, win_length_sec=1.0, hop_length_sec=0.5, sr=16000):
    dataset_ventanas = []

    # Convertir segundos a muestras
    win_length = int(win_length_sec * sr)
    hop_length = int(hop_length_sec * sr)

    for item in dataset_completo:
        audio = item['audio']
        cut_point = item['cut_point']

        # Recorrer el audio con la ventana deslizante
        for start in range(0, len(audio) - win_length, hop_length):
            end = start + win_length
            ventana = audio[start:end]
            # 1. EXTRAER EL ESPECTROGRAMA
            feat = extraer_mel_spectrogram(ventana, sr)

            # 2. NORMALIZACIÓN (Importante para Deep Learning)
            # Esto hace que los valores estén en un rango manejable para la red
            feat_norm = (feat - np.mean(feat)) / (np.std(feat) + 1e-8)
            # Determinar si el empalme ocurrió DENTRO de esta ventana
            contiene_empalme = False
            if cut_point is not None:
                if start <= cut_point <= end:
                    contiene_empalme = True

            # Guardamos la metadata de la ventana
            dataset_ventanas.append({
                "mel_spectrogram": feat_norm,
                "ventana_audio": ventana,
                "label_spliced": item['label'], # 0 si el audio base es corrupto
                "is_transition_window": contiene_empalme, # Si es la ventana en especifico en la cual esta el corte
                "device_a": item['device_a'],
                "device_b": item['device_b'] if contiene_empalme or (start > (cut_point or float('inf'))) else item['device_a'],
                "timestamp_sec": start / sr
            })

    return dataset_ventanas
#Extraer espectrograma de mel
def extraer_mel_spectrogram(y, sr=16000):
    # Convertimos el audio de 1s en una "imagen" de 128x32 (aprox)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db
# Aplicar la segmentación
dataset_final_para_entrenar = segmentar_en_ventanas(dataset_audios_completos)

print(f"Total de ventanas generadas: {len(dataset_final_para_entrenar)}")


###########################################
"""Preparar Pares"""
###########################################

class ParesAudioSplicingDataset(Dataset):
    def __init__(self, lista_ventanas, device_to_label):
        self.data = lista_ventanas
        self.device_to_label = device_to_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        spec = torch.tensor(item['mel_spectrogram'], dtype=torch.float32).unsqueeze(0)
        label = self.device_to_label[item['device_b']]
        return spec, torch.tensor(label, dtype=torch.long)

# dataset_final_para_entrenar es la lista que ya creaste con tu función de segmentación
ds = ParesAudioSplicingDataset(dataset_final_para_entrenar,device_to_label)
train_loader = DataLoader(ds, batch_size=32, shuffle=True)

###########################################
"""Print de pares"""
###########################################

specs, labels = next(iter(train_loader))

#print(f"Forma del batch de espectrogramas: {len(specs)}")
# Convierte la lista de tensores en un solo tensor dimensional

print(f"Forma del batch de espectrogramas: {specs.shape}")
# Debería ser: torch.Size([32, 1, 128, 32])

print(f"Etiquetas del batch: {labels}")
# Debería ser un tensor de números entre 0 y 2


###########################################
"""Similarity_masks"""
###########################################


def get_similarity_masks(labels: torch.IntTensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """
    Crea máscaras lógicas para identificar pares positivos y negativos en un batch.

    Args:
        - labels: Tensor 1D con los IDs de los dispositivos (ej: [0, 0, 1, 2])
    Returns:
        - pos_mask: True donde audio i y j son del MISMO dispositivo.
        - neg_mask: True donde audio i y j son de DIFERENTES dispositivos.
    """

    # 1. Creamos la matriz de comparación (Batch x Batch)
    # Compara cada etiqueta con todas las demás etiquetas del lote
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    # 2. Extraemos la parte Triangular Superior (diagonal=1 excluye la diagonal propia)
    # Esto evita comparar el Audio A con el Audio A, y evita duplicar (A con B y B con A)

    # Pares Positivos: Iguales y en la mitad superior
    pos_mask = torch.triu(labels_equal, diagonal=1)

    # Pares Negativos: Diferentes (~ invierte el True/False) y en la mitad superior
    neg_mask = torch.triu(~labels_equal, diagonal=1)

    return pos_mask, neg_mask


###########################################
"""prueba similarity masks"""
###########################################

# Simulamos un batch de 6 ventanas de audio:
# 3 de Huawei (0), 2 de Poco (1), 1 de Redmi (2)
test_labels = torch.tensor([0, 0, 0, 1, 1, 2])

p_mask, n_mask = get_similarity_masks(test_labels)

print("Etiquetas del Batch:", test_labels.tolist())
print(f"\nPares POSITIVOS encontrados: {p_mask.sum().item()}")
print(f"Pares NEGATIVOS encontrados: {n_mask.sum().item()}")

# Visualización rápida en consola (1 es True, 0 es False)
print("\nMatriz de Pares Positivos (Mismo Celular):")
print(p_mask.int())
# 2. Definir la función de visualización de pares (la de la referencia)
def mask2pairs(mask, labels):
    # Esta función agrupa los pares (i, j) según el ID del dispositivo (c)
    return [[(i, j) for i in range(len(labels)) for j in range(len(labels))
             if mask[i,j] and labels[i]==c]
            for c in range(int(max(labels)) + 1)]

# PRUEBA CON TUS DATOS
# Etiquetas: 3 de Huawei (0), 2 de Poco (1), 1 de Redmi (2)
labels_batch = torch.tensor([0, 0, 0, 1, 1, 2])

# Generamos las máscaras
pos_mask, neg_mask = get_similarity_masks(labels_batch)

# Imprimimos los resultados
print('DESGLOSE DE PARES PARA EL BATCH')
print(f"Etiquetas: {labels_batch.tolist()}")
print()

print('Pares Positivos (Mismo dispositivo, agrupados por ID):')
# Usamos * para despaquetar la lista de listas
print(*mask2pairs(pos_mask, labels_batch), sep='\n')

print('\nPares Negativos (Dispositivos diferentes, agrupados por ID):')
print(*mask2pairs(neg_mask, labels_batch), sep='\n')


###########################################
"""Visualizacion de pares como heatmap"""
###########################################



def visualizar_logica_pares(labels):
    # Generamos las máscaras con la función que tienes de referencia
    pos_mask, neg_mask = get_similarity_masks(labels)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Visualizar Pares Positivos (Mismo dispositivo)
    sns.heatmap(pos_mask.cpu().numpy(), ax=ax[0], cbar=False, cmap="Greens")
    ax[0].set_title("Máscara de Pares POSITIVOS\n(Deben estar cerca)")

    # Visualizar Pares Negativos (Dispositivos diferentes)
    sns.heatmap(neg_mask.cpu().numpy(), ax=ax[1], cbar=False, cmap="Reds")
    ax[1].set_title("Máscara de Pares NEGATIVOS\n(Deben estar lejos)")

    plt.show()

# Ejemplo con un batch de tu loader
specs, labels = next(iter(train_loader))
visualizar_logica_pares(labels)

###########################################
"""Loss Module"""
###########################################



#class ContrastiveLoss(nn.Module):
#    def __init__(self, margin: float = 1.0):
#        super().__init__()
#        self.margin = margin

#    def forward(self, outputs, label_par):
#        emb_a, emb_b = outputs           # desempaca lo que devuelve EmbeddingNet
#        dist = F.pairwise_distance(emb_a, emb_b, p=2)

        # label 0 = mismo dispositivo → minimizar distancia
        # label 1 = diferente dispositivo → empujar más allá del margen
#        positive_loss = (1 - label_par) * dist.pow(2)
#        negative_loss = label_par * F.relu(self.margin - dist).pow(2)

#        return (positive_loss + negative_loss).mean()
    

###########################################
"""Training the network
Separar Dataset en entrenamiento y prueba"""
###########################################


# 1. División de los datos originales
train_data, val_data = train_test_split(
    dataset_final_para_entrenar,
    test_size=0.2,
    random_state=42,
    stratify=[d['device_b'] for d in dataset_final_para_entrenar]
)

# 2. Creación de los Datasets (¡IMPORTANTE añadir device_to_label!)
# Sin esto, el Dataset no sabrá que Huawei es 0, Poco es 1, etc.
train_dataset = ParesAudioSplicingDataset(train_data, device_to_label)
val_dataset   = ParesAudioSplicingDataset(val_data,   device_to_label)


# 3. DataLoaders listos para el Trainer
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

print(f" Configuración exitosa:")
print(f"   - Entrenamiento: {len(train_dataset)} ventanas")
print(f"   - Validación: {len(val_dataset)} ventanas")

###########################################
"""Prueba"""
###########################################

# 3. Prueba de fuego (Corregida para listas)
batch_ejemplo = next(iter(train_loader))

# En PyTorch, si el dataset devuelve (x, y), el batch es una lista [tensor_x, tensor_y]
espectrogramas, etiquetas = batch_ejemplo
# Convierte la lista de tensores en un único tensor de 4D

print(f" Batch cargado correctamente:")
print(f"   - Tamaño de los espectrogramas: {espectrogramas.shape}") # Debería ser [32, 1, 128, 32]
print(f"   - Tamaño de las etiquetas: {etiquetas.shape}")         # Debería ser [32]
print(f"   - Ejemplo de etiquetas en este batch: {etiquetas[:5]}")

###########################################
"""Embeddingnet Backbone+Neck+Embeddingnet"""
###########################################




class Backbone(nn.Sequential):
    def __init__(self):
        super().__init__(
            # Entrada: [1, 128, 31-32] (Mel x Tiempo)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> [64, 64, 16]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> [128, 32, 8]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> [256, 16, 4]
        )

class Neck(nn.Sequential):
    def __init__(self, out_dim):
        super().__init__(
            nn.Flatten(),
            # AJUSTE AQUÍ: 256 canales * 16 de alto * 4 de ancho = 16384
            nn.Linear(256 * 16 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

class EmbeddingNet(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck(out_dim)

    def forward(self, x):
        return F.normalize(self.neck(self.backbone(x)), p=2, dim=1)
    

###########################################
""" Trainer de training.py"""
###########################################




# 1. Parámetros
OUT_DIM = 64
MARGIN = 1.0
EPOCHS = 10
LR = 0.001

# 2. Instanciar Componentes
model = EmbeddingNet(out_dim=OUT_DIM)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)#Atenua los picos con respecto al Adam normal
miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard")
loss_fn = losses.TripletMarginLoss(margin=0.2)
# 3. Preparar Datos (usando tus loaders previos)
# Asegúrate de que train_loader devuelva (spec, label)
# tal como lo configuramos en ParesAudioSplicingDataset

# 4. Iniciar el Trainer
def pml_adapter(model, batch, func):
    specs, labels = batch
    embeddings = model(specs)
    hard_pairs = miner(embeddings, labels)   # mina los tripletas difíciles
    return func(embeddings, labels, hard_pairs)

trainer = Trainer()
trainer.set_adapter(pml_adapter)

# Si tienes GPU en Colab, esto lo moverá automáticamente
history = trainer.fit(
    model=model,
    loader=train_loader,
    valid_loader=val_loader, # Opcional
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=EPOCHS
)


###########################################
"""Vissualizacion del loss"""
###########################################



plt.plot(history['train_loss'], label='train loss')
plt.legend()
plt.show()

###########################################
"""Evaluacion con knn"""
###########################################
# Loader auxiliar solo para extraer embeddings con label de dispositivo
class EmbeddingExtractionDataset(Dataset):
    def __init__(self, lista_ventanas, device_to_label):
        self.data = lista_ventanas
        self.device_to_label = device_to_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        spec = torch.tensor(item['mel_spectrogram'], dtype=torch.float32).unsqueeze(0)
        label = self.device_to_label[item['device_b']]
        return spec, torch.tensor(label, dtype=torch.long)

train_emb_loader = DataLoader(EmbeddingExtractionDataset(train_data, device_to_label), batch_size=32)
val_emb_loader   = DataLoader(EmbeddingExtractionDataset(val_data,   device_to_label), batch_size=32)

#def extract_embeddings(model, loader, device):
#    model.eval()
#    embeddings = []
#    labels = []
#
#    with torch.no_grad():
#        for specs, target_labels in loader:
#            # Enviamos a la GPU/CPU según corresponda
#            specs = specs.to(device)
#            # Obtenemos el vector de características (huella digital)
#            output = model(specs)#
#
#            embeddings.append(output.cpu())
#            labels.append(target_labels)
#
#    # Concatenamos todo en un solo tensor
#    return torch.cat(embeddings), torch.cat(labels)
def extract_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            specs, target_labels = batch

            # Quitar el torch.stack, specs ya es un tensor
            specs = specs.to(device)
            output = model(specs)

            embeddings.append(output.cpu())
            labels.append(target_labels)

    return torch.cat(embeddings), torch.cat(labels)
# Extraemos los datos de tus loaders
device = trainer._device # El dispositivo que usó tu trainer
model.to(device)
train_embs, train_labs = extract_embeddings(model, train_emb_loader, device)
val_embs,   val_labs   = extract_embeddings(model, val_emb_loader,   device)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Definir el clasificador (k=5 es un buen estándar)
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")

# 2. Entrenar el k-NN con los embeddings del set de entrenamiento
knn.fit(train_embs.numpy(), train_labs.numpy())

# 3. Predecir las etiquetas del set de validación (test)
preds = knn.predict(val_embs.numpy())

# 4. Calcular la precisión
knn_accuracy = accuracy_score(val_labs.numpy(), preds)


print(f" Precisión del k-NN: {knn_accuracy:.2%}")



###########################################
"""Visulaizacion del training"""
###########################################




# 1. Extraer embeddings de validación
#model.eval()
#embs, targets = [], []
#with torch.no_grad():
#    for specs, labels in val_loader:
#        output = model(specs.to(trainer._device))
#        embs.append(output.cpu())
#        targets.append(labels)

#embs = torch.cat(embs).numpy()
#targets = torch.cat(targets).numpy()
model.eval()
embs, targets = [], []
with torch.no_grad():
    for specs, labels in val_emb_loader:   # ← usa el loader auxiliar
        specs = specs.to(trainer._device)
        output = model(specs)              # devuelve un solo embedding
        embs.append(output.cpu())
        targets.append(labels)

embs = torch.cat(embs).numpy()
targets = torch.cat(targets).numpy()
# 2. Reducir a 2D
tsne = TSNE(n_components=2, random_state=42)
embs_2d = tsne.fit_transform(embs)

# 3. Graficar
plt.figure(figsize=(10, 6))
for dev, idx in device_to_label.items():
    mask = targets == idx
    plt.scatter(embs_2d[mask, 0], embs_2d[mask, 1], label=dev, alpha=0.6)
plt.legend()
plt.title("Separación de Dispositivos en el Espacio de Embeddings")
plt.show()




# Invertimos el diccionario para obtener el nombre desde el número
# Resultado esperado: {0: 'huawei_nova_9', 1: 'poco_m4_pro', 2: 'redmi_note_11'}
id_to_device = {v: k for k, v in device_to_label.items()}
target_names = [id_to_device[i] for i in range(len(device_to_label))]
# 1. Configurar t-SNE (reducir de 64 dimensiones a 2 para la gráfica)
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')

# 2. Transformar los embeddings (esto puede tardar unos segundos)
train_tsne = tsne.fit_transform(train_embs.numpy())
val_tsne = tsne.fit_transform(val_embs.numpy())

# 3. Crear la visualización
plt.figure(figsize=(14, 6))
plt.suptitle("Visualización t-SNE de Huellas Digitales por Dispositivo", fontsize=16)

# Gráfica de Entrenamiento
plt.subplot(1, 2, 1)
scatter_train = plt.scatter(train_tsne[:, 0], train_tsne[:, 1],  c=train_labs.numpy(), cmap="Set1", alpha=0.7)
plt.legend(handles=scatter_train.legend_elements()[0], labels=target_names, title="Dispositivos")
plt.title("Conjunto de Entrenamiento")
plt.grid(True, linestyle='--', alpha=0.5)

# Gráfica de Validación (Test)
plt.subplot(1, 2, 2)
scatter_val = plt.scatter(val_tsne[:, 0], val_tsne[:, 1],  c=val_labs.numpy(), cmap="Set1", alpha=0.7)
plt.legend(handles=scatter_val.legend_elements()[0], labels=target_names, title="Dispositivos")
plt.title("Conjunto de Validación")
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()




# Invertimos el diccionario para obtener el nombre desde el número
id_to_device = {v: k for k, v in device_to_label.items()}
target_names = [id_to_device[i] for i in range(len(device_to_label))]

# 1. JUNTAMOS LOS EMBEDDINGS para que t-SNE los proyecte en el mismo mapa
all_embs = np.concatenate([train_embs.numpy(), val_embs.numpy()], axis=0)
n_train = len(train_embs)

# 2. Configurar t-SNE y transformar TODO JUNTO
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
all_tsne = tsne.fit_transform(all_embs)

# 3. Volvemos a separar las coordenadas para graficar
train_tsne = all_tsne[:n_train]
val_tsne = all_tsne[n_train:]

# 4. Crear la visualización
plt.figure(figsize=(14, 6))
plt.suptitle("Visualización t-SNE de Huellas Digitales por Dispositivo", fontsize=16)

# Gráfica de Entrenamiento
plt.subplot(1, 2, 1)
scatter_train = plt.scatter(train_tsne[:, 0], train_tsne[:, 1], c=train_labs.numpy(), cmap="Set1", alpha=0.7)
plt.legend(handles=scatter_train.legend_elements()[0], labels=target_names, title="Dispositivos")
plt.title("Conjunto de Entrenamiento")
plt.grid(True, linestyle='--', alpha=0.5)

# Gráfica de Validación (Test)
plt.subplot(1, 2, 2)
scatter_val = plt.scatter(val_tsne[:, 0], val_tsne[:, 1], c=val_labs.numpy(), cmap="Set1", alpha=0.7)
plt.legend(handles=scatter_val.legend_elements()[0], labels=target_names, title="Dispositivos")
plt.title("Conjunto de Validación")
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

###########################################
"""Distancia euclidiana"""
###########################################

def detectar_punto_empalme(embeddings, timestamps, threshold=0.6):
    """
    Analiza la distancia entre ventanas consecutivas para hallar el salto.
    """
    distancias = []
    punto_detectado = None

    for i in range(1, len(embeddings)):
        # Distancia entre el embedding de la ventana actual y la anterior
        d = np.linalg.norm(embeddings[i] - embeddings[i-1])
        distancias.append(d)

        # Si la distancia supera el umbral, marcamos el segundo
        if d > threshold and punto_detectado is None:
            punto_detectado = timestamps[i]

    return punto_detectado, distancias

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from datetime import datetime

def generar_reporte_forense(audio_sospechoso, modelo, device, sr=16000,
                             win_sec=1.0, hop_sec=0.5, threshold=0.6):
    """
    Dado un audio nuevo, genera un reporte forense completo con:
    1. Gráfica de distancias en el tiempo
    2. Mapa de calor de similitud entre segmentos
    3. t-SNE del audio comparado contra audios de referencia conocidos
    """

    # ── 1. Segmentar el audio sospechoso en ventanas ──────────────────────────
    win_len = int(win_sec * sr)
    hop_len = int(hop_sec * sr)
    ventanas, timestamps = [], []

    for start in range(0, len(audio_sospechoso) - win_len, hop_len):
        ventana = audio_sospechoso[start:start + win_len]
        mel = librosa.feature.melspectrogram(y=ventana, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)
        ventanas.append(mel_norm)
        timestamps.append(start / sr)

    # ── 2. Extraer embeddings ─────────────────────────────────────────────────
    modelo.eval()
    specs = torch.tensor(np.array(ventanas), dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        embs = modelo(specs).cpu().numpy()

    # ── 3. Calcular distancias entre ventanas consecutivas ────────────────────
    distancias = [np.linalg.norm(embs[i] - embs[i-1]) for i in range(1, len(embs))]
    timestamps_dist = timestamps[1:]

    # Detectar punto de empalme
    idx_max = int(np.argmax(distancias))
    tiempo_empalme = timestamps_dist[idx_max]
    hay_empalme = distancias[idx_max] > threshold

    # ── 4. Matriz de similitud entre todos los segmentos ─────────────────────
    n = len(embs)
    matriz_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matriz_sim[i, j] = np.linalg.norm(embs[i] - embs[j])

    # ── 5. Construir el reporte visual ────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"REPORTE FORENSE DE AUTENTICIDAD DE AUDIO\n"
        f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=14, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: Forma de onda con marca del empalme ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :])  # ocupa todo el ancho
    tiempo_audio = np.linspace(0, len(audio_sospechoso) / sr, len(audio_sospechoso))
    ax1.plot(tiempo_audio, audio_sospechoso, color='steelblue', linewidth=0.5, alpha=0.8)
    if hay_empalme:
        ax1.axvline(x=tiempo_empalme, color='red', linewidth=2, linestyle='--',
                    label=f'Empalme detectado: {tiempo_empalme:.2f}s')
        ax1.fill_betweenx(
            [audio_sospechoso.min(), audio_sospechoso.max()],
            tiempo_empalme - hop_sec, tiempo_empalme + hop_sec,
            alpha=0.2, color='red'
        )
    ax1.set_title('Forma de Onda — Señal de Audio Analizada', fontweight='bold')
    ax1.set_xlabel('Tiempo (segundos)')
    ax1.set_ylabel('Amplitud')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Distancias en el tiempo ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(timestamps_dist, distancias, color='darkorange', linewidth=1.5, marker='o',
             markersize=3, label='Distancia entre segmentos')
    ax2.axhline(y=threshold, color='gray', linestyle=':', linewidth=1.5,
                label=f'Umbral = {threshold}')
    if hay_empalme:
        ax2.axvline(x=tiempo_empalme, color='red', linewidth=2, linestyle='--',
                    label=f'Pico máximo: {tiempo_empalme:.2f}s')
        ax2.scatter([tiempo_empalme], [distancias[idx_max]],
                    color='red', s=100, zorder=5)
    ax2.set_title('Distancia Euclidiana entre Segmentos Consecutivos', fontweight='bold')
    ax2.set_xlabel('Tiempo (segundos)')
    ax2.set_ylabel('Distancia en espacio de embeddings')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Mapa de calor de similitud ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    im = ax3.imshow(matriz_sim, cmap='hot_r', aspect='auto',
                    extent=[timestamps[0], timestamps[-1],
                            timestamps[-1], timestamps[0]])
    plt.colorbar(im, ax=ax3, label='Distancia euclidiana')
    if hay_empalme:
        ax3.axvline(x=tiempo_empalme, color='cyan', linewidth=1.5, linestyle='--',
                    label=f'Empalme: {tiempo_empalme:.2f}s')
        ax3.axhline(y=tiempo_empalme, color='cyan', linewidth=1.5, linestyle='--')
    ax3.set_title('Mapa de Similitud entre Segmentos\n(más oscuro = más similares)',
                  fontweight='bold')
    ax3.set_xlabel('Tiempo (segundos)')
    ax3.set_ylabel('Tiempo (segundos)')
    ax3.legend(fontsize=9)

    # ── Veredicto textual ─────────────────────────────────────────────────────
    if hay_empalme:
        veredicto = (f"⚠ EMPALME DETECTADO en t = {tiempo_empalme:.2f}s  "
                     f"(distancia = {distancias[idx_max]:.4f}, umbral = {threshold})")
        color_veredicto = 'red'
    else:
        veredicto = f"✔ AUDIO AUTÉNTICO — distancia máxima {max(distancias):.4f} < umbral {threshold}"
        color_veredicto = 'green'

    fig.text(0.5, 0.01, veredicto, ha='center', fontsize=13,
             fontweight='bold', color=color_veredicto,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor=color_veredicto))

    plt.savefig("reporte_forense.pdf", bbox_inches='tight', dpi=150)
    plt.show()

    return {
        "hay_empalme": hay_empalme,
        "tiempo_empalme_seg": tiempo_empalme if hay_empalme else None,
        "distancia_maxima": float(distancias[idx_max]),
        "threshold_usado": threshold,
        "embeddings": embs,
        "timestamps": timestamps,
        "distancias": distancias
    }
# audio_nuevo es cualquier array numpy cargado con librosa
#resultado = generar_reporte_forense(
#    audio_sospechoso=audio_nuevo,
#    modelo=model,
#    device=trainer._device,
#    threshold=0.6  # ajusta según tus resultados de validación
#)