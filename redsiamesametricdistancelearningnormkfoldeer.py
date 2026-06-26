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
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from eer_evaluation import evaluar_eer_completo
device_to_label = {
    "huawei_nova_9": 0,
    "poco_m4_pro": 1,
    "redmi_note_11": 2,
}
# 1. Parámetros
OUT_DIM = 64
MARGIN = 1.0
EPOCHS = 3
LR = 0.001
n_splits = 2
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
    
######
"""Separar TEST"""    
#####
# Separar audios crudos ANTES de generar el dataset


# Separar por dispositivo para que haya representación de todos
test_audios = []
train_audios_crudos = []

for dev in dispositivos:
    audios_dev = [a for a in dataset_audios if a['device'] == dev]
    # 10% para test, 90% para entrenar
    train_dev, test_dev = train_test_split(audios_dev, test_size=0.1, random_state=42)
    test_audios.extend(test_dev)
    train_audios_crudos.extend(train_dev)



###########################################
"""generar dataset de splicing+original"""
###########################################


# def preparar_dataset_splicing_completo(dataset_audios, num_por_dispositivo=15):
#     dataset_spliced_completo = []

#     # Agrupar audios por dispositivo
#     dispositivos = list(set([a['device'] for a in dataset_audios]))
#     audios_por_dev = {dev: [a for a in dataset_audios if a['device'] == dev] for dev in dispositivos}

#     for dev_a in dispositivos:
#         # 1. Seleccionamos 10 audios aleatorios de este dispositivo para corromper
#         audios_para_corromper = random.sample(audios_por_dev[dev_a], num_por_dispositivo)

#         for audio_info_a in audios_para_corromper:
#             # Elegir un dispositivo B diferente
#             dev_b = random.choice([d for d in dispositivos if d != dev_a])
#             audio_info_b = random.choice(audios_por_dev[dev_b])

#             # Aplicamos tu función original
#             resultado = generar_par_splicing(audio_info_a, audio_info_b)

#             if resultado:
#                 # Guardamos el diccionario con el audio completo ("y_spliced")
#                 dataset_spliced_completo.append({
#                     "audio": resultado['audio'], # Audio completo empalmado
#                     "label": 0,                  # Spliced
#                     "device_a": resultado['source_a'],
#                     "device_b": resultado['source_b'],
#                     "cut_point": resultado['cut_point'],
#                     "is_spliced": True
#                 })

#         # 2. También agregamos 10 audios originales (Genuinos) para comparar
#         audios_genuinos = random.sample(audios_por_dev[dev_a], num_por_dispositivo)
#         for audio_info in audios_genuinos:
#             dataset_spliced_completo.append({
#                 "audio": audio_info['audio'],    # Audio original sin tocar
#                 "label": 1,                      # Genuino
#                 "device_a": audio_info['device'],
#                 "device_b": audio_info['device'],
#                 "cut_point": None,               # No hay punto de corte real
#                 "is_spliced": False
#             })

#     random.shuffle(dataset_spliced_completo)
#     return 
def preparar_dataset_splicing_completo(dataset_audios, num_por_dispositivo=15, min_viable=3):
    """
    num_por_dispositivo: cuántos audios usar por dispositivo (ideal)
    min_viable: mínimo aceptable por fold; si no se alcanza, lanza warning
    """
    dataset_spliced_completo = []

    dispositivos = list(set([a['device'] for a in dataset_audios]))
    audios_por_dev = {dev: [a for a in dataset_audios if a['device'] == dev] for dev in dispositivos}

    for dev_a in dispositivos:
        audios_disponibles = len(audios_por_dev[dev_a])

        if audios_disponibles < min_viable:
            print(f"ADVERTENCIA: {dev_a} solo tiene {audios_disponibles} audios "
                  f"(mínimo recomendado: {min_viable}). "
                  f"Agrega más datos para este dispositivo.")
            if audios_disponibles == 0:
                continue

        # Se adapta al disponible pero respeta num_por_dispositivo cuando hay suficientes
        objetivo = audios_disponibles if num_por_dispositivo is None else num_por_dispositivo
        n = min(objetivo, audios_disponibles)

        audios_para_corromper = random.sample(audios_por_dev[dev_a], n)

        for audio_info_a in audios_para_corromper:
            dev_b = random.choice([d for d in dispositivos if d != dev_a])
            audio_info_b = random.choice(audios_por_dev[dev_b])
            resultado = generar_par_splicing(audio_info_a, audio_info_b)

            if resultado:
                dataset_spliced_completo.append({
                    "audio":      resultado['audio'],
                    "label":      0,
                    "device_a":   resultado['source_a'],
                    "device_b":   resultado['source_b'],
                    "cut_point":  resultado['cut_point'],
                    "is_spliced": True
                })

        audios_genuinos = random.sample(audios_por_dev[dev_a], n)
        for audio_info in audios_genuinos:
            dataset_spliced_completo.append({
                "audio":      audio_info['audio'],
                "label":      1,
                "device_a":   audio_info['device'],
                "device_b":   audio_info['device'],
                "cut_point":  None,
                "is_spliced": False
            })

    random.shuffle(dataset_spliced_completo)
    return dataset_spliced_completo
# Uso
#dataset_audios_completos = preparar_dataset_splicing_completo(dataset_audios)
# A partir de aquí todo el pipeline usa train_audios_crudos en vez de dataset_audios
dataset_audios_completos = preparar_dataset_splicing_completo(train_audios_crudos)
#f

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
    def __init__(self, lista_ventanas, device_to_label, mean_global=None, std_global=None):
        self.data = lista_ventanas
        self.device_to_label = device_to_label
        self.mean_global = mean_global
        self.std_global  = std_global

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        spec = torch.tensor(item['mel_spectrogram'], dtype=torch.float32).unsqueeze(0)

        if self.mean_global is not None and self.std_global is not None:
            spec = (spec - self.mean_global) / (self.std_global + 1e-8)

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
###
"""Embedding Extraction Dataset"""
###
# Loader auxiliar solo para extraer embeddings con label de dispositivo
class EmbeddingExtractionDataset(Dataset):
    def __init__(self, lista_ventanas, device_to_label, mean_global=None, std_global=None):
        self.data = lista_ventanas
        self.device_to_label = device_to_label
        self.mean_global = mean_global
        self.std_global  = std_global

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        spec = torch.tensor(item['mel_spectrogram'], dtype=torch.float32).unsqueeze(0)

        if self.mean_global is not None and self.std_global is not None:
            spec = (spec - self.mean_global) / (self.std_global + 1e-8)

        label = self.device_to_label[item['device_b']]
        return spec, torch.tensor(label, dtype=torch.long)
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
"""Training con K-Fold + Modelo Final"""
###########################################


def entrenar_con_kfold(train_audios_crudos, device_to_label,
                        n_splits=5, OUT_DIM=64, EPOCHS=10, LR=0.001):
    """
    K-Fold correcto: el split se hace sobre audios CRUDOS (no ventanas)
    para evitar leakage entre ventanas del mismo archivo.
    """
    kf      = GroupKFold(n_splits=n_splits)
    indices = np.arange(len(train_audios_crudos))

    fold_histories = []
    fold_metrics   = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices, groups=indices)):
        print(f"\n{'='*50}")
        print(f"  FOLD {fold+1}/{n_splits}")
        print(f"{'='*50}")

        # ── Split sobre audios crudos ──────────────────────────────────────
        audios_train_fold = [train_audios_crudos[i] for i in train_idx]
        audios_val_fold   = [train_audios_crudos[i] for i in val_idx]

        # ── Generar dataset y ventanas DENTRO del fold ─────────────────────
        ds_train_fold = preparar_dataset_splicing_completo(audios_train_fold, num_por_dispositivo=None)
        ds_val_fold   = preparar_dataset_splicing_completo(audios_val_fold,   num_por_dispositivo=None)
        ventanas_train = segmentar_en_ventanas(ds_train_fold)
        ventanas_val   = segmentar_en_ventanas(ds_val_fold)

        # ── Normalización calculada SOLO sobre train del fold ──────────────
        feats_train = np.array([v['mel_spectrogram'] for v in ventanas_train])
        mean_fold   = float(feats_train.mean())
        std_fold    = float(feats_train.std())

        # ── Datasets y Loaders ─────────────────────────────────────────────
        train_ds = ParesAudioSplicingDataset(ventanas_train, device_to_label, mean_fold, std_fold)
        val_ds   = ParesAudioSplicingDataset(ventanas_val,   device_to_label, mean_fold, std_fold)

        train_loader_fold = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader_fold   = DataLoader(val_ds,   batch_size=32, shuffle=False)

        # ── Modelo fresco por fold ─────────────────────────────────────────
        model_fold     = EmbeddingNet(out_dim=OUT_DIM)
        optimizer_fold = torch.optim.AdamW(model_fold.parameters(), lr=LR)
        miner_fold     = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard")
        loss_fn_fold   = losses.TripletMarginLoss(margin=0.2)

        def pml_adapter(model, batch, func):
            specs, labels = batch
            embeddings    = model(specs)
            hard_pairs    = miner_fold(embeddings, labels)
            return func(embeddings, labels, hard_pairs)

        trainer_fold = Trainer()
        trainer_fold.set_adapter(pml_adapter)

        history = trainer_fold.fit(
            model=model_fold,
            loader=train_loader_fold,
            valid_loader=val_loader_fold,
            loss_fn=loss_fn_fold,
            optimizer=optimizer_fold,
            epochs=EPOCHS
        )

        # ── Evaluación k-NN ────────────────────────────────────────────────
        device = trainer_fold._device
        model_fold.to(device)

        train_emb_loader = DataLoader(
            EmbeddingExtractionDataset(ventanas_train, device_to_label, mean_fold, std_fold),
            batch_size=32
        )
        val_emb_loader = DataLoader(
            EmbeddingExtractionDataset(ventanas_val, device_to_label, mean_fold, std_fold),
            batch_size=32
        )

        train_embs, train_labs = extract_embeddings(model_fold, train_emb_loader, device)
        val_embs,   val_labs   = extract_embeddings(model_fold, val_emb_loader,   device)

        knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        knn.fit(train_embs.numpy(), train_labs.numpy())
        preds   = knn.predict(val_embs.numpy())
        knn_acc = accuracy_score(val_labs.numpy(), preds)

        print(f"Fold {fold+1} — kNN Accuracy: {knn_acc:.2%}")

        fold_histories.append(history)
        fold_metrics.append({
            'fold':           fold + 1,
            'knn_accuracy':   knn_acc,
            'final_val_loss': history['valid_loss'][-1],
            'mean_fold':      mean_fold,
            'std_fold':       std_fold,
        })

    return fold_histories, fold_metrics


###########################################
"""Modelo Final — entrenado con TODOS los datos"""
###########################################
# Después del K-Fold, entrenamos el modelo definitivo con todo train_audios_crudos.
# Los hiperparámetros ya fueron validados por K-Fold, aquí no hay val set.

def entrenar_modelo_final(train_audios_crudos, device_to_label,
                           OUT_DIM=64, EPOCHS=10, LR=0.001):
    """
    Entrena el modelo definitivo usando TODOS los audios de entrenamiento.
    La normalización global se calcula aquí y se guarda junto al modelo.
    """
    # Generar dataset completo sin reservar validación
    ds_completo    = preparar_dataset_splicing_completo(train_audios_crudos)
    ventanas_todas = segmentar_en_ventanas(ds_completo)

    # Normalización global (se guarda para inferencia)
    feats      = np.array([v['mel_spectrogram'] for v in ventanas_todas])
    mean_global = float(feats.mean())
    std_global  = float(feats.std())
    print(f"Media global: {mean_global:.4f} | Std global: {std_global:.4f}")

    train_ds     = ParesAudioSplicingDataset(ventanas_todas, device_to_label, mean_global, std_global)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model     = EmbeddingNet(out_dim=OUT_DIM)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    miner     = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard")
    loss_fn   = losses.TripletMarginLoss(margin=0.2)

    def pml_adapter(model, batch, func):
        specs, labels = batch
        embeddings    = model(specs)
        hard_pairs    = miner(embeddings, labels)
        return func(embeddings, labels, hard_pairs)

    trainer = Trainer()
    trainer.set_adapter(pml_adapter)

    # Sin valid_loader porque usamos todos los datos
    history = trainer.fit(
        model=model,
        loader=train_loader,
        valid_loader=None,       # ← intencional
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=EPOCHS
    )

    return model, history, mean_global, std_global, trainer._device

####
"""Visualizacion con kfold"""
####
def visualizar_resultados_kfold(fold_histories, fold_metrics):
    n_folds = len(fold_metrics)
    accs    = [m['knn_accuracy']    for m in fold_metrics]
    losses  = [m['final_val_loss']  for m in fold_metrics]

    print("\n===== RESUMEN K-FOLD =====")
    for m in fold_metrics:
        print(f"Fold {m['fold']}: kNN={m['knn_accuracy']:.2%} | val_loss={m['final_val_loss']:.4f}")
    print(f"\nkNN Accuracy: {np.mean(accs):.2%} ± {np.std(accs):.2%}")
    print(f"Val Loss:     {np.mean(losses):.4f} ± {np.std(losses):.4f}")

    # Gráfica de loss por fold
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, history in enumerate(fold_histories):
        axes[0].plot(history['train_loss'], label=f'Fold {i+1}', alpha=0.7)
    axes[0].set_title('Train Loss por Fold')
    axes[0].set_xlabel('Época')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(1, n_folds+1), accs, color='steelblue', alpha=0.7)
    axes[1].axhline(np.mean(accs), color='red', linestyle='--', 
                    label=f'Media: {np.mean(accs):.2%}')
    axes[1].set_title('kNN Accuracy por Fold')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
###########################################
"""Ejecución"""
###########################################

# Paso 1: K-Fold para validar que el modelo generaliza
fold_histories, fold_metrics = entrenar_con_kfold(
    train_audios_crudos, device_to_label, n_splits=n_splits, EPOCHS=EPOCHS, LR=LR
)
visualizar_resultados_kfold(fold_histories, fold_metrics)

# Paso 2: Modelo final con todos los datos de entrenamiento
model, history, mean_global, std_global, device = entrenar_modelo_final(
    train_audios_crudos, device_to_label,EPOCHS=EPOCHS, LR=LR
)

random.seed(42)#para q la generacion del dataset sea reproducible

###########################################
"""Evaluacion con knn"""
###########################################

# Generar ventanas de TRAIN (todos los datos usados para entrenar el modelo final)
ds_train_final       = preparar_dataset_splicing_completo(train_audios_crudos)
ventanas_train_final = segmentar_en_ventanas(ds_train_final)

# Generar ventanas de TEST (el 10% reservado al inicio, nunca visto en entrenamiento)
ds_test_final       = preparar_dataset_splicing_completo(test_audios)
ventanas_test_final = segmentar_en_ventanas(ds_test_final)

train_emb_loader = DataLoader(
    EmbeddingExtractionDataset(ventanas_train_final, device_to_label, mean_global, std_global),
    batch_size=32
)
val_emb_loader = DataLoader(
    EmbeddingExtractionDataset(ventanas_test_final, device_to_label, mean_global, std_global),
    batch_size=32
)

# El modelo y el device ya vienen listos desde entrenar_modelo_final()
train_embs, train_labs = extract_embeddings(model, train_emb_loader, device)
val_embs,   val_labs   = extract_embeddings(model, val_emb_loader,   device)

knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn.fit(train_embs.numpy(), train_labs.numpy())
preds = knn.predict(val_embs.numpy())
knn_accuracy = accuracy_score(val_labs.numpy(), preds)

print(f" Precisión del k-NN (modelo final vs test set): {knn_accuracy:.2%}")

# Paso 3: Guardar
torch.save({
    'model_state_dict': model.state_dict(),
    'mean_global':      mean_global,
    'std_global':       std_global,
}, "modelo_siames.pth")
print("Modelo final guardado en modelo_siames.pth")

# Paso 4: Visualizar loss del modelo final
plt.plot(history['train_loss'], label='train loss (modelo final)')
plt.legend()
plt.show()


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
""" Trainer de training.py"""
###########################################


ENTRENAR=True  #  poner True la PRIMERA vez para generar el archivo

# 2. Instanciar componentes (siempre)
# model     = EmbeddingNet(out_dim=OUT_DIM)
# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
# miner     = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard")
# loss_fn   = losses.TripletMarginLoss(margin=0.2)

# # 3. Adapter y Trainer (siempre)
# def pml_adapter(model, batch, func):
#     specs, labels = batch
#     embeddings = model(specs)
#     hard_pairs = miner(embeddings, labels)
#     return func(embeddings, labels, hard_pairs)

# trainer = Trainer()
# trainer.set_adapter(pml_adapter)

# # 4. Entrenar o cargar
# if ENTRENAR:
#     history = trainer.fit(
#         model=model,
#         loader=train_loader,
#         valid_loader=val_loader,
#         loss_fn=loss_fn,
#         optimizer=optimizer,
#         epochs=EPOCHS
#     )
#     #Guardar el modelo y sus estadísticas de normalización global
#     torch.save({
#     'model_state_dict': model.state_dict(),
#     'mean_global': mean_global,
#     'std_global':  std_global,
# }, "modelo_siames.pth")
#     print("Modelo guardado en modelo_siames.pth")

#     # Visualizar loss solo si se entrenó
#     plt.plot(history['train_loss'], label='train loss')
#     plt.legend()
#     plt.show()

# else:
#     checkpoint  = torch.load("modelo_siames.pth", map_location=trainer._device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     mean_global = checkpoint['mean_global']
#     std_global  = checkpoint['std_global']
#     model.to(trainer._device)
#     model.eval()
#     print("Modelo cargado correctamente")
###########################################
"""Vissualizacion del loss"""
###########################################


plt.plot(history['train_loss'], label='train loss')
plt.legend()
plt.show()


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
        specs = specs.to(device)
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
        mel_norm = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)#Normalizacion por instancia
        mel_norm = (mel_norm - mean_global) / (std_global + 1e-8)#Normalizacion global
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
###
"""distancias"""
###


###
#PRUEBA
####
# 1. Tomar dos audios que el modelo NO haya visto durante entrenamiento
#    (usa índices del final del dataset, o separa un test set desde el principio)

# Y para probar usas test_audios
audio_test_a = random.choice([a for a in test_audios if a['device'] == 'huawei_nova_9'])
audio_test_b = random.choice([a for a in test_audios if a['device'] == 'poco_m4_pro'])
# 2. Generar el splicing con tu función existente
resultado_test = generar_par_splicing(audio_test_a, audio_test_b)

audio_con_splicing = resultado_test['audio']
punto_corte_real   = resultado_test['cut_point']
tiempo_corte_real  = resultado_test['cut_time']

print(f"Punto de corte real: muestra {punto_corte_real} -> {tiempo_corte_real:.2f} segundos")
# 3. Pasarlo al reporte forense
resultado = generar_reporte_forense(
    audio_sospechoso=audio_con_splicing,
    modelo=model,
    device=device,
    threshold=0.6
)
# 3. Pasarlo al reporte forense
# Ver todas las distancias para entender el rango
distancias = resultado['distancias']
print(f"Distancia mínima:   {min(distancias):.4f}")
print(f"Distancia máxima:   {max(distancias):.4f}")
print(f"Distancia promedio: {np.mean(distancias):.4f}")
print(f"Distancia std:      {np.std(distancias):.4f}")
# Threshold automático basado en las distancias del propio audio
threshold_auto = np.mean(distancias) + np.std(distancias)
print(f"Threshold sugerido: {threshold_auto:.4f}")


resultado = generar_reporte_forense(
    audio_sospechoso=audio_con_splicing,
    modelo=model,
    device=device,
    threshold=threshold_auto  # usar este en vez de 0.6
)
# observacion cuál es la distancia máxima real que está obteniendo
print(f"Distancia máxima encontrada: {resultado['distancia_maxima']:.4f}")
print(f"Threshold actual:            {resultado['threshold_usado']:.4f}")
print(f"¿Detectó empalme?:           {resultado['hay_empalme']}")

# 4. Comparar lo que detectó el modelo vs la realidad

if resultado['hay_empalme']:
    error = abs(tiempo_corte_real - resultado['tiempo_empalme_seg'])
    print(f"Modelo detectó: empalme en {resultado['tiempo_empalme_seg']:.2f}s")
    print(f"Error:          {error:.2f}s")
else:
    print(f"Modelo detectó: AUDIO AUTÉNTICO (no superó threshold {resultado['threshold_usado']:.4f})")
    print(f"Distancia máx:  {resultado['distancia_maxima']:.4f}")


resultados_eer = evaluar_eer_completo(
    model           = model,          # sale de entrenar_modelo_final()
    test_audios     = test_audios,    # el 10% reservado desde el principio
    device_to_label = device_to_label,
    mean_global     = mean_global,
    std_global      = std_global,
    device          = device,
    n_audios_test   = 60,
    guardar_pdf     = True,
)

# Reemplaza el threshold fijo 0.6 que tenías hardcodeado:
resultado = generar_reporte_forense(
    audio_sospechoso = audio_con_splicing,
    modelo           = model,
    device           = device,
    threshold        = resultados_eer['threshold_recomendado'],  # ← automático
)