import torch
import nlpaug.augmenter.word as naw
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading


# Opción 1: Distribución manual por GPUs
def balance_with_multi_gpu(df, text_col='text', label_col='value', gpu_ids=[0, 1]):
    """Balancea dataset usando múltiples GPUs"""
    
    # Verificar GPUs disponibles
    available_gpus = torch.cuda.device_count()
    print(f"GPUs disponibles: {available_gpus}")
    
    if available_gpus < len(gpu_ids):
        print(f"⚠️ Solo {available_gpus} GPUs disponibles, usando las primeras {available_gpus}")
        gpu_ids = gpu_ids[:available_gpus]
    
    # Crear augmenters para cada GPU
    augmenters = {}
    for gpu_id in gpu_ids:
        augmenters[gpu_id] = naw.ContextualWordEmbsAug(
            model_path='bert-base-multilingual-uncased',
            aug_p=0.1,
            device=f'cuda:{gpu_id}'
        )
        print(f"✅ Augmenter creado en GPU {gpu_id}")
    
    class_counts = df[label_col].value_counts()
    target_count = class_counts.max()
    
    def process_class_on_gpu(args):
        """Procesa una clase en una GPU específica"""
        class_label, needed_samples, class_texts, gpu_id = args
        
        augmenter = augmenters[gpu_id]
        augmented_data = []
        
        print(f"🚀 GPU {gpu_id}: Procesando clase {class_label} ({needed_samples} muestras)")
        
        batch_size = 16
        for i in range(0, needed_samples, batch_size):
            batch_end = min(i + batch_size, needed_samples)
            batch_texts = [class_texts[j % len(class_texts)] for j in range(i, batch_end)]
            print("batch_size", batch_size, "batch_end", batch_end, "needed_samples", needed_samples)
            
            try:
                augmented_batch = augmenter.augment(batch_texts)
                
                for aug_text in augmented_batch:
                    augmented_data.append({
                        text_col: aug_text,
                        label_col: class_label,
                        "lang": "en",
                        'gpu_used': gpu_id
                    })
            except Exception as e:
                print(f"Error en GPU {gpu_id}: {e}")
                # Fallback
                for text in batch_texts:
                    augmented_data.append({
                        text_col: text,
                        label_col: class_label,
                        "lang": "en",
                        'gpu_used': gpu_id
                    })
        
        print(f"✅ GPU {gpu_id}: Completada clase {class_label}")
        return augmented_data
    
    # Preparar trabajos para cada GPU
    jobs = []
    gpu_idx = 0
    
    for class_label, current_count in class_counts.items():
        needed = target_count - current_count
        if needed <= 0:
            continue
            
        class_texts = df[df[label_col] == class_label][text_col].tolist()
        gpu_id = gpu_ids[gpu_idx % len(gpu_ids)] # Rotar entre GPUs
        
        jobs.append((class_label, needed, class_texts, gpu_id))
        gpu_idx += 1
    
    # Ejecutar en paralelo
    all_augmented_data = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        results = executor.map(process_class_on_gpu, jobs)
        
        for result in results:
            all_augmented_data.extend(result)
    
    # Combinar resultados
    augmented_df = pd.DataFrame(all_augmented_data)
    balanced_df = pd.concat([df, augmented_df], ignore_index=True)
    
    print(f"\n🎉 Completado!")
    print(f"Total original: {len(df)}")
    print(f"Total final: {len(balanced_df)}")
    print(f"Muestras añadidas: {len(augmented_df)}")
    
    # Mostrar uso por GPU
    if len(augmented_df) > 0:
        gpu_usage = augmented_df['gpu_used'].value_counts()
        print("\nUso por GPU:")
        for gpu_id, count in gpu_usage.items():
            print(f"  GPU {gpu_id}: {count} muestras")
    
    return balanced_df

# Usar con 2 GPUs
balanced_df = balance_with_multi_gpu(df_all, gpu_ids=[0, 1])





# Opción 2: Distribución por lotes entre GPUs
def balance_with_batch_distribution(df, text_col='text', label_col='value', gpu_ids=[0, 1]):
    """Distribuye lotes entre GPUs de forma más equilibrada"""
    
    available_gpus = torch.cuda.device_count()
    gpu_ids = gpu_ids[:available_gpus]
    
    # Crear augmenters
    augmenters = {
        gpu_id: naw.ContextualWordEmbsAug(
            model_path='bert-base-multilingual-uncased',
            aug_p=0.1,
            device=f'cuda:{gpu_id}'
        ) for gpu_id in gpu_ids
    }
    
    class_counts = df[label_col].value_counts()
    target_count = class_counts.max()
    
    # Recopilar todos los trabajos
    all_jobs = []
    for class_label, current_count in class_counts.items():
        needed = target_count - current_count
        if needed <= 0:
            continue
            
        class_texts = df[df[label_col] == class_label][text_col].tolist()
        
        # Dividir en lotes
        batch_size = 32
        for i in range(0, needed, batch_size):
            batch_end = min(i + batch_size, needed)
            batch_texts = [class_texts[j % len(class_texts)] for j in range(i, batch_end)]
            all_jobs.append((class_label, batch_texts))
    
    def process_batch_on_gpu(args):
        job_idx, (class_label, batch_texts) = args
        gpu_id = gpu_ids[job_idx % len(gpu_ids)]  
        augmenter = augmenters[gpu_id]
        
        try:
            augmented_batch = augmenter.augment(batch_texts)
            return [(text, class_label, gpu_id) for text in augmented_batch]
        except Exception as e:
            print(f"Error en GPU {gpu_id}: {e}")
            return [(text, class_label, gpu_id) for text in batch_texts]
    
    # Procesar en paralelo
    all_results = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        indexed_jobs = list(enumerate(all_jobs))
        results = executor.map(process_batch_on_gpu, indexed_jobs)
        
        for result in results:
            all_results.extend(result)
    
    # Crear DataFrame final
    augmented_data = [{
        text_col: text,
        label_col: label,
        'gpu_used': gpu_id
    } for text, label, gpu_id in all_results]
    
    augmented_df = pd.DataFrame(augmented_data)
    balanced_df = pd.concat([df, augmented_df], ignore_index=True)
    
    return balanced_df

# Usar distribución por lotes
balanced_df = balance_with_batch_distribution(df_all, gpu_ids=[0, 1])





# Opción 3: Usando DataParallel (más automático)
import torch.nn as nn

def balance_with_dataparallel(df, text_col='text', label_col='value'):
    """Usa DataParallel para distribución automática"""
    
    if torch.cuda.device_count() < 2:
        print("⚠️ Se necesitan al menos 2 GPUs para DataParallel")
        return df
    
    # Crear augmenter que use todas las GPUs disponibles
    aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-multilingual-uncased',
        aug_p=0.1,
        device='cuda'
    )
    
    # El modelo interno se distribuirá automáticamente
    print(f"🚀 Usando {torch.cuda.device_count()} GPUs con DataParallel")
    
    class_counts = df[label_col].value_counts()
    target_count = class_counts.max()
    
    augmented_data = []
    
    for class_label, current_count in class_counts.items():
        needed = target_count - current_count
        if needed <= 0:
            continue
            
        class_texts = df[df[label_col] == class_label][text_col].tolist()
        
        # Procesar en lotes grandes para mejor paralelización
        batch_size = 64  # Lotes más grandes para mejor distribución
        for i in range(0, needed, batch_size):
            batch_end = min(i + batch_size, needed)
            batch_texts = [class_texts[j % len(class_texts)] for j in range(i, batch_end)]
            
            try:
                augmented_batch = aug.augment(batch_texts)
                for aug_text in augmented_batch:
                    augmented_data.append({
                        text_col: aug_text,
                        label_col: class_label
                    })
            except Exception as e:
                print(f"Error: {e}")
    
    return pd.concat([df, pd.DataFrame(augmented_data)], ignore_index=True)





# Recomendación: Usa la Opción 1 para control total sobre qué GPU procesa qué, o la Opción 2 para mejor distribución de carga. La Opción 3 es más simple pero menos control.