"""
Laboratorio de Inteligencia Artificial - unintep
Anexo Práctico a la Guía de Estudio Avanzada de LLMs

Prerrequisitos: pip install torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def ejemplo_desvanecimiento_rnn():
    """
    Capítulo 1: Demuestra cómo las Redes Neuronales Recurrentes (RNN) pierden
    la memoria de los primeros tokens en secuencias largas (Vanishing Gradient).
    """
    print("="*60)
    print("EJEMPLO 1: DESVANECIMIENTO DE MEMORIA EN RNN")
    print("="*60)
    
    seq_len = 100 # Secuencia larga para evidenciar el cuello de botella
    input_size = 10
    hidden_size = 20
    
    # Inicializamos una RNN estándar
    rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    
    # Creamos un tensor de entrada y le exigimos gradientes para rastrear el aprendizaje
    x = torch.randn(1, seq_len, input_size, requires_grad=True) 
    
    # Forward pass
    out, h_n = rnn(x)
    
    # Simulamos una tarea donde SOLO importa el último token (ej. predecir el sentimiento de todo el texto)
    # Calculamos una pérdida ficticia sumando los valores del último estado oculto
    loss = out[0, -1].sum() 
    
    # Backward pass: propagamos el error hacia atrás en el tiempo (BPTT)
    loss.backward()
    
    # Analizamos la magnitud del gradiente que logró llegar a los primeros y últimos tokens
    grad_norm_inicio = x.grad[0, 0].norm().item()
    grad_norm_mitad = x.grad[0, seq_len//2].norm().item()
    grad_norm_fin = x.grad[0, -1].norm().item()
    
    print(f"Magnitud de actualización (gradiente) en token t=0:  {grad_norm_inicio:.8f}")
    print(f"Magnitud de actualización (gradiente) en token t=50: {grad_norm_mitad:.8f}")
    print(f"Magnitud de actualización (gradiente) en token t=99: {grad_norm_fin:.8f}")
    print("\nConclusión: El gradiente en el paso t=0 es minúsculo en comparación con t=99.")
    print("Esto demuestra matemáticamente por qué las RNN 'olvidan' el inicio de oraciones largas.\n")


class MultiHeadAttention(nn.Module):
    """
    Capítulo 2: Implementación de Auto-Atención Multicabeza para demostrar
    cómo el Transformer procesa toda la secuencia matricialmente sin bucles temporales.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        # Proyecciones lineales para Query, Key y Value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        # 1. Proyecciones (Procesamos TODOS los tokens al mismo tiempo)
        Q = self.W_q(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Producto escalar escalado (Atención) - ¡Cero bucles 'for' sobre seq_length!
        # Multiplicación de matrices masiva en paralelo
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        # 3. Ensamblaje y proyección final
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.W_o(context)
        return output

def ejemplo_atencion_paralela():
    print("="*60)
    print("EJEMPLO 2: ATENCIÓN EN PARALELO (TRANSFORMER)")
    print("="*60)
    
    batch_size = 32
    seq_len = 512 # Media página de texto
    d_model = 768 # Dimensión estándar tipo BERT Base
    num_heads = 12
    
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Tensor de entrada que representa 32 textos, cada uno de 512 palabras
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Ingresando un lote de dimensiones: {x.shape} (Batch, Secuencia, D_Model)")
    print("Ejecutando propagación hacia adelante (forward pass)...")
    
    out = mha(x)
    
    print(f"Salida obtenida con dimensiones:   {out.shape}")
    print("\nConclusión: El modelo procesó 32 secuencias de 512 tokens simultáneamente")
    print("con una sola serie de operaciones matriciales, aprovechando la arquitectura GPU.")
    print("Esta es la ventaja computacional del Transformer frente al procesamiento secuencial RNN.\n")


class LoRALinear(nn.Module):
    """
    Capítulo 8: Implementación de una capa densa inyectada con Adaptación de Bajo Rango (LoRA).
    """
    def __init__(self, linear_layer, rank=4, alpha=16):
        super().__init__()
        self.linear = linear_layer
        
        # 1. CONGELAMOS la capa base original para que no consuma memoria de gradientes
        for param in self.linear.parameters():
            param.requires_grad = False
            
        out_features, in_features = self.linear.weight.shape
        
        # 2. INYECTAMOS matrices de bajo rango A y B
        # A reduce la dimensionalidad de in_features -> rank
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        # B escala la dimensionalidad de rank -> out_features (Iniciada en 0 para no alterar la capa base al inicio)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        self.scaling = alpha / rank
        
    def forward(self, x):
        # Y = W_0*X + (B*A)*X * scaling
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

def ejemplo_eficiencia_computacional():
    print("="*60)
    print("EJEMPLO 3: EFICIENCIA (FLASH-ATTENTION Y LORA)")
    print("="*60)
    
    # --- Parte A: FlashAttention ---
    print(">> A) FlashAttention (PyTorch >= 2.0)")
    q = torch.randn(1, 12, 1024, 64, dtype=torch.float32)
    k = torch.randn(1, 12, 1024, 64, dtype=torch.float32)
    v = torch.randn(1, 12, 1024, 64, dtype=torch.float32)
    
    try:
        # PyTorch detecta automáticamente el hardware y aplica FlashAttention si es posible
        out_flash = F.scaled_dot_product_attention(q, k, v)
        print("La API 'scaled_dot_product_attention' ejecutó el algoritmo con éxito.")
        print("En una GPU real, esto fusiona los kernels CUDA, evitando escribir y leer")
        print("la inmensa matriz (1024x1024) en la lenta memoria global (HBM) de la tarjeta gráfica.\n")
    except Exception as e:
        print(f"Error al ejecutar FlashAttention: {e}\n")

    # --- Parte B: LoRA ---
    print(">> B) Adaptación de Bajo Rango (LoRA)")
    in_dim, out_dim = 4096, 4096 # Dimensiones típicas en LLMs como Llama o GPT-3
    capa_base = nn.Linear(in_dim, out_dim)
    
    # Contamos parámetros de un Fine-Tuning completo (todas las matrices entrenables)
    parametros_totales = sum(p.numel() for p in capa_base.parameters())
    
    # Aplicamos LoRA con rango r=8
    rango_lora = 8
    capa_adaptada = LoRALinear(capa_base, rank=rango_lora)
    
    # Contamos SOLAMENTE los parámetros con requires_grad=True
    parametros_entrenables_lora = sum(p.numel() for p in capa_adaptada.parameters() if p.requires_grad)
    
    print(f"Parámetros a entrenar en Fine-Tuning Tradicional: {parametros_totales:,}")
    print(f"Parámetros a entrenar usando LoRA (r={rango_lora}):           {parametros_entrenables_lora:,}")
    
    ahorro = (1 - (parametros_entrenables_lora / parametros_totales)) * 100
    print(f"Reducción de parámetros actualizables:              {ahorro:.4f}%")
    print("\nConclusión: Con LoRA reducimos el esfuerzo computacional en más de un 99%,")
    print("haciendo posible afinar LLMs gigantescos en laboratorios universitarios.")


if __name__ == '__main__':
    try:
        ejemplo_desvanecimiento_rnn()
        ejemplo_atencion_paralela()
        ejemplo_eficiencia_computacional()
    except KeyboardInterrupt:
        print("\nEjecución interrumpida.")