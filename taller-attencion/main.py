
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        # Verificar que las dimensiones se puedan dividir
        assert d_model % num_heads == 0, (
            "d_model debe ser divisible entre num_heads"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Proyecciones para consultas, claves y valores
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Proyección final
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):

        # x: [batch, secuencia, d_model]
        batch_size, seq_len, _ = x.shape

        # Calcular consultas, claves y valores
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Separar las cabezas de atención
        Q = Q.reshape(
            batch_size, seq_len,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = K.reshape(
            batch_size, seq_len,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        V = V.reshape(
            batch_size, seq_len,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Calcular la importancia de cada posición
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        # Convertir los puntajes en pesos de atención
        weights = torch.softmax(scores, dim=-1)

        # Combinar la información de los valores
        output = torch.matmul(weights, V)

        # Unir nuevamente todas las cabezas
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(
            batch_size, seq_len, self.d_model
        )

        # Transformación final
        output = self.W_o(output)

        return output, weights


# Crear el módulo
d_model = 16
num_heads = 4

attention = MultiHeadAttention(d_model, num_heads)

# Crear datos sintéticos:
# 2 secuencias, 5 posiciones y 16 características
x = torch.randn(2, 5, d_model)

# Ejecutar la atención
output, weights = attention(x)

# Mostrar las dimensiones
print("Entrada:", x.shape)
print("Salida:", output.shape)
print("Pesos de atención:", weights.shape)

# Verificar la conservación de dimensiones
assert output.shape == x.shape

print("Prueba superada: entrada y salida tienen "
      "las mismas dimensiones.")
