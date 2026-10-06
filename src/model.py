"""
Transformer Encoder-Decoder
"""
import tensorflow as tf
from tensorflow.keras import layers

# Converte ogni dispositivo in un vettore di dimensione d_model
class EmbeddingLayer(layers.Layer):
    def __init__(
        self,
        d_model, # Dimensione del vettore in uscita
        num_types: int = 3, 
        num_positions: int = 2,
        name="embedding_layer",
        **kwargs):

        super().__init__(name=name, **kwargs)

        # Hyperparameters
        self.num_types = num_types
        self.num_positions = num_positions
        self.d_model = d_model

        # Proiezione lineare
        self.projection = layers.Dense(self.d_model, name="embedding_projection")

    def call(self, x):
        types = x[:, :, 0]
        positions = x[:, :, 1]
        states = x[:, :, 2]

        # One-hot encoding del tipo
        type_onehot = tf.one_hot(tf.cast(types, tf.int32), depth = self.num_types)
        # One-hot encoding della posizione
        position_onehot = tf.one_hot(tf.cast(positions, tf.int32), depth = self.num_positions)

        states = tf.expand_dims(states, axis = -1)

        # Concatenazione delle feature
        concat = tf.concat([type_onehot, position_onehot, states], axis = -1)

        embedded = self.projection(concat)

        return embedded

class EncoderLayer(layers.Layer):
    def __init__(
        self, 
        d_model, 
        num_heads, # Numero di teste di attention
        dff, # Dimensione della Feed-Forward
        dropout_rate = 0.1, 
        name = 'encoder_layer',
        **kwargs):
        
        super().__init__(name = name, **kwargs)

        # Multi-Head Self-Attention
        # Ogni testa lavora su d_model // num_heads dimensioni
        self.attention = layers.MultiHeadAttention(num_heads = num_heads, key_dim = d_model // num_heads, name = "self_attention")

        # Feed Forward 
        self.ffn_dense1 = layers.Dense(dff, activation = 'gelu', name = "ffn_dense1") # Espansione con gelu
        self.ffn_dense2 = layers.Dense(d_model, name = "ffn_dense2") # Compressione

        # Layer Normalization
        self.layernorm1 = layers.LayerNormalization(name = "layernorm1")
        self.layernorm2 = layers.LayerNormalization(name = "layernorm2")

        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training = False):
        ## SELF-ATTENTION
        # Normalizzazione di x prima dell'attention
        x_norm = self.layernorm1(x)

        # Self-Attention
        attention_output = self.attention(query = x_norm, key = x_norm, value = x_norm, training = training)

        # Dropout
        attention_output = self.dropout1(attention_output, training = training)

        # Somma dell'output all'input originale
        x = x + attention_output

        ## FEED-FORWARD NETWORK
        x_norm = self.layernorm2(x)

        ffn_output = self.ffn_dense1(x_norm) # Espansione
        ffn_output = self.ffn_dense2(ffn_output) # Compressione

        # Dropout
        ffn_output = self.dropout2(ffn_output, training = training)

        # Connessione residua
        x = x + ffn_output

        return x

class Encoder(layers.Layer):
    def __init__(
        self, 
        num_layers,
        d_model, 
        num_heads, 
        dff, 
        dropout_rate = 0.1, 
        name = 'encoder',
        **kwargs):
            
        super().__init__(name = name, **kwargs)

        # Creazione di num_layers blocchi con i propri pesi
        self.encoder_layers = [
            EncoderLayer(
                d_model = d_model,
                num_heads = num_heads,
                dff = dff,
                dropout_rate = dropout_rate,
                name = f'encoder_layer_{i}'
            )
            for i in range(num_layers)
        ]

    def call(self, x, training = False):
        # L'output di ogni blocco è l'input del successivo
        for layer in self.encoder_layers:
            x = layer(x, training = training)
        
        return x

class DecoderLayer(layers.Layer):
    def __init__(
        self, 
        d_model, 
        num_heads, 
        dff, 
        dropout_rate = 0.1, 
        name = 'decoder_layer',
        **kwargs):
        
        super().__init__(name = name, **kwargs)

        # Multi-Head Self-Attention
        self.attention = layers.MultiHeadAttention(num_heads = num_heads, key_dim = d_model // num_heads, name = "self_attention")

        # Multi-Head Cross-Attention
        self.cross_attention = layers.MultiHeadAttention(num_heads = num_heads, key_dim = d_model // num_heads, name = "cross_attention")

        # Feed Forward Network
        self.ffn_dense1 = layers.Dense(dff, activation = 'gelu', name = "ffn_dense1")
        self.ffn_dense2 = layers.Dense(d_model, name = "ffn_dense2")

        # Normalization Layer
        self.layernorm1 = layers.LayerNormalization(name = "layernorm1")
        self.layernorm2 = layers.LayerNormalization(name = "layernorm2")
        self.layernorm3 = layers.LayerNormalization(name = "layernorm3")

        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, x, encoder_output, training = False):
        ## SELF-ATTENTION
        # Normalizzazione di x prima dell'attention
        x_norm = self.layernorm1(x)
        # Self-Attention
        attention_output = self.attention(query = x_norm, key = x_norm, value = x_norm, training = training)
        # Dropout
        attention_output = self.dropout1(attention_output, training = training)
        # Somma dell'output all'input originale
        x = x + attention_output

        ## CROSS-ATTENTION
        # Normalizzazione di x prima dell'attention
        x_norm = self.layernorm2(x)
        # Cross-Attention
        cross_output = self.cross_attention(query = x_norm, key = encoder_output, value = encoder_output, training = training)
        # Dropout
        cross_output = self.dropout2(cross_output, training = training)
        # Somma dell'output all'input originale
        x = x + cross_output

        ## FEED-FORWARD NETWORK
        x_norm = self.layernorm3(x)
        ffn_output = self.ffn_dense1(x_norm) # Espansione
        ffn_output = self.ffn_dense2(ffn_output) # Compressione
        # Dropout
        ffn_output = self.dropout3(ffn_output, training = training)
        x = x + ffn_output

        return x

class Decoder(layers.Layer):
    def __init__(
        self, 
        num_layers,
        d_model, 
        num_heads, 
        dff, 
        dropout_rate = 0.1, 
        name = 'decoder',
        **kwargs):
            
        super().__init__(name = name, **kwargs)

        self.decoder_layers = [
            DecoderLayer(
                d_model = d_model,
                num_heads = num_heads,
                dff = dff,
                dropout_rate = dropout_rate,
                name = f'decoder_layer_{i}'
            )
            for i in range(num_layers)
        ]

    def call(self, x, encoder_output, training = False):
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, training = training)
        
        return x

class SeqGen(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        prediction_steps,
        num_sensors = 2,
        num_actuators = 2,
        dropout_rate= 0.1,
        name="seqgen",
        **kwargs,
    ):
        super().__init__(name = name, **kwargs)

        self.num_sensors = num_sensors
        self.num_actuators = num_actuators

        # Embedding
        self.embedding = EmbeddingLayer(d_model = d_model)

        # Encoder che elabora gli attuatori
        self.encoder = Encoder(
            num_layers = num_layers,
            d_model = d_model,
            num_heads = num_heads,
            dff = dff,
            dropout_rate = dropout_rate
        )

        # Decoder che elabora i sensori dall'output dell'encoder
        self.decoder = Decoder(
            num_layers = num_layers,
            d_model = d_model,
            num_heads = num_heads,
            dff = dff,
            dropout_rate = dropout_rate
        )

        # Proiezione finale
        self.final_dense = layers.Dense(prediction_steps, name = "output_projection")

    def call(self, x, training = False):
        sensors_input = x[:, :self.num_sensors, :]
        actuators_input = x[:, self.num_sensors:, :]

        sensors_embedded = self.embedding(sensors_input)
        actuators_embedded = self.embedding(actuators_input)

        encoder_output = self.encoder(actuators_embedded, training = training)

        decoder_output = self.decoder(sensors_embedded, encoder_output, training = training)

        # Previsioni
        predictions = self.final_dense(decoder_output)

        return predictions