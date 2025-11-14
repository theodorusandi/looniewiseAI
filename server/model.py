import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout


class MultiHeadSelfAttention(Layer):
    """
    Multi-head self-attention mechanism for transformer architectures.

    This layer implements the scaled dot-product attention with multiple attention heads,
    allowing the model to jointly attend to information from different representation
    subspaces at different positions.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input and output embeddings.
    num_heads : int, default=8
        Number of parallel attention heads. Must evenly divide embed_dim.
    dropout : float, default=0.1
        Dropout rate applied to attention weights for regularization.
    kernel_regularizer : keras.regularizers.Regularizer, optional
        Regularizer function applied to the dense layer kernels.

    Raises
    ------
    ValueError
        If embed_dim is not divisible by num_heads.

    Notes
    -----
    The attention mechanism computes:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    where d_k is the projection dimension (embed_dim // num_heads).

    References
    ----------
    Vaswani et al., "Attention is All You Need", NeurIPS 2017.
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.1, kernel_regularizer=None):
        super(MultiHeadSelfAttention, self).__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.scale = tf.math.sqrt(tf.cast(self.projection_dim, tf.float32))
        self.query_dense = Dense(embed_dim, kernel_regularizer=kernel_regularizer)
        self.key_dense = Dense(embed_dim, kernel_regularizer=kernel_regularizer)
        self.value_dense = Dense(embed_dim, kernel_regularizer=kernel_regularizer)
        self.combine_heads = Dense(embed_dim, kernel_regularizer=kernel_regularizer)

        self.attention_dropout = Dropout(dropout)

    def attention(self, query, key, value, training=None):
        score = tf.matmul(query, key, transpose_b=True)
        scaled_score = score / self.scale
        weights = tf.nn.softmax(scaled_score, axis=-1)
        weights = self.attention_dropout(weights, training=training)
        output = tf.matmul(weights, value)

        return output

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        attention = self.attention(query, key, value, training=training)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)

        return output


class TransformerBlock(Layer):
    """
    Transformer encoder block with pre-normalization architecture.

    This layer implements a standard transformer encoder block consisting of multi-head
    self-attention followed by a position-wise feedforward network. It uses pre-layer
    normalization (LayerNorm before each sublayer) and residual connections around
    both sublayers.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input and output embeddings.
    num_heads : int
        Number of attention heads in the multi-head attention layer.
    ff_dim : int
        Dimensionality of the hidden layer in the feedforward network.
    dropout : float, default=0.1
        Dropout rate applied after attention and feedforward sublayers.
    self_attention_reg : keras.regularizers.Regularizer, optional
        Regularizer function applied to the attention layer kernels.
    ff_reg : keras.regularizers.Regularizer, optional
        Regularizer function applied to the feedforward network kernels.

    Notes
    -----
    Architecture follows the pre-norm variant:
        1. LayerNorm -> Multi-Head Attention -> Dropout -> Residual Add
        2. LayerNorm -> Feedforward -> Dropout -> Residual Add

    The feedforward network uses GELU activation, which is standard in modern
    transformers like BERT and GPT.

    References
    ----------
    Vaswani et al., "Attention is All You Need", NeurIPS 2017.
    Xiong et al., "On Layer Normalization in the Transformer Architecture", ICML 2020.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        dropout=0.1,
        self_attention_reg=None,
        ff_reg=None,
    ):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(
            embed_dim, num_heads, dropout=dropout, kernel_regularizer=self_attention_reg
        )
        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, activation="gelu", kernel_regularizer=ff_reg),
                Dense(embed_dim, kernel_regularizer=ff_reg),
            ]
        )

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training=None):
        normed_inputs = self.layernorm1(inputs)
        attn_output = self.att(normed_inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output

        normed_out1 = self.layernorm2(out1)
        ffn_output = self.ffn(normed_out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return out1 + ffn_output


class TransformerEncoder(Layer):
    """
    Stack of transformer encoder blocks with final layer normalization.

    This encoder stacks multiple transformer blocks and applies a final layer
    normalization to stabilize outputs. It follows the pre-norm architecture
    where each block applies normalization before sublayers.

    Parameters
    ----------
    num_layers : int
        Number of transformer blocks to stack.
    embed_dim : int
        Dimensionality of the input and output embeddings.
    num_heads : int
        Number of attention heads in each transformer block.
    ff_dim : int
        Dimensionality of the feedforward network hidden layer.
    dropout : float, default=0.1
        Dropout rate applied in each transformer block.
    self_attention_reg : keras.regularizers.Regularizer, optional
        Regularizer for attention layer weights.
    ff_reg : keras.regularizers.Regularizer, optional
        Regularizer for feedforward network weights.

    Notes
    -----
    The final layer normalization helps stabilize gradients and outputs,
    particularly important for deeper networks.

    References
    ----------
    Vaswani et al., "Attention is All You Need", NeurIPS 2017.
    """

    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        ff_dim,
        dropout=0.1,
        self_attention_reg=None,
        ff_reg=None,
    ):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.enc_layers = [
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                self_attention_reg=self_attention_reg,
                ff_reg=ff_reg,
            )
            for _ in range(num_layers)
        ]

        self.final_layernorm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None):
        x = inputs

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training)

        x = self.final_layernorm(x)

        return x


class LearnedPositionalEncoding(Layer):
    """
    Learned positional encoding for transformer models.

    Adds trainable position-specific embeddings to input sequences, enabling
    the model to learn optimal positional representations for the task.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments passed to the Layer base class.

    Notes
    -----
    - Requires fixed sequence length at build time
    - No dropout applied (position information should be deterministic)
    - Embeddings initialized from N(0, 0.02²) distribution

    References
    ----------
    Gehring et al., "Convolutional Sequence to Sequence Learning", ICML 2017.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, L, D)
        if len(input_shape) != 3:
            raise ValueError("Expected input of shape (batch, seq_len, embed_dim).")

        _, L, D = input_shape

        if L is None or D is None:
            raise ValueError("seq_len and embed_dim must be known at build time.")

        self.seq_len = int(L)
        self.embed_dim = int(D)

        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(1, self.seq_len, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            dtype=self.compute_dtype,
        )

        super().build(input_shape)

    def call(self, x):
        # Broadcast (1, L, D) -> (B, L, D)
        return x + tf.cast(self.pos_emb, x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape
