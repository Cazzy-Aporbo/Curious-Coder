# Deep Learning for Biological and Medical Data
## Neural Networks Meet the Complexity of Life

### Intent
Deep learning has revolutionized biological data analysis - from predicting protein structures with AlphaFold to diagnosing diseases from medical images. This document provides rigorous frameworks for applying deep learning to biological data, addressing unique challenges like small sample sizes, interpretability requirements, and biological constraints.

### Mathematical Foundations for Biological Deep Learning

**Universal Approximation in Biology:**

For any continuous biological function f: ℝⁿ → ℝᵐ and ε > 0, there exists a neural network g with one hidden layer such that:

```
||f(x) - g(x)|| < ε for all x in compact set K
```

**But biological constraints add complexity:**
- Non-negativity (concentrations, counts)
- Sum-to-one (proportions, probabilities)
- Symmetries (molecular rotations, cell orientations)
- Hierarchies (cells → tissues → organs)

### Architectures for Biological Data Types

#### 1. Convolutional Networks for Biological Images

```python
def biological_cnn(input_shape, n_classes, data_type='histopathology'):
    """
    CNN architecture adapted for biological imaging
    """
    from tensorflow.keras import layers, models, regularizers
    
    model = models.Sequential()
    
    if data_type == 'histopathology':
        # Multi-scale features important for tissue architecture
        
        # Multi-resolution pathway
        # Branch 1: Fine details (cells)
        model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                                input_shape=input_shape))
        model.add(layers.BatchNormalization())
        
        # Branch 2: Medium features (glands)
        model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(layers.BatchNormalization())
        
        # Branch 3: Large features (tissue regions)
        model.add(layers.Conv2D(64, (7, 7), activation='relu'))
        model.add(layers.BatchNormalization())
        
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Deeper layers
        for filters in [128, 256, 512]:
            model.add(layers.Conv2D(filters, (3, 3), activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Conv2D(filters, (3, 3), activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(0.25))
        
    elif data_type == 'microscopy':
        # Handle multiple channels (fluorescence)
        
        # Attention to different stains
        model.add(layers.Conv2D(16, (1, 1), activation='relu',
                                input_shape=input_shape))  # Channel attention
        
        # Spatial features
        for filters in [32, 64, 128]:
            model.add(layers.Conv2D(filters, (3, 3), activation='relu',
                                   padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.Conv2D(filters, (3, 3), activation='relu',
                                   padding='same'))
            model.add(layers.BatchNormalization())
            
            # Residual connection
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(0.3))
    
    elif data_type == 'mri':
        # 3D convolutions for volumetric data
        # Would use Conv3D layers
        pass
    
    # Classification head
    model.add(layers.GlobalAveragePooling2D())  # Better than Flatten for localization
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(n_classes, activation='softmax'))
    
    return model

def attention_unet_segmentation(input_shape=(256, 256, 3)):
    """
    U-Net with attention gates for medical image segmentation
    """
    from tensorflow.keras import layers, models, Input
    
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bottleneck
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    
    # Decoder with Attention
    def attention_gate(x, g, inter_channels):
        """
        Attention mechanism for skip connections
        """
        theta_x = layers.Conv2D(inter_channels, 1, strides=1)(x)
        phi_g = layers.Conv2D(inter_channels, 1, strides=1)(g)
        
        add = layers.Add()([theta_x, phi_g])
        relu = layers.Activation('relu')(add)
        psi = layers.Conv2D(1, 1, strides=1, activation='sigmoid')(relu)
        
        return layers.Multiply()([x, psi])
    
    # Upsample with attention
    up3 = layers.UpSampling2D(size=(2, 2))(conv4)
    att3 = attention_gate(conv3, up3, 256)
    up3 = layers.Concatenate()([up3, att3])
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(up3)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)
    
    up2 = layers.UpSampling2D(size=(2, 2))(conv5)
    att2 = attention_gate(conv2, up2, 128)
    up2 = layers.Concatenate()([up2, att2])
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(up2)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)
    
    up1 = layers.UpSampling2D(size=(2, 2))(conv6)
    att1 = attention_gate(conv1, up1, 64)
    up1 = layers.Concatenate()([up1, att1])
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(up1)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)
    
    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv7)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model
```

#### 2. Recurrent Networks for Biological Sequences

```python
def biological_sequence_model(sequence_length, vocab_size, 
                             task='protein_function'):
    """
    RNN/LSTM for biological sequences
    """
    from tensorflow.keras import layers, models
    
    model = models.Sequential()
    
    if task == 'protein_function':
        # Embedding for amino acids
        model.add(layers.Embedding(vocab_size, 128, 
                                   input_length=sequence_length))
        
        # Bidirectional LSTM to capture long-range dependencies
        model.add(layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.3)
        ))
        model.add(layers.Bidirectional(
            layers.LSTM(128, dropout=0.3)
        ))
        
        # Prediction head
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(n_functions, activation='sigmoid'))  # Multi-label
        
    elif task == 'dna_regulatory':
        # Convolutional layers for motif detection
        model.add(layers.Embedding(vocab_size, 64,
                                   input_length=sequence_length))
        
        # Different filter sizes for different motif lengths
        conv_layers = []
        for filter_size in [3, 5, 7, 9]:
            conv = layers.Conv1D(128, filter_size, activation='relu',
                                padding='same')(embedded)
            conv = layers.GlobalMaxPooling1D()(conv)
            conv_layers.append(conv)
        
        # Merge convolutions
        merged = layers.Concatenate()(conv_layers)
        
        # Dense layers
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
    
    return model
```

#### 3. Graph Neural Networks for Molecular Data

```python
def molecular_graph_network(n_features, n_edge_features, n_outputs):
    """
    GNN for molecular property prediction
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool
    
    class MolecularGNN(nn.Module):
        def __init__(self):
            super(MolecularGNN, self).__init__()
            
            # Node embedding
            self.node_embedding = nn.Linear(n_features, 128)
            
            # Graph convolutions
            self.conv1 = GCNConv(128, 256)
            self.conv2 = GCNConv(256, 256)
            self.conv3 = GCNConv(256, 128)
            
            # Edge network
            self.edge_network = nn.Sequential(
                nn.Linear(n_edge_features, 64),
                nn.ReLU(),
                nn.Linear(64, 256 * 256)  # For message passing
            )
            
            # Readout
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, n_outputs)
            
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x, edge_index, edge_attr, batch):
            # Node features
            x = F.relu(self.node_embedding(x))
            
            # Graph convolutions with residual connections
            x1 = F.relu(self.conv1(x, edge_index))
            x1 = self.dropout(x1)
            
            x2 = F.relu(self.conv2(x1, edge_index))
            x2 = self.dropout(x2)
            x2 = x2 + x1  # Residual
            
            x3 = F.relu(self.conv3(x2, edge_index))
            x3 = self.dropout(x3)
            
            # Global pooling
            x = global_mean_pool(x3, batch)
            
            # Final prediction
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x
    
    return MolecularGNN()
```

#### 4. Transformer Models for Biological Data

```python
def biological_transformer(seq_length, vocab_size, d_model=512):
    """
    Transformer for biological sequences (proteins, DNA)
    """
    from tensorflow.keras import layers, models
    import tensorflow as tf
    
    class MultiHeadSelfAttention(layers.Layer):
        def __init__(self, d_model, num_heads):
            super(MultiHeadSelfAttention, self).__init__()
            self.num_heads = num_heads
            self.d_model = d_model
            
            self.depth = d_model // num_heads
            
            self.wq = layers.Dense(d_model)
            self.wk = layers.Dense(d_model)
            self.wv = layers.Dense(d_model)
            
            self.dense = layers.Dense(d_model)
        
        def split_heads(self, x, batch_size):
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])
        
        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            
            # Linear transformations
            q = self.wq(inputs)
            k = self.wk(inputs)
            v = self.wv(inputs)
            
            # Split heads
            q = self.split_heads(q, batch_size)
            k = self.split_heads(k, batch_size)
            v = self.split_heads(v, batch_size)
            
            # Scaled dot-product attention
            matmul_qk = tf.matmul(q, k, transpose_b=True)
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
            
            # Softmax
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            
            # Apply attention to values
            output = tf.matmul(attention_weights, v)
            
            # Concatenate heads
            output = tf.transpose(output, perm=[0, 2, 1, 3])
            output = tf.reshape(output, (batch_size, -1, self.d_model))
            
            # Final linear transformation
            output = self.dense(output)
            
            return output
    
    class TransformerBlock(layers.Layer):
        def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
            super(TransformerBlock, self).__init__()
            self.att = MultiHeadSelfAttention(d_model, num_heads)
            self.ffn = models.Sequential([
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(d_model),
            ])
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = layers.Dropout(dropout_rate)
            self.dropout2 = layers.Dropout(dropout_rate)
        
        def call(self, inputs, training):
            # Self-attention
            attn_output = self.att(inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            
            # Feed-forward
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)
    
    # Build model
    inputs = layers.Input(shape=(seq_length,))
    
    # Embedding
    x = layers.Embedding(vocab_size, d_model)(inputs)
    
    # Positional encoding
    positions = tf.range(start=0, limit=seq_length, delta=1)
    position_embedding = layers.Embedding(seq_length, d_model)(positions)
    x = x + position_embedding
    
    # Transformer blocks
    for _ in range(6):
        x = TransformerBlock(d_model, 8, 2048)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model
```

### Handling Small Sample Sizes

```python
class SmallSampleStrategies:
    """
    Strategies for deep learning with limited biological data
    """
    
    @staticmethod
    def data_augmentation_biological(images, labels, augmentation_factor=10):
        """
        Biology-aware augmentation
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Augmentation that preserves biological features
        datagen = ImageDataGenerator(
            rotation_range=20,      # Cells can be rotated
            width_shift_range=0.1,  # Small translations
            height_shift_range=0.1,
            horizontal_flip=True,   # Many biological structures symmetric
            vertical_flip=True,
            zoom_range=0.2,        # Magnification differences
            brightness_range=[0.8, 1.2],  # Staining variations
            channel_shift_range=20,  # Stain color variations
            fill_mode='reflect'    # Better than zeros for biology
        )
        
        # Generate augmented samples
        augmented_images = []
        augmented_labels = []
        
        for i in range(len(images)):
            img = images[i:i+1]
            
            # Generate augmentations
            aug_iter = datagen.flow(img, batch_size=1)
            
            for _ in range(augmentation_factor):
                aug_img = next(aug_iter)[0]
                augmented_images.append(aug_img)
                augmented_labels.append(labels[i])
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    @staticmethod
    def transfer_learning_biological(base_model_name='inception_v3', 
                                    n_classes=2):
        """
        Transfer learning from ImageNet or biological models
        """
        from tensorflow.keras.applications import InceptionV3
        from tensorflow.keras import layers, models
        
        # Load pre-trained model
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=(299, 299, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom head
        inputs = layers.Input(shape=(299, 299, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Biological-specific layers
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(n_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        
        # Fine-tuning strategy
        def unfreeze_and_compile(model, learning_rate=1e-5):
            """
            Gradual unfreezing for fine-tuning
            """
            # Unfreeze top layers
            base_model.trainable = True
            
            # Freeze all but top 20 layers
            for layer in base_model.layers[:-20]:
                layer.trainable = False
            
            # Recompile with lower learning rate
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        
        return model, unfreeze_and_compile
    
    @staticmethod
    def few_shot_learning(support_set, query_set, n_way=5, k_shot=5):
        """
        Prototypical networks for few-shot learning
        """
        import torch
        import torch.nn as nn
        
        class PrototypicalNet(nn.Module):
            def __init__(self, encoder):
                super(PrototypicalNet, self).__init__()
                self.encoder = encoder
            
            def forward(self, support, query):
                # Encode support and query
                z_support = self.encoder(support)
                z_query = self.encoder(query)
                
                # Compute prototypes (class centers)
                z_support = z_support.reshape(n_way, k_shot, -1)
                prototypes = z_support.mean(dim=1)
                
                # Compute distances
                distances = torch.cdist(z_query, prototypes)
                
                # Return negative distances (for softmax)
                return -distances
        
        # Simple CNN encoder
        encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 256)
        )
        
        model = PrototypicalNet(encoder)
        
        return model
```

### Interpretability and Biological Constraints

```python
class InterpretableDeepLearning:
    """
    Making deep learning interpretable for biology
    """
    
    @staticmethod
    def attention_visualization(model, input_sequence):
        """
        Visualize attention weights for sequence models
        """
        # Get attention weights
        attention_layer = model.get_layer('attention')
        attention_model = models.Model(
            inputs=model.input,
            outputs=attention_layer.output
        )
        
        attention_weights = attention_model.predict(input_sequence)
        
        # Visualize
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(20, 5))
        plt.imshow(attention_weights[0].T, cmap='hot', aspect='auto')
        plt.colorbar()
        plt.xlabel('Sequence Position')
        plt.ylabel('Attention Head')
        plt.title('Attention Weights')
        
        return attention_weights
    
    @staticmethod
    def grad_cam(model, image, class_idx, layer_name='conv5_block16_2_conv'):
        """
        Gradient-weighted Class Activation Mapping
        """
        import tensorflow as tf
        
        # Create model to output conv layer and predictions
        grad_model = models.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weighted combination
        cam = tf.reduce_sum(weights * conv_outputs, axis=-1)
        
        # ReLU and normalize
        cam = tf.nn.relu(cam)
        cam = cam / tf.reduce_max(cam)
        
        return cam.numpy()
    
    @staticmethod
    def integrated_gradients(model, baseline, input_image, steps=50):
        """
        Integrated gradients for feature attribution
        """
        import tensorflow as tf
        
        # Generate interpolated images
        alphas = tf.linspace(0.0, 1.0, steps + 1)
        
        interpolated = []
        for alpha in alphas:
            interpolated.append(
                baseline + alpha * (input_image - baseline)
            )
        interpolated = tf.concat(interpolated, axis=0)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            predictions = model(interpolated)
            
        gradients = tape.gradient(predictions, interpolated)
        
        # Approximate integral
        avg_gradients = tf.reduce_mean(gradients[:-1], axis=0)
        integrated_grads = (input_image - baseline) * avg_gradients
        
        return integrated_grads
```

### Biological Constraints and Custom Losses

```python
def biological_loss_functions():
    """
    Custom loss functions for biological constraints
    """
    import tensorflow as tf
    from tensorflow.keras import backend as K
    
    def survival_loss(y_true, y_pred):
        """
        Cox proportional hazards loss for survival analysis
        """
        # y_true: [time, event]
        # y_pred: risk scores
        
        time = y_true[:, 0]
        event = y_true[:, 1]
        
        # Partial likelihood
        risk = tf.exp(y_pred)
        
        # For each event, sum risk of those still at risk
        loss = 0
        for i in tf.where(event == 1):
            at_risk = tf.where(time >= time[i])
            loss -= y_pred[i] - tf.math.log(tf.reduce_sum(risk[at_risk]))
        
        return loss / tf.reduce_sum(event)
    
    def ordinal_loss(n_classes):
        """
        Loss for ordinal data (disease stages, grades)
        """
        def loss(y_true, y_pred):
            # Convert to cumulative probabilities
            cum_true = tf.cumsum(y_true, axis=1)[:, :-1]
            cum_pred = tf.cumsum(y_pred, axis=1)[:, :-1]
            
            # Binary crossentropy on cumulative
            return K.binary_crossentropy(cum_true, cum_pred)
        
        return loss
    
    def dice_loss(smooth=1e-6):
        """
        Dice loss for segmentation (handles class imbalance)
        """
        def loss(y_true, y_pred):
            y_true_flat = K.flatten(y_true)
            y_pred_flat = K.flatten(y_pred)
            
            intersection = K.sum(y_true_flat * y_pred_flat)
            union = K.sum(y_true_flat) + K.sum(y_pred_flat)
            
            dice = (2. * intersection + smooth) / (union + smooth)
            
            return 1 - dice
        
        return loss
    
    def physics_informed_loss(pde_weight=0.1):
        """
        Incorporate physical/biological equations
        """
        def loss(y_true, y_pred):
            # Data fitting term
            data_loss = K.mean(K.square(y_true - y_pred))
            
            # PDE residual (example: diffusion equation)
            # ∂u/∂t = D∇²u
            with tf.GradientTape() as tape:
                tape.watch(y_pred)
                dy_dx = tape.gradient(y_pred, x)
                d2y_dx2 = tape.gradient(dy_dx, x)
            
            pde_residual = dy_dt - D * d2y_dx2
            pde_loss = K.mean(K.square(pde_residual))
            
            return data_loss + pde_weight * pde_loss
        
        return loss
    
    return {
        'survival': survival_loss,
        'ordinal': ordinal_loss,
        'dice': dice_loss,
        'physics_informed': physics_informed_loss
    }
```

### Uncertainty Quantification

```python
class UncertaintyQuantification:
    """
    Uncertainty estimation in deep learning predictions
    """
    
    @staticmethod
    def monte_carlo_dropout(model, X, n_iterations=100):
        """
        MC Dropout for uncertainty estimation
        """
        # Enable dropout during inference
        predictions = []
        
        for _ in range(n_iterations):
            # Forward pass with dropout
            pred = model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Mean and uncertainty
        mean_prediction = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        
        # Epistemic (model) uncertainty
        epistemic = uncertainty.mean(axis=1)
        
        # Aleatoric (data) uncertainty
        aleatoric = predictions.var(axis=2).mean(axis=0)
        
        return {
            'mean': mean_prediction,
            'total_uncertainty': uncertainty,
            'epistemic': epistemic,
            'aleatoric': aleatoric
        }
    
    @staticmethod
    def ensemble_uncertainty(models, X):
        """
        Deep ensemble for uncertainty
        """
        predictions = []
        
        for model in models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Ensemble statistics
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # Predictive entropy
        mean_probs = predictions.mean(axis=0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=-1)
        
        # Mutual information (disagreement between models)
        individual_entropies = []
        for pred in predictions:
            ent = -np.sum(pred * np.log(pred + 1e-10), axis=-1)
            individual_entropies.append(ent)
        
        expected_entropy = np.mean(individual_entropies, axis=0)
        mutual_info = entropy - expected_entropy
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'entropy': entropy,
            'mutual_information': mutual_info
        }
```

### Complete Pipeline

```python
class BiologicalDeepLearningPipeline:
    """
    End-to-end pipeline for biological deep learning
    """
    
    def __init__(self, data_type='genomic', task='classification'):
        self.data_type = data_type
        self.task = task
        self.model = None
        self.history = None
        
    def prepare_data(self, X, y, validation_split=0.2):
        """
        Prepare biological data for deep learning
        """
        # Handle different data types
        if self.data_type == 'genomic':
            # Normalize
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            
        elif self.data_type == 'image':
            # Standardize to [0, 1]
            X = X / 255.0
            
        elif self.data_type == 'sequence':
            # Tokenize
            X = self.tokenize_sequences(X)
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y
        )
        
        return X_train, X_val, y_train, y_val
    
    def build_model(self, input_shape, n_outputs):
        """
        Build appropriate model for data type
        """
        if self.data_type == 'genomic':
            self.model = self.build_genomic_model(input_shape, n_outputs)
            
        elif self.data_type == 'image':
            self.model = biological_cnn(input_shape, n_outputs)
            
        elif self.data_type == 'sequence':
            self.model = biological_sequence_model(
                input_shape[0], input_shape[1], n_outputs
            )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, 
             epochs=100, batch_size=32):
        """
        Train with biological considerations
        """
        from tensorflow.keras.callbacks import (
            EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Class weights for imbalanced data
        if self.task == 'classification':
            from sklearn.utils.class_weight import compute_class_weight
            
            classes = np.unique(y_train)
            class_weights = compute_class_weight(
                'balanced',
                classes=classes,
                y=y_train.argmax(axis=1) if len(y_train.shape) > 1 else y_train
            )
            class_weight_dict = dict(zip(classes, class_weights))
        else:
            class_weight_dict = None
        
        # Compile
        if self.task == 'classification':
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )
        elif self.task == 'regression':
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict
        )
        
        return self.history
    
    def evaluate_biologically(self, X_test, y_test):
        """
        Biological evaluation metrics
        """
        predictions = self.model.predict(X_test)
        
        results = {}
        
        if self.task == 'classification':
            from sklearn.metrics import (
                roc_auc_score, precision_recall_curve, 
                confusion_matrix, classification_report
            )
            
            # Standard metrics
            results['auc'] = roc_auc_score(y_test, predictions, 
                                          multi_class='ovr')
            
            # Precision-recall (better for imbalanced)
            precision, recall, _ = precision_recall_curve(
                y_test.ravel(), predictions.ravel()
            )
            results['auprc'] = np.trapz(precision, recall)
            
            # Confusion matrix
            y_pred_class = predictions.argmax(axis=1)
            y_true_class = y_test.argmax(axis=1)
            results['confusion'] = confusion_matrix(y_true_class, y_pred_class)
            
            # Per-class performance
            results['classification_report'] = classification_report(
                y_true_class, y_pred_class
            )
            
        # Biological validation
        results['biological_validity'] = self.check_biological_constraints(
            predictions
        )
        
        return results
```

### References
- LeCun, Y. et al. (2015). Deep learning. Nature
- Esteva, A. et al. (2019). A guide to deep learning in healthcare
- Jumper, J. et al. (2021). Highly accurate protein structure prediction with AlphaFold
- Ching, T. et al. (2018). Opportunities and obstacles for deep learning in biology and medicine