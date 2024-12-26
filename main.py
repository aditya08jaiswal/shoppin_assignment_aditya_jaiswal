import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_curve, average_precision_score
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# 1. Data Loading and Preprocessing
def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    return x_train, y_train, x_test, y_test

# 2. ResNet50 Feature Extraction
def create_resnet_feature_extractor():
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return Model(inputs=base_model.input, outputs=base_model.output)

def extract_resnet_features(images, model):
    processed_images = [tf.image.resize(np.repeat(img, 3, axis=-1), (224, 224)) for img in images]
    processed_images = np.array(processed_images)
    features = model.predict(processed_images)
    return features.reshape(features.shape[0], -1)

# Fine-Tune ResNet50
def finetune_resnet50(x_train, y_train, x_val, y_val, num_classes=10):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True  # Allow fine-tuning the entire model

    # Add classification layers
    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Preprocess data for ResNet50
    x_train_resized = np.array([tf.image.resize(np.repeat(img, 3, axis=-1), (224, 224)) for img in x_train])
    x_val_resized = np.array([tf.image.resize(np.repeat(img, 3, axis=-1), (224, 224)) for img in x_val])

    # Train the model
    model.fit(x_train_resized, y_train, validation_data=(x_val_resized, y_val), epochs=5, batch_size=32)

    return model

# Evaluate Fine-Tuned ResNet50
def evaluate_finetuned_resnet50(model, x_test, y_test):
    x_test_resized = np.array([tf.image.resize(np.repeat(img, 3, axis=-1), (224, 224)) for img in x_test])
    predictions = model.predict(x_test_resized)
    predicted_labels = np.argmax(predictions, axis=1)

    # Use the original evaluate_retrieval function
    precision = evaluate_retrieval(predictions, y_test)
    return precision

# 3. Siamese Network
def create_pairs(x, y, num_pairs=10000):
    pairs, labels = [], []
    classes = np.unique(y)
    for _ in range(num_pairs):
        idx1 = np.random.randint(0, len(x))
        img1, label1 = x[idx1], y[idx1]
        if np.random.rand() > 0.5:
            idx2 = np.random.choice(np.where(y == label1)[0])
            img2, label = x[idx2], 1
        else:
            label2 = np.random.choice(classes[classes != label1])
            idx2 = np.random.choice(np.where(y == label2)[0])
            img2, label = x[idx2], 0
        pairs.append([img1, img2])
        labels.append(label)
    return np.array(pairs), np.array(labels)

def build_siamese_model(input_shape=(28, 28, 1)):
    input = Input(input_shape)
    x = Conv2D(64, (3, 3), activation='relu')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    model = Model(inputs=input, outputs=x)

    input_a, input_b = Input(input_shape), Input(input_shape)
    processed_a, processed_b = model(input_a), model(input_b)
    # Compute absolute difference using a Keras layer
    distance = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([processed_a, processed_b])
    output = Dense(1, activation='sigmoid')(distance)
    siamese_model = Model(inputs=[input_a, input_b], outputs=output)
    siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001))
    return siamese_model

def train_siamese_network(x_train, y_train):
    pairs_train, labels_train = create_pairs(x_train, y_train)
    model = build_siamese_model(input_shape=(28, 28, 1))
    model.fit([pairs_train[:, 0], pairs_train[:, 1]], labels_train, batch_size=64, epochs=5, validation_split=0.2)
    return model

def evaluate_siamese_model(siamese_model, x_test, y_test):
    pairs_test, labels_test = create_pairs(x_test, y_test, num_pairs=2000)
    predictions = siamese_model.predict([pairs_test[:, 0], pairs_test[:, 1]])
    precision, recall, thresholds = precision_recall_curve(labels_test, predictions)
    avg_precision = average_precision_score(labels_test, predictions)

    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve (Avg Precision = {avg_precision:.2f})')
    plt.show()

    return avg_precision

# 4. Autoencoder
def create_autoencoder():
    input_img = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    return autoencoder, encoder

# 5. CLIP for Feature Extraction
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

def extract_clip_features(images):
    inputs = clip_processor(images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    return outputs.numpy()

# 6. Evaluation
def evaluate_retrieval(features, labels, n_neighbors=5):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(features)
    distances, indices = nbrs.kneighbors(features)
    precision_at_k = []
    for i in range(len(labels)):
        relevant = sum(labels[indices[i][1:]] == labels[i])
        precision = relevant / (n_neighbors - 1)
        precision_at_k.append(precision)
    return np.mean(precision_at_k)

# Main Execution
def main():
    x_train, y_train, x_test, y_test = load_fashion_mnist()

    # ResNet50 Features
    print("Extracting ResNet50 features...")
    resnet_model = create_resnet_feature_extractor()
    resnet_features = extract_resnet_features(x_train[:1000], resnet_model)
    resnet_precision = evaluate_retrieval(resnet_features, y_train[:1000])
    print(f"ResNet Precision: {resnet_precision:.3f}")

    # Fine-Tune ResNet50
    # print("Fine-Tuning ResNet50...")
    # val_split = 0.2  # Split for validation
    # val_size = int(len(x_train) * val_split)
    # x_val, y_val = x_train[:val_size], y_train[:val_size]
    # x_train_finetune, y_train_finetune = x_train[val_size:], y_train[val_size:]

    # finetuned_resnet_model = finetune_resnet50(x_train_finetune, y_train_finetune, x_val, y_val)
    # finetuned_resnet_precision = evaluate_finetuned_resnet50(finetuned_resnet_model, x_test, y_test)
    # print(f"Fine-Tuned ResNet Precision@5: {finetuned_resnet_precision:.3f}")

    # Siamese Network
    print("Training Siamese Network...")
    siamese_model = train_siamese_network(x_train, y_train)
    siamese_precision = evaluate_siamese_model(siamese_model, x_test, y_test)
    print(f"Siamese Network Avg Precision: {siamese_precision:.3f}")

    # Autoencoder Features
    print("Training Autoencoder...")
    autoencoder, encoder = create_autoencoder()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train, epochs=5, batch_size=128, validation_split=0.2)
    encoded_features = encoder.predict(x_train[:1000]).reshape(len(x_train[:1000]), -1)
    autoencoder_precision = evaluate_retrieval(encoded_features, y_train[:1000])
    print(f"Autoencoder Precision: {autoencoder_precision:.3f}")

    # CLIP Features
    print("Extracting CLIP features...")
    x_train_pil = [Image.fromarray((img.squeeze() * 255).astype(np.uint8)) for img in x_train]
    clip_features = extract_clip_features(x_train_pil[:1000])
    clip_precision = evaluate_retrieval(clip_features, y_train[:1000])
    print(f"CLIP Precision: {clip_precision:.3f}")

if __name__ == "__main__":
    main()