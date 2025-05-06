import os
import cv2
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import Sequential, Model, load_model
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.applications import ResNet50


# ImageNet Normalization Parameters for ResNet
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# GPU for mac
def setup_gpu():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("running on GPU")
        else:
            print("running on CPU")
    except Exception as e:
        print("running on CPU")


# Load Data and Preprocess
def load_data(dataset_dir):
    categories = {"IC": 0, "Capacitor": 1, "Resistor": 2}
    image_paths, labels = [], []

    for category, label in categories.items():
        category_path = os.path.join(dataset_dir, category)
        if os.path.exists(category_path):
            files = glob.glob(os.path.join(category_path, "*.jpeg")) + \
                    glob.glob(os.path.join(category_path, "*.jpg")) + \
                    glob.glob(os.path.join(category_path, "*.png"))
            image_paths.extend(files)
            labels.extend([label] * len(files))
            print(f"{category}: {len(files)} images found")
        else:
            raise FileNotFoundError(f"{category_path} not found")

    print(f"Total Images: {len(image_paths)}")

    df = pd.DataFrame({"image_path": image_paths, "label": labels})
    df["label"] = df["label"].astype(str)

    return df, categories


# Filter Invalid Images
def is_valid_image(image_path):
    return os.path.exists(image_path) and cv2.imread(image_path) is not None    # type: ignore


# Data Splitting
def split_data(df):
    # Oversampling before splitting
    ros = RandomOverSampler(random_state=42)
    image_paths, labels = ros.fit_resample(np.array(df["image_path"]).reshape(-1, 1), df["label"])
    image_paths = image_paths.flatten()
    df = pd.DataFrame({"image_path": image_paths, "label": labels})
    df["label"] = df["label"].astype(str)

    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

    # Remove invalid images
    train_df = train_df[train_df["image_path"].apply(is_valid_image)].reset_index(drop=True)
    val_df = val_df[val_df["image_path"].apply(is_valid_image)].reset_index(drop=True)
    test_df = test_df[test_df["image_path"].apply(is_valid_image)].reset_index(drop=True)

    print(f"Filtered Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df

# ResNet Preprocessing
def preprocess_resnet_image(x):
    return (x / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

# Data Generators
def create_data_generators(train_df, val_df, test_df, batch_size=32):
    # Augmentation only for Training Set
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    val_test_gen = ImageDataGenerator(rescale=1./255)

    # Resnet Normalization
    train_resnet_gen = ImageDataGenerator(preprocessing_function=preprocess_resnet_image)
    val_resnet_gen = ImageDataGenerator(preprocessing_function=preprocess_resnet_image)
    
    return (
        train_gen.flow_from_dataframe(train_df, x_col="image_path", y_col="label", target_size=(224, 224), batch_size=batch_size, class_mode="sparse"),
        val_test_gen.flow_from_dataframe(val_df, x_col="image_path", y_col="label", target_size=(224, 224), batch_size=batch_size, class_mode="sparse"),
        val_test_gen.flow_from_dataframe(test_df, x_col="image_path", y_col="label", target_size=(224, 224), batch_size=batch_size, class_mode="sparse", shuffle=False),
        train_resnet_gen.flow_from_dataframe(train_df, x_col="image_path", y_col="label", target_size=(224, 224), batch_size=batch_size, class_mode="sparse"),
        val_resnet_gen.flow_from_dataframe(val_df, x_col="image_path", y_col="label", target_size=(224, 224), batch_size=batch_size, class_mode="sparse"),
        val_resnet_gen.flow_from_dataframe(test_df, x_col="image_path", y_col="label", target_size=(224, 224), batch_size=batch_size, class_mode="sparse", shuffle=False)
    )
    
    # # Custom CNN Data Generators
    # train_cnn_generator = train_gen.flow_from_dataframe(train_df, x_col="image_path", y_col="label", target_size=(224, 224), batch_size=batch_size, class_mode="sparse")
    # val_cnn_generator = val_test_gen.flow_from_dataframe(val_df, x_col="image_path", y_col="label", target_size=(224, 224), batch_size=batch_size, class_mode="sparse")
    # test_cnn_generator = val_test_gen.flow_from_dataframe(test_df, x_col="image_path", y_col="label", target_size=(224, 224), batch_size=batch_size, class_mode="sparse", shuffle=False)

    # # ResNet Data Generators
    # train_resnet_generator = train_resnet_gen.flow_from_dataframe(train_df, x_col="image_path", y_col="label", target_size=(224, 224), batch_size=batch_size, class_mode="sparse")
    # val_resnet_generator = val_resnet_gen.flow_from_dataframe(val_df, x_col="image_path", y_col="label", target_size=(224, 224), batch_size=batch_size, class_mode="sparse")
    # test_resnet_generator = val_resnet_gen.flow_from_dataframe(test_df, x_col="image_path", y_col="label", target_size=(224, 224), batch_size=batch_size, class_mode="sparse", shuffle=False)

    # return train_cnn_generator, val_cnn_generator, test_cnn_generator, train_resnet_generator, val_resnet_generator, test_resnet_generator


# Baseline CNN Model
def build_baseline_cnn(input_shape=(224, 224, 3), num_classes=3):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ResNet Model
def build_resnet_model(input_shape=(224, 224, 3), num_classes=3):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# Train and Evaluate Models
def train_and_evaluate_model(model, model_name, train_gen, val_gen, test_gen):
    model_save_path = f"./model/{model_name}.keras"
    checkpoint = ModelCheckpoint(model_save_path, monitor="val_loss", save_best_only=True, mode="min", verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(train_gen, validation_data=val_gen, epochs=40, verbose=1, callbacks=[early_stopping, checkpoint])
    model = load_model(model_save_path)
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"{model_name} - Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    return model, history

# Plot Accuracy and Loss
def plot_training_history(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy Graph
    axes[0].plot(history.history["accuracy"], label=f"{model_name} Train Accuracy", linestyle="-")
    axes[0].plot(history.history["val_accuracy"], label=f"{model_name} Validation Accuracy", linestyle="--")
    axes[0].set_title(f"{model_name} Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss Graph
    axes[1].plot(history.history["loss"], label=f"{model_name} Train Loss", linestyle="-")
    axes[1].plot(history.history["val_loss"], label=f"{model_name} Validation Loss", linestyle="--")
    axes[1].set_title(f"{model_name} Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.show()

# Evaluate Model
def evaluate_model(model, test_gen, model_name):
    CATEGORIES = {"IC": 0, "Capacitor": 1, "Resistor": 2}
    
    y_true = test_gen.classes
    y_pred = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_true, y_pred_classes))

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CATEGORIES.keys(), yticklabels=CATEGORIES.keys())
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()


def main():
    # GPU, Dataset Path
    setup_gpu()
    dataset_dir = "./Dataset"
    
    df, categories = load_data(dataset_dir)
    train_df, val_df, test_df = split_data(df)
    train_cnn_gen, val_cnn_gen, test_cnn_gen, train_resnet_gen, val_resnet_gen, test_resnet_gen = create_data_generators(train_df, val_df, test_df)

    print("Data Generators created successfully.")

    # Build Models
    baseline_cnn = build_baseline_cnn()
    resnet_model = build_resnet_model()

    # Train Models
    baseline_cnn, baseline_cnn_history = train_and_evaluate_model(baseline_cnn, "baseline_cnn", train_cnn_gen, val_cnn_gen, test_cnn_gen)
    resnet_model, resnet_model_history = train_and_evaluate_model(resnet_model, "resnet_model", train_resnet_gen, val_resnet_gen, test_resnet_gen)

    # Plot Training History
    plot_training_history(baseline_cnn_history, "Baseline CNN")
    plot_training_history(resnet_model_history, "ResNet")

    # Evaluate Models
    evaluate_model(baseline_cnn, test_cnn_gen, "Baseline CNN")
    evaluate_model(resnet_model, test_resnet_gen, "ResNet Model")


if __name__ == "__main__":
    main()
