# Environment Setup
- **Python Version:** 3.10
- **Virtual Environment:** Created using `python3.10 -m venv <env_name>`
- **Reason:** For Tensorflow

# Model Configuration

## Baseline CNN Model

### Input Shape
- **Shape:** (224, 224, 3)
- **Reason:** Standard input size for many images

### Layers
1. **Conv2D Layer 1**
   - **Filters:** 32
   - **Kernel Size:** (3, 3)
   - **Activation:** ReLU
   - **Reason:** Initial feature

2. **BatchNormalization Layer 1**

3. **MaxPooling2D Layer 1**
   - **Pool Size:** (2, 2)
   - **Reason:** Downsampling

4. **Conv2D Layer 2**
   - **Filters:** 64
   - **Kernel Size:** (3, 3)
   - **Activation:** ReLU
   - **Reason:** Increased filters for complex feature extraction

5. **BatchNormalization Layer 2**

6. **MaxPooling2D Layer 2**
   - **Pool Size:** (2, 2)
   - **Reason:** Further downsampling.

7. **Conv2D Layer 3**
   - **Filters:** 128
   - **Kernel Size:** (3, 3)
   - **Activation:** ReLU

8. **BatchNormalization Layer 3**

9. **MaxPooling2D Layer 3**
   - **Pool Size:** (2, 2)
   - **Reason:** Final downsampling before flattening.

10. **Flatten Layer**

11. **Dense Layer 1**
    - **Units:** 128
    - **Activation:** ReLU

12. **BatchNormalization Layer 4**

13. **Dropout Layer**
    - **Rate:** 0.5

14. **Dense Layer 2**
    - **Units:** 3 num_classses
    - **Activation:** softmax

### Compilation
- **Optimizer:** Adam
  - **Learning Rate:** 0.0001
  - **Reason:** Adaptive learning rate optimization, default choice for many problems
- **Loss Function:** Sparse Categorical Crossentropy
  - **Reason:** Suitable for multi-class classificat
- **Metrics:** Accuracy


## ResNet Model

### Input Shape
- **Shape:** (224, 224, 3)
- **Reason:** Standard input size for many images

### Base Model
- **Model:** ResNet50
  - **Weights:** Pre-trained on ImageNet
  - **Include Top:** False
  - **Trainable:** False
  - **Reason:** Use a pre-trained model for feature extraction and fine-tuning.

### Layers
1. **GlobalAveragePooling2D Layer**
   - **Reason:** Reduces each feature map to a single value, reducing param numbers,  preventing overfitting

2. **Dense Layer 1**
   - **Units:** 128
   - **Activation:** ReLU

3. **Dropout Layer**
   - **Rate:** 0.5
   - **Reason:** Regularization

4. **Dense Layer 2**
   - **Units:** 3 (number of classes)
   - **Activation:** Softmax
   - **Reason:** Output layer for multi-class classification

### Compilation
- **Optimizer:** Adam
  - **Learning Rate:** 0.0001
  - **Reason:** Adaptive learning rate optimization, default choice for many problems.
- **Loss Function:** Sparse Categorical Crossentropy
  - **Reason:** Suitable for multi-class classification
- **Metrics:** Accuracy

## Data Augmentation

### Custom CNN Data Generators
- **Rescale:** 1./255
  - **Reason:** Normalize pixel values to [0, 1]
- **Rotation Range:** 15 degrees
- **Width Shift Range:** 0.2
- **Height Shift Range:** 0.2
- **Shear Range:** 0.2
- **Zoom Range:** 0.2
- **Horizontal Flip:** True

### ResNet Data Generators
- **Preprocessing Function:** Normalize using ImageNet mean and standard deviat

## Training Parameters
- **Epochs:** 40
- **Batch Size:** 32

## Callbacks
- **EarlyStopping**
  - **Monitor:** val_loss
  - **Patience:** 5
  - **Restore Best Weights:** True

- **ModelCheckpoint**
  - **Monitor:** val_loss
  - **Save Best Only:** True
  - **Mode:** min

## Class Weights
- **Reason:** Handle class imbalance by assigning higher weights to less frequent classes.

## etc
- **Data Oversampling:** Used RandomOverSampler to balance the dataset.
- **Data Generators:** Used ImageDataGenerator for data augmentation and normalization.
- **Overfitting:** Performance may improve if the model overfits.