# Image Processing

## Process
1. Preprocessing (Convert Color, Noise Reduction/Smooth)
2. Thresholding (Binary, Edge Detect)
3. Masking Color
4. Contours, Bounding Box (Remove Redundant)
5. Save Segmentation Output (ROI Extraction)

---

### Preprocessing:
1. **Color Conversion:**
   - **Reason:** RGB space is sensitive to lighting changes, making color-based object detection difficult.
   - **Technique:** HSV (Hue, Saturation, Value) conversion.
     - **H(Hue):** Color information -> Can detect specific color components
     - **S(Saturation):** Color purity -> Distinguish clear colors.
     - **V(Value):** Brightness -> Reduces impact of lighting changes.

2. **Noise Reduction/Smoothing:**
   - **Reason:** Minor variations in background can generate unnecessary features.
   - **Techniques:** 
     - **Gaussian Blur:** Removes small noise while maintaining edge portions.
     - **Morphological Operations:** `cv2.morphologyEx()` (Opening/Closing) to refine filtered mask.

---

### Thresholding, Edge Detect:
1. **Thresholding:**
   - **Reason:** Segmentation requires distinguishing between foreground and background pixels.
   - **Techniques:**
     - **Otsu Thresholding:** Automatically determines the optimal threshold value.
     - **Adaptive Thresholding:** Adjusts threshold locally based on image regions.
     - **Adaptive vs Otsu Selection Logic:** Computes mean values of both results and selects the better one.

2. **Edge Detection:**
   - **Reason:** Object boundaries need to be clearly detected for accurate segmentation.
   - **Techniques:**
     - **Canny Edge Detection:** Finds strong edges while reducing noise.
     - **Sobel Operator:** Detects horizontal/vertical gradients to enhance edge clarity.

---

### Masking Color:
1. **HSV Color Filter**
   - **Reason:** Images (Resistor, Capacitor) typically have distinct colors, making color-based segmentation effective.
   - **Techniques:**
     - Convert image to HSV color space.
     - Define lower/upper bounds for each target color.
     - Apply `cv2.inRange()` to create a binary mask of detected color.
     - Apply **Morphological Operations** (Closing & Opening) to remove noise.

---

### Contours & Bounding Box:
1. **Contour Detection**
   - **Reason:** Once objects are segmented, their shape must be analyzed.
   - **Techniques:**
     - `cv2.findContours()` extracts object boundaries.
     - `cv2.RETR_EXTERNAL` retrieves only the outer contours.

2. **Bounding Box Extraction**
   - **Reason:** After detecting contours, objects must be isolated.
   - **Techniques:**
     - `cv2.boundingRect()` creates a rectangle around detected objects.
     - **Rotated Bounding Box** can be applied for non-axis-aligned objects.

3. **Remove Redundant Contours**
   - **Reason:** Prevents multiple bounding boxes around the same object.
   - **Techniques:**
     - Compare `(x, y)` positions of detected boxes.
     - If two boxes are too close, keep only the larger one.

---

### Segmentation Output:
1. **Region of Interest (ROI) Extraction**
   - **Reason:** To save separately detected components.
   - **Techniques:**
     - Crop/Extract bounding box from detected contours.
     - Save individual component images.
