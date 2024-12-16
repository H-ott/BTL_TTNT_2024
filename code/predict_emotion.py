import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt

# Tải mô hình đã train
model = load_model('Đường dẫn đến mô hình đã TRAIN') # /content/drive/MyDrive/BTL_AI/1/emotion_detection_model.h5

# Đường dẫn tới ảnh
img_path = '..\image_test\Training_2913.jpg'

# Chuyển ảnh sang grayscale và sau đó chuyển thành RGB
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh grayscale
img = cv2.resize(img, (48, 48))  # Resize về kích thước phù hợp với mô hình
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Chuyển từ grayscale sang RGB (3 kênh)
img = img / 255.0 
img = np.expand_dims(img, axis=0)

# Dự đoán cảm xúc
predictions = model.predict(img)
emotion = np.argmax(predictions)
plt.imshow(img[0])
plt.axis('off')
# In ra hình ảnh sau khi đưa vào mô hình
plt.show()
# In ra ma trận hình ảnh
print(img)
# In ra cảm xúc dự đoán
emotions_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
print(f"Predicted Emotion: {emotions_dict[emotion]}")
