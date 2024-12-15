import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Xây dựng mô hình CNN
model = Sequential()

# Thêm lớp Conv2D và MaxPooling2D
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))  # 48x48 hình ảnh RGB
model.add(MaxPooling2D((2, 2)))

# Thêm các lớp Conv2D và MaxPooling2D khác
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Chuyển đổi hình ảnh thành một vector
model.add(Flatten())

# Thêm lớp Dense với 128 neuron và Dropout
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout để tránh overfitting

# Lớp đầu ra với 7 lớp cảm xúc (0-6)
model.add(Dense(7, activation='softmax'))  # 7 lớp tương ứng với 7 cảm xúc

# Biên dịch mô hình
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Xem mô hình
model.summary()

# Chuẩn bị dữ liệu và tăng cường dữ liệu (data augmentation)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, shear_range=0.2)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Đọc dữ liệu từ thư mục train và validation
train_generator = train_datagen.flow_from_directory('..\tap_du_lieu\train',
                                                   target_size=(48, 48),
                                                   batch_size=32,
                                                   class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory('..\tap_du_lieu\validation',
                                                             target_size=(48, 48),
                                                             batch_size=32,
                                                             class_mode='categorical')

# Huấn luyện mô hình
history = model.fit(train_generator,
                    epochs=25,
                    validation_data=validation_generator)

# Lưu mô hình đã huấn luyện
model.save('..\mo_hinh_da_train\emotion_detection_model.h5')

# Đánh giá mô hình trên tập kiểm tra (test set)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('..\tap_du_lieu\test',
                                                  target_size=(48, 48),
                                                  batch_size=32,
                                                  class_mode='categorical')

score = model.evaluate(test_generator)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")

# Vẽ đồ thị biểu diễn độ chính xác và mất mát trong quá trình huấn luyện
plt.figure(figsize=(12, 6))

# Đồ thị độ chính xác
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Đồ thị mất mát
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()