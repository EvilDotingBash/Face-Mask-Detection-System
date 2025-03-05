from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the CNN model
mymodel = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(1, activation='sigmoid')  # Binary classification (mask/no mask)
])

mymodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Organize the data
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('train', target_size=(150,150), batch_size=16, class_mode='binary')
test_set = test_datagen.flow_from_directory('test', target_size=(150,150), batch_size=16, class_mode='binary')

# Train the model
mymodel.fit(train_set, epochs=10, validation_data=test_set)

# Save the trained model
mymodel.save('mask1.h5', save_format='h5')
print("Model saved successfully as mask.h5")
