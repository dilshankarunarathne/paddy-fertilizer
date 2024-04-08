def predict_image_class(image_data):
    from keras.models import load_model
    import numpy as np
    from PIL import Image, ImageOps

    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("lcc/keras_Model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Convert image data to PIL Image
    image = Image.fromarray(image_data).convert("RGB")

    # Resize the image to be at least 224x224 and then crop from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name[2:], confidence_score


def estimate_npa(lcc_score):
    pass
