import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'intel_classifier.keras')

def predict_single_image(image_path, model_path=DEFAULT_MODEL_PATH, show_plot=False):
    if not os.path.exists(image_path):
        print(f"Error: image not found at '{image_path}'")
        sys.exit(1)
        
    if not os.path.exists(model_path):
        print(f"Error: model not found at '{model_path}'")
        sys.exit(1)

    print(f"Loading model from: {model_path} ...")
    model = keras.models.load_model(model_path)

    print(f"Processing image: {image_path} ...")
    img = keras.utils.load_img(image_path, target_size=(150, 150))
    img_array = keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch, verbose=0)
    pred_index = np.argmax(predictions[0])
    pred_class = CLASS_NAMES[pred_index]
    confidence = np.max(predictions[0]) * 100

    print("--------------------------------------------------")
    print(f"Predicted Class : {pred_class.upper()}")
    print(f"Confidence      : {confidence:.2f}%")
    print("--------------------------------------------------")
    print("All Probabilities:")
    for cls, prob in zip(CLASS_NAMES, predictions[0]):
        bar = "█" * int(prob * 20)
        print(f"  {cls:<10}: {prob * 100:6.2f}% {bar}")
    print("--------------------------------------------------")

    if show_plot:
        plt.imshow(img)
        plt.title(f"Prediction: {pred_class} ({confidence:.2f}%)")
        plt.axis("off")
        plt.show()

    return pred_class, confidence

def main():
    parser = argparse.ArgumentParser(description="Classify an image using the trained Intel Image CNN model.")
    parser.add_argument("image_path", nargs="?", default="data/seg_pred/seg_pred/10004.jpg",
                        help="Path to the image to classify (default: data/seg_pred/seg_pred/10004.jpg)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="Path to the trained .keras model file")
    parser.add_argument("--show", action="store_true",
                        help="Display the image with prediction title in a matplotlib window")
    args = parser.parse_args()

    predict_single_image(args.image_path, model_path=args.model, show_plot=args.show)

if __name__ == "__main__":
    main()
