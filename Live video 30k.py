# Create a VideoCapture object to capture video from the default camera
import base64
import json
import collections
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from flask import Flask, render_template, request
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import pyttsx3
app = Flask(__name__)
model=load_model('model_weights_30k/model_19_30k.h5')
from keras.applications import ResNet50
model2=ResNet50(weights="imagenet",input_shape=(224,224,3))
from keras import Model
new_model=Model(model2.input,model2.layers[-2].output)
def preprocess_img(img):
    img=image.load_img(img,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)#for reshaping
    #Normalization
    img=preprocess_input(img)
    return img
def encode_image(img):
    img = preprocess_img(img)
    feature_vector = new_model.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    return feature_vector


def get_caption(photo):
    descriptions = None
    with open('descriptions__30k.txt', 'r') as f:
        descriptions = f.read()
    json_string = descriptions.replace("'", "\"")
    descriptions = json.loads(json_string)
    total_words = []
    for key in descriptions.keys():
        [total_words.append(i) for des in descriptions[key] for i in des.split()]
    counter = collections.Counter(total_words)
    freq_count = dict(counter)
    sorted_freq_count = sorted(freq_count.items(), reverse=True, key=lambda x: x[1])
    threshold = 10
    sorted_freq_count = [x for x in sorted_freq_count if x[1] > threshold]
    total_words = [x[0] for x in sorted_freq_count]
    word_to_idx = {}
    idx_to_word = {}
    for i, word in enumerate(total_words):
        word_to_idx[word] = i + 1
        idx_to_word[i + 1] = word
    idx_to_word[5119] = "startseq"
    idx_to_word[5120] = "endseq"
    word_to_idx["startseq"] = 5119
    word_to_idx['endseq'] = 5120
    vocab_size = len(word_to_idx) + 1
    maxlen=74

    in_text='startseq'
    for i in range(maxlen):
        sequence=[word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence=pad_sequences([sequence],maxlen=maxlen,padding='post')
        ypred=model.predict([photo,sequence])
        ypred=ypred.argmax()#gives word with maximum probability--greedy sampling
        word=idx_to_word[ypred]
        in_text+=' '+word
        if word=='endseq':
            break
    final_caption=in_text.split()[1:-1]
    final_caption=' '.join(final_caption)
    return final_caption
import cv2
import pyttsx3
import threading
import time

# Function to perform video processing
def process_video(video_capture):
    # Set the camera properties
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize variables
    frame_count = 0
    start_time = time.time()

    # Example function to get caption based on frame number
    while True:
        # Read a frame from the video capture
        ret, frame = video_capture.read()

        # Check if the frame reading was successful
        if not ret:
            print("Failed to read a frame.")
            break

        # Calculate the elapsed time
        elapsed_time = time.time() - start_time

        # Check if the desired interval has passed
        if elapsed_time >= interval:
            # Save the frame as an image file
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

            # Get the caption for the current frame
            photo_2048 = encode_image(frame_path).reshape((1, 2048))
            caption = get_caption(photo_2048)

            # Reset the start time and increment the frame count
            start_time = time.time()
            frame_count += 1

            # Add the caption to the frame
            frames_with_captions.append((frame, caption))

    # Release the video capture object
    video_capture.release()

# Function to perform audio insertion
def insert_audio():
    engine = pyttsx3.init()

    while True:
        if frames_with_captions:
            frame, caption = frames_with_captions.pop(0)

            # Add the caption to the frame
            cv2.putText(frame, caption, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Speak the caption
            engine.say(caption)
            engine.runAndWait()

            # Display the frame
            cv2.imshow("Live Video", frame)

        # Check for keyboard interrupt (press 'q' to exit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close any open windows
    cv2.destroyAllWindows()

# Main program
if __name__ == "__main__":
    # Create a folder to store the extracted frames
    output_folder = "frames_output"
    os.makedirs(output_folder, exist_ok=True)

    # Initialize variables
    frames_with_captions = []
    interval = 10

    # Create a video capture object
    video_capture = cv2.VideoCapture(0)

    # Check if the video capture is successfully opened
    if not video_capture.isOpened():
        print("Failed to open video capture.")
        exit()

    # Create and start the video processing thread
    video_thread = threading.Thread(target=process_video, args=(video_capture,))
    video_thread.start()

    # Start the audio insertion function in the main thread
    insert_audio()

    # Wait for the video processing thread to finish
    video_thread.join()