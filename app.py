import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import string
from tensorflow import keras


# Cache models to prevent reloading
def load_model(model_path):
    return keras.models.load_model(model_path)

@st.cache_resource
def get_models():
    return {
        "ASL": load_model("model_asl.h5"),
        "ISL": load_model("model_isl.h5")
    }

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define the alphabet
asl_alphabet = ['0','1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)
isl_alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

# Sentence Construction
# Initialize session state if not already set
if "sentence" not in st.session_state:
    st.session_state.sentence = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "count" not in st.session_state:
    st.session_state.count = 0


def calc_landmark_list(image, landmarks):
    """Extracts hand landmarks from the image"""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    """Normalize the landmarks for model input"""
    base_x, base_y = landmark_list[0][0], landmark_list[0][1]
    temp_landmark_list = [[x - base_x, y - base_y] for x, y in landmark_list]
    temp_landmark_list = np.array(temp_landmark_list).flatten()
    max_value = np.max(np.abs(temp_landmark_list))
    return (temp_landmark_list / max_value).tolist()

def predict_sign(landmarks, model, alphabet):
    """Predicts the sign language gesture"""
    df = pd.DataFrame([landmarks])
    predictions = model.predict(df, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return alphabet[predicted_class]



def main():
    global sentence, last_prediction, count

    st.title("👋 Hand-Speak")
    st.sidebar.header("Settings ⚙️")

    # Load models
    models = get_models()

    # Model selection
    model_choice = st.sidebar.selectbox("Choose Category 👇", ["American Sign Language (ASL)", "Indian Sign Language (ISL)"])
    model_key = "ASL" if model_choice == "American Sign Language (ASL)" else "ISL"
    model = models[model_key]
    alphabet = asl_alphabet if model_key == "ASL" else isl_alphabet
    
    # Load and display image at the bottom of the sidebar
    st.sidebar.markdown("<br>" * 8, unsafe_allow_html=True)  # Push image down
    st.sidebar.image("Poster_Logo.png", use_column_width=True)

    hands = mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Start
    st.write("Sign Language Recognition")
    cap = cv2.VideoCapture(0)

    # Streamlit UI elements
    col1, col2 = st.columns([1, 1])
    with col1:
        frame_placeholder = st.empty()
    with col2:
        if(model_key == "ISL"):
            st.image("Poster_isl.png")
        else:
            st.image("Poster_asl.png")

    sentence_placeholder = st.empty()

    col3, col4, col5 = st.columns([1, 1, 1])


    with col3:
        if st.button("Space"):
            st.session_state.sentence.append(" ")
    with col4:
        if st.button("Backspace") and st.session_state.sentence:
            st.session_state.sentence.pop()

    with col5:
        if st.button("Clear All"):
            st.session_state.sentence = []  # Reset sentence list


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(frame, hand_landmarks)
                processed_landmarks = pre_process_landmark(landmark_list)
                prediction = predict_sign(processed_landmarks, model, alphabet)              
                
                # Count stable predictions
                if prediction == st.session_state.last_prediction:
                    st.session_state.count += 1
                else:
                    st.session_state.count = 1
                st.session_state.last_prediction = prediction
                
                if st.session_state.count == 10:  # Add to sentence after 10 consistent predictions
                    st.session_state.sentence.append(prediction)
                    st.session_state.count = 0  # Reset count

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                cv2.putText(frame, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        
        frame_placeholder.image(frame, channels="BGR")
        sentence_placeholder.text("Sentence: " + "".join(st.session_state.sentence))

    cap.release()
    hands.close()
    
if __name__ == "__main__":
    main()
