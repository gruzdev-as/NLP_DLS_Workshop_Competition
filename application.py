import streamlit as st
import altair as alt
import numpy as np 
import pandas as pd
import time 
import random
import yaml
from yaml.loader import SafeLoader
from model_inference import Model_inference
from pathlib import Path

# Базовые настройки и константы
if 'models_loaded' not in st.session_state:
    st.session_state['models_loaded'] = False
if 'pseudo_inference' not in st.session_state:
    st.session_state['pseudo_inference'] = False
if 'model_inference' not in st.session_state:  # Initialize session state for model_inference
    st.session_state['model_inference'] = None

st.session_state['pseudo_inference'] = st.checkbox("Pseudo Inference", value=False)

with open('config.yaml', 'r', encoding='utf8') as f:
    config_yaml = yaml.load(f, Loader=SafeLoader)

TRENDS_DESCRIPTION = config_yaml['TRENDS']
TAGS_OPTIONS = config_yaml["TAGS"]

DATA_PATH = Path('data')

def main_interface():
    st.title("Программа оценки обратной связи пользователей")
    st.subheader("Подгрузите необходимые модели")
    
    BASEMODEL_PATH = st.text_input("Base Model Path", value="unsloth/gemma-2-27b-bnb-4bit")
    LORA_MODEL_PATH = st.text_input("Adapter Model Path", value="TheStrangerOne/gemma-2-27b-it-bnb-4bit-lora-multilabel")

    if st.button("Load Models"):
        if not st.session_state['pseudo_inference']:
            st.session_state['model_inference'] = Model_inference(BASEMODEL_PATH, LORA_MODEL_PATH)
            st.session_state['models_loaded'] = True 
        else:
            get_pseudo_inference()
    
    st.title("Пожалуйста, поделитесь с нами своей обратной связью")
    st.slider("Оценка сервиса", min_value=0, max_value=6, value=0, key="assessment_slider", disabled=not st.session_state['models_loaded'])
    st.multiselect("Тэги", options=TAGS_OPTIONS, key="tags", disabled=not st.session_state['models_loaded'])
    st.text_area("Ваш отзыв", max_chars=250, key="text_area", disabled=not st.session_state['models_loaded'])
    st.button("Отправить отзыв", key="Отправить отзыв", disabled=not st.session_state['models_loaded'])
    
    if st.session_state["Отправить отзыв"]:
        get_review()


def get_review():
    
    assessment = st.session_state["assessment_slider"]
    selected_tags = st.session_state["tags"]
    review = st.session_state["text_area"]

    if len(review) == 0:
        st.error("Review cannot be empty.")
    else:
        full_review = f'User assesment:{assessment}\n\nTags:{selected_tags}\n\nReview:{review}'
        predicted_probas, predicted_classes = st.session_state['model_inference'].inference(full_review)
        present_result(predicted_probas, predicted_classes)

def get_pseudo_inference():
    probas = np.genfromtxt(DATA_PATH / 'probs_gemma_all_train_27b.csv', delimiter=',')
    test = pd.read_csv(DATA_PATH / 'test.csv.csv')
    random_index = random.randint(0, len(probas))

    st.write(f"**Text:** {test.iloc[random_index]['text']}")
    st.write(f"**Tags:** {test.iloc[random_index]['tags']}")
    st.write(f"**Assessment:** {test.iloc[random_index]['assessment']}")

    classes = np.where(probas[random_index] > 0.55)[0]
    if len(classes) == 0:
        classes = np.where(probas[random_index] > 0.45)[0]
        if len(classes) == 0:
            classes = np.where(probas[random_index] > 0.35)[0]
            if len(classes) == 0:
                classes = np.where(probas[random_index] > 0.2)[0]
                if len(classes) == 0:
                    classes = np.array([19])

    present_result(probas[random_index], classes)

def present_result(predicted_probas:np.array, classes:np.array):
    placeholder = st.empty()
    time.sleep(1)
    
    st.subheader("Результаты прогнозирования")

    data = {
        'Labels': list(range(50)),
        'Values': predicted_probas
    }
    max_value = max(data['Values'])

    animation_df = pd.DataFrame({
        'Labels': data['Labels'],
        'Values': [0] * len(data['Values'])
    })

    for i in range(1, 101, 5):
        animation_df['Values'] = np.array(data['Values']) * (i / 100)
        chart = alt.Chart(animation_df).mark_bar().encode(
            x=alt.X('Labels'),
            y=alt.Y('Values', scale=alt.Scale(domain=[0, max_value]))  # Fix y-axis range
            ).properties(
                width=900,
                height=400
        )

        line = alt.Chart(pd.DataFrame({'y': [0.55]})).mark_rule(color='red').encode(
            y='y'
        )
        text = alt.Chart(pd.DataFrame({'y': [0.55], 'text': ['BASE PREDICTION THRESHOLD = 0.55']})).mark_text(
            align='left',
            baseline='bottom',
            dy=-5,  
            color='red'
            ).encode(
                y='y',
                text='text'
        )
        combined_chart = chart + line + text
        placeholder.altair_chart(combined_chart)

        time.sleep(0.01)
    
    st.write("Predicted trends:")
    for i in classes:
        st.write(f'{i}: {TRENDS_DESCRIPTION[i]}')

def main():
    main_interface()

if __name__ == "__main__":
    main()