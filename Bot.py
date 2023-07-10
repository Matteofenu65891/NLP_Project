import os
import PreProcessing as pp
from matplotlib import pyplot as plt
import Main
import numpy as np
import telebot
from telebot import types
import random
import pandas as pd

BOT_TOKEN = '6300330428:AAF5n-6htRZQMLyqQ2TMp7f-olVJC861hU8'

bot = telebot.TeleBot(BOT_TOKEN)

array_last_predictions = []

model = Main.loadModel('linearSVC.sav')

keyboard = types.InlineKeyboardMarkup()

last_bot_message = None
last_user_message = None

def check_if_model_trained():
    return model is None


def error_logger(chat_id, error_message):
    unified_message_sender(chat_id, "Oops, something went wrong\n{}".format(error_message))


def unified_message_sender(chat_id, message, keyboard_markup=None, parse_mode=None):
    global last_bot_message
    sent_message = bot.send_message(chat_id, message, reply_markup=keyboard_markup, parse_mode=parse_mode)
    last_bot_message = sent_message



def register_user_feedback(text, label):
    try:
        new_element = [text, label]
        file_path = os.path.join(os.path.dirname(__file__), 'dataset', 'banking-training-user.csv')
        new_dataframe = pd.DataFrame([new_element])
        new_dataframe.to_csv(file_path, mode='a', header=False, index=False)

    except FileNotFoundError:
        print("Parquet file not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


@bot.message_handler(commands=['start'])
def send_welcome(message):
    keyboard_greet = types.InlineKeyboardMarkup()
    unified_message_sender(message.chat.id, "Hi,make me a question",
                           keyboard_markup=keyboard_greet)


@bot.message_handler(commands=['stop'])
def send_goodbye(message):
    unified_message_sender(message.chat.id, "Goodbye")
    bot.stop_polling()


@bot.message_handler(func=lambda msg: True)
def answer_request(message):
    try:
        if check_if_model_trained():
            unified_message_sender(message.chat.id, "No trained model")
            return
        global last_user_message
        last_user_message = message
        text_preprocessed = pp.CleanText(message.text)

        y_pred = Main.PredictAnswers(text_preprocessed,model)
        unified_message_sender(message.chat.id,y_pred,parse_mode='HTML', keyboard_markup=keyboard)
    except Exception as e:
        unified_message_sender(message.chat.id, str(e))


bot.infinity_polling()