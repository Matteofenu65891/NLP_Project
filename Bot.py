import os
import PreProcessing as pp
import Main
import telebot
from telebot import types
import pandas as pd

BOT_TOKEN = '6300330428:AAF5n-6htRZQMLyqQ2TMp7f-olVJC861hU8'
bot = telebot.TeleBot(BOT_TOKEN)

model1 = Main.loadModel('LinearSVCNaive.sav')
model2 = Main.loadModel('LinearSVC.sav')

keyboard = types.InlineKeyboardMarkup()

id_literal="Literal (string,number,date...)"
id_resource="Resource (City,Person,Company...) "
id_boolean="Boolean (What else?)"
id_none="I don't know"
buttonLiteral=types.KeyboardButton(id_literal)
buttonResource = types.KeyboardButton(id_resource)
buttonBoolean=types.KeyboardButton(id_boolean)
buttonNull=types.KeyboardButton(id_none)


keyboardButtons = types.ReplyKeyboardMarkup(one_time_keyboard=True)
keyboardButtons.add(buttonLiteral)
keyboardButtons.add(buttonResource)
keyboardButtons.add(buttonBoolean)
keyboardButtons.add(buttonNull)

last_user_message = None

def check_if_model_trained():
    return model1 is None and model2 is None


def error_logger(chat_id, error_message):
    unified_message_sender(chat_id, "Oops, something went wrong\n{}".format(error_message))


def unified_message_sender(chat_id, message, keyboard_markup=None, parse_mode=None):
    bot.send_message(chat_id, message, reply_markup=keyboard_markup, parse_mode=parse_mode)



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
    global last_user_message
    last_user_message=None
    unified_message_sender(message.chat.id, "Hi, make me a question",
                           keyboard_markup=keyboard)


@bot.message_handler(commands=['stop'])
def send_goodbye(message):
    unified_message_sender(message.chat.id, "Goodbye")
    global last_user_message
    last_user_message=None
    bot.stop_polling()


@bot.message_handler(func=lambda msg: True)
def answer_request(message):
    try:
        if check_if_model_trained():
            unified_message_sender(message.chat.id, "No model found")
            return
        global last_user_message
        if(last_user_message==None):
            last_user_message = message
            unified_message_sender(message.chat.id, "Can You tell me the category of the answer?", keyboard_markup=keyboardButtons)
        else:
            text_preprocessed = pp.CleanText(last_user_message.text)
            if(message.text==id_none):
                y_pred = Main.PredictAnswerNaive(text_preprocessed, model1)
            else:
                if(message.text == id_literal):
                    category="literal"
                elif (message.text == id_resource):
                    category="resource"
                else:
                    category="boolean"

                y_pred = Main.PredictAnswers(text_preprocessed , category, model2)

            unified_message_sender(message.chat.id, y_pred, parse_mode='HTML', keyboard_markup=keyboard)
            last_user_message=None
    except Exception as e:
        unified_message_sender(message.chat.id, str(e))


bot.infinity_polling()