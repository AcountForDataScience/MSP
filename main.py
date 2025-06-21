import os
import telebot
import numpy as np
import pandas as pd
import random
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from scipy import stats
from telebot import types
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io
import re

import heapq

import csv

from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as index

Access_dic = {
    'aramasht@gmail.com': '6719',
    'test@test.com': 'test'
}
Access_dic_0 = str(list(Access_dic.keys())[0])
Access_dic_1 = str(list(Access_dic.keys())[1])

YesNo_dict = {
    'No': 0,
    'Yes': 1
}
YesNo_dict_0 = str(list(YesNo_dict.keys())[0])
YesNo_dict_1 = str(list(YesNo_dict.keys())[1])

City = None
Troop_Movements = None
Air_Attacks_Last_72h = None
Artillery_Intensity_Index = None
Fuel_Shortage = None
Evacuation_Started = None
Defense_Reinforcements = None
Intel_Warning_Level = None
Cyber_Disruption = None
Population_size = None

def Check_Password(password):
  for value in Access_dic.values():
    if value == password:
      return True
result = Check_Password('6719')


def Predict_City_Capture(new_city):
# 🔹 1. Завантаження даних
  df = pd.read_csv("real_city_risk_training.csv")

  # 🔹 2. Фічі та мішень
  features = [
      "Troop_Movements",
      "Air_Attacks_Last_72h",
      "Artillery_Intensity_Index",
      "Fuel_Shortage",
      "Evacuation_Started",
      "Defense_Reinforcements",
      "Intel_Warning_Level",
      "Cyber_Disruptions",
      "Population_size"
  ]
  X = df[features]
  y = df["City_Captured"]

  # 🔹 3. Train/test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # 🔹 4. Навчання моделі
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  # 🔹 5. Оцінка
  y_pred = model.predict(X_test)

  prob = model.predict_proba(new_city)[0][1] * 100
  pred = model.predict(new_city)[0]

  if pred < 1:
    pred = 'not expected'
  else:
    pred = 'is expected'
  return prob, pred

bot = telebot.TeleBot('7424733861:AAETBLpo2fNyuafns02g-EAQ5CbemUYdLyg')
#t.me/MSP_CC_bot
#Military Strategic Planning City Capture
#7424733861:AAETBLpo2fNyuafns02g-EAQ5CbemUYdLyg
@bot.message_handler(commands=['help', 'start'])

def send_welcome(message):
    msg = bot.send_message(message.chat.id, "\n\nHello, I'm the military strategic planning AI bot for forecasting the risk of city capture!")
    chat_id = message.chat.id
    msg = bot.reply_to(message, 'Please enter your password')
    bot.register_next_step_handler(msg, process_Password_step)

def process_Password_step(message):
  try:
    chat_id = message.chat.id
    Password_message = message.text
    result = Check_Password(Password_message)
    if result == True:
      markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
      markup.add('Next')
      msg = bot.reply_to(message, 'You are welcome. Please press Next to continue', reply_markup=markup)
      bot.register_next_step_handler(msg, process_City_Name_Request_step)
    else:
      msg = bot.reply_to(message, '❌ Incorrect password. Please try again.')
      bot.register_next_step_handler(msg, process_Password_step)
  except Exception as e:
    bot.reply_to(message, 'oooops process_Password_step')

def process_City_Name_Request_step(message):
    try:
      chat_id = message.chat.id
      City_Name_Request_message = message.text
      if (City_Name_Request_message == 'Next'):
        msg = bot.reply_to(message, 'Please enter city name')
        bot.register_next_step_handler(msg, process_City_Name_step)
      else:
        raise Exception("process_City_Name_Request_step")
    except Exception as e:
        bot.reply_to(message, 'oooops process_City_Name_Request_step')

def process_City_Name_step(message):
    try:
      chat_id = message.chat.id
      City_Name_message = message.text
      global City
      City = City_Name_message
      msg = bot.reply_to(message, 'Please enter troop_movements value')
      bot.register_next_step_handler(msg, process_Troop_Movements_step)
    except Exception as e:
        bot.reply_to(message, 'oooops process_City_Name_step')

def process_Troop_Movements_step(message):
    try:
        chat_id = message.chat.id
        Troop_Movements_message = message.text
        if not Troop_Movements_message.isdigit():
          msg = bot.reply_to(message, 'Troop movements must be a number. Please enter a Troop movements.')
          bot.register_next_step_handler(msg, process_Troop_Movements_step)
        else:
          global Troop_Movements
          Troop_Movements = int(Troop_Movements_message)
          msg = bot.reply_to(message, 'Please enter Air_Attacks_Last_72h')
          bot.register_next_step_handler(msg, process_Air_Attacks_Last_72h_step)
    except Exception as e:
        bot.reply_to(message, 'oooops process_Troop_Movements_step')


def process_Air_Attacks_Last_72h_step(message):
    try:
        chat_id = message.chat.id
        Air_Attacks_Last_72h_message = message.text
        if not Air_Attacks_Last_72h_message.isdigit():
          msg = bot.reply_to(message, 'Air Attacks Last 72h must be a number. Please enter a Air Attacks Last 72h .')
          bot.register_next_step_handler(msg, process_Air_Attacks_Last_72h_step)
        else:
          global Air_Attacks_Last_72h
          Air_Attacks_Last_72h = int(Air_Attacks_Last_72h_message)
          msg = bot.reply_to(message, 'Please enter Artillery Intensity Index')
          bot.register_next_step_handler(msg, process_Artillery_Intensity_Index_step)
    except Exception as e:
        bot.reply_to(message, 'oooops process_Air_Attacks_Last_72h_step')

def process_Artillery_Intensity_Index_step(message):
    try:
        chat_id = message.chat.id
        Artillery_Intensity_Index_message = message.text
        Artillery_Intensity_Index_message = float(Artillery_Intensity_Index_message)
        print(Artillery_Intensity_Index_message)

        global Artillery_Intensity_Index
        Artillery_Intensity_Index = Artillery_Intensity_Index_message
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
        markup.add(YesNo_dict_0, YesNo_dict_1)
        msg = bot.reply_to(message, 'Please enter Fuel Shortage', reply_markup=markup)
        bot.register_next_step_handler(msg, process_Fuel_Shortage_step)
    except Exception as e:
        bot.reply_to(message, 'oooops process_Artillery_Intensity_Index_step')

def process_Fuel_Shortage_step(message):
    try:
        chat_id = message.chat.id
        Fuel_Shortage_message = message.text
        if (Fuel_Shortage_message == YesNo_dict_0) or (Fuel_Shortage_message == YesNo_dict_1):
          global Fuel_Shortage
          Fuel_Shortage = YesNo_dict[Fuel_Shortage_message]
          markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
          markup.add(YesNo_dict_0, YesNo_dict_1)
          msg = bot.reply_to(message, 'Please enter Evacuation_Started', reply_markup=markup)
          bot.register_next_step_handler(msg, process_Evacuation_Started_step)
    except Exception as e:
        bot.reply_to(message, 'oooops process_Fuel_Shortage_step')

def process_Evacuation_Started_step(message):
    try:
        chat_id = message.chat.id
        Evacuation_Started_message = message.text
        if (Evacuation_Started_message == YesNo_dict_0) or (Evacuation_Started_message == YesNo_dict_1):
          global Evacuation_Started
          Evacuation_Started = YesNo_dict[Evacuation_Started_message]
          markup_remove = types.ReplyKeyboardRemove(selective=False)
          msg = bot.reply_to(message, 'Please enter Defense Reinforcements', reply_markup=markup_remove)
          bot.register_next_step_handler(msg, process_Defense_Reinforcements_step)
    except Exception as e:
        bot.reply_to(message, 'oooops process_Evacuation_Started_step')

def process_Defense_Reinforcements_step(message):
    try:
        chat_id = message.chat.id
        Defense_Reinforcements_message = message.text
        if not Defense_Reinforcements_message.isdigit():
          msg = bot.reply_to(message, 'Defense Reinforcements must be a number. Please enter a Defense Reinforcements.')
          bot.register_next_step_handler(msg, process_Defense_Reinforcements_step)
        else:
          global Defense_Reinforcements
          Defense_Reinforcements = int(Defense_Reinforcements_message)
          msg = bot.reply_to(message, 'Please enter Intel Warning Level')
          bot.register_next_step_handler(msg, process_Intel_Warning_Level_step)
    except Exception as e:
        bot.reply_to(message, 'oooops process_Defense_Reinforcements_step')

def process_Intel_Warning_Level_step(message):
    try:
        chat_id = message.chat.id
        Intel_Warning_Level_message = message.text
        if not Intel_Warning_Level_message.isdigit():
          msg = bot.reply_to(message, 'Intel Warning Level message must be a number. Please enter a Intel Warning Level.')
          bot.register_next_step_handler(msg, process_Intel_Warning_Level_step)
        else:
          global Intel_Warning_Level
          Intel_Warning_Level = int(Intel_Warning_Level_message)
          markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
          markup.add(YesNo_dict_0, YesNo_dict_1)
          msg = bot.reply_to(message, 'Cyber Disruptions', reply_markup=markup)
          bot.register_next_step_handler(msg, process_Population_size_step)
    except Exception as e:
        bot.reply_to(message, 'oooops process_Intel_Warning_Level_step')

def process_Population_size_step(message):
    try:
        chat_id = message.chat.id
        Cyber_Disruptions_message = message.text
        if (Cyber_Disruptions_message == YesNo_dict_0) or (Cyber_Disruptions_message == YesNo_dict_1):
          global Cyber_Disruption
          Cyber_Disruption = YesNo_dict[Cyber_Disruptions_message]
          markup_remove = types.ReplyKeyboardRemove(selective=False)
          msg = bot.reply_to(message, 'Please enter a Population_size.', reply_markup=markup_remove)
          bot.register_next_step_handler(msg, predict_City_Capture_step)
    except Exception as e:
        bot.reply_to(message, 'oooops process_Population_size_step')

def predict_City_Capture_step(message):
  try:
    chat_id = message.chat.id
    Population_size_message = message.text
    Population_size_message = float(Population_size_message)

    global Population_size
    Population_size = int(Population_size_message)
    new_city = pd.DataFrame([{
    "Troop_Movements": Troop_Movements,
    "Air_Attacks_Last_72h": Air_Attacks_Last_72h,
    "Artillery_Intensity_Index": Artillery_Intensity_Index,
    "Fuel_Shortage": Fuel_Shortage,
    "Evacuation_Started": Evacuation_Started,
    "Defense_Reinforcements": Defense_Reinforcements,
    "Intel_Warning_Level": Intel_Warning_Level,
    "Cyber_Disruptions": Cyber_Disruption,
    "Population_size": Population_size
    }])
    prob, pred = Predict_City_Capture(new_city)

    bot.send_message(chat_id,

    '\n\n - Risk of ' + str(City) + ' capture: ' + str(pred)+
    '\n- Probability of '+ str(City) + ' captured in percent: ' + str(prob) + ' %'  +
    '\n' +                  '(RandomForestClassifier)' +
    '\n\nFor forecasting, the model uses training data for the cities of Liman, Izyum, Popasna, Mariupol, and Severodonetsk.' +
    '\n\n Перейти на оперативне планування @COA_NATO_OPP_bot'
    )

    markup = types.ReplyKeyboardMarkup(one_time_keyboard=False)
    markup.add('Спробувати знову')
    msg = bot.reply_to(message, 'Спробувати знову', reply_markup=markup)
    bot.register_next_step_handler(msg, send_welcome)

  except Exception as e:
    bot.reply_to(message, 'oooops predict_City_Capture_step')



bot.infinity_polling()
