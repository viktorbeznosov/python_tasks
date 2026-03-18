from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, BotCommand
from aiogram import Router, F, Bot
from aiogram.fsm.context import FSMContext
from aiogram import Router
from aiogram.fsm.state import State, StatesGroup
from keyboards import services, create_inline_keyboard, create_reply_keyboard, create_correction_keyboard, create_correction_keyboard_final
from func_base import online_consultant, user_question, generate_client_report, refine_client_report, generate_presentation, generate_offer, user_objection_router, user_objection_close, save_to_table
from knowledge_base import create_db_index, load_db_index
import logging
import os
from datetime import datetime

router = Router()

class SaleState(StatesGroup):
    select_service = State()     # выбор услуги
    gather_details = State()     # уточняющие вопросы
    confirm_info = State()       # подтверждение данных
    correct_info = State()       # выбор поля, которое нужно исправить
    correction =  State()         # корректировка первоначальных данных
    consultant = State()         # работает онлайн-консультант
    contact = State()            # Запрос телефонного номера
    contact_name = State()       # Запрос имени
    contact_mail = State()       # Ввод почты
    custom_request = State()     # Ввод описания своего варианта услуг
    final_confirm = State()      # подтверждение отчёта
    additional_questions = State() # сбор ответов на уточняющие вопросы от gpt
    final_correction = State()     # корректировка ответов на уточняющие вопросы
    finalize = State()             # запись на замер
    address = State()             # Запрос адреса
    measurement_date = State()    # Запрос даты замера


# Если векторная база еще не создана:
if not os.path.exists('db_index.faiss'):
    create_db_index(os.getenv("DATA_DOC_URL"))
    logging.info(f"create_db_index() - OK")
db_index = load_db_index('db_index')

# Меню бота
@router.startup()  # Действия, выполняемые при запуске бота
async def on_startup(bot: Bot):
    # Устанавливаем команду /start с пояснением
    main_menu_commands = [
        BotCommand(command='start', description="Запуск бота"),
        BotCommand(command='clear_history', description="Очистка истории")
   ]   
    await bot.set_my_commands(main_menu_commands)


# Обработка команды /start
@router.message(Command('start'))
async def cmd_start(message: Message, state: FSMContext):
# Сброс текущего состояния
    await state.clear()
    # await clear_memory(message.from_user.id)
    greeting_message = (
        "Здравствуйте! 😊\n"
        "Вас приветствует ваш персональный нейро-ассистент по услугам продажи и установки пластиковых окон!\n\n"
        "Моя задача — помочь вам на каждом этапе: от выбора до работы замерщика или представителя компании.\n\n"
        "Воспользуйтесь кнопками внизу экрана для выбора требуемого действия.\n\n"
    )
    await message.answer(greeting_message, reply_markup=await create_reply_keyboard())

# Очистка всей истории чата
@router.message(Command('clear_history'))
async def clear_chat_history(message: Message, state: FSMContext):
    await state.clear()  # Полностью сбрасываем состояние
    await state.update_data(consultant_history=[])  # Очищаем историю общения с консультантом
    await message.answer("История диалога очищена. Вы можете начать новый запрос по кнопке внизу экрана.", reply_markup=await create_reply_keyboard())


# Обработчик нопки "Онлайн-консультант"
@router.message(F.text == "Онлайн-консультант")
async def start_consultant(message: Message, state: FSMContext):
    await state.clear() 
    await message.answer("Я - ваш помощник-консультант компании 'Северный профиль'. Пожалуйста, задайте свой вопрос о компании или ее услугах ")
    await state.set_state(SaleState.consultant) # Установить состояние


# Обработчик состояния "Онлайн-консультант"
@router.message(SaleState.consultant)
async def consultant(message: Message, state: FSMContext):
    if message.text.lower() == "стоп":
        await message.answer("Спасибо за ваше время. Если возникнут вопросы, обращайтесь!")
        await state.clear()
        return
      # Если пользователь выбирает другую функцию (например, "Связь с менеджером")
    if message.text == "Связь с менеджером компании":
        await state.clear()  # Очищаем текущее состояние
        await request_contact_data(message, state)  # Переходим в состояние менеджера
        return
    if message.text == "Выбрать услугу":
        await state.clear()  # Очищаем текущее состояние
        await get_service(message, state)  # Переходим в состояние выбора услуги
        return
       # Передача запроса консультанту
    data = await state.get_data()
    chat_history = data.get("consultant_history", [])  # Получаем историю диалога из состояния
    chat_history.append(f"Пользователь: {message.text}") # Добавляем новый вопрос пользователя в историю
    response = await online_consultant(message.text, chat_history, db_index) # Вызываем консультанта с историей
    chat_history.append(f"Бот: {response}") # Добавляем ответ бота в историю
    await state.update_data(consultant_history=chat_history) # Обновляем историю в состоянии
    await message.answer(response, reply_markup=await create_reply_keyboard()) # Отправка ответа пользователю


# Функция для запроса контактных данных менеджеру
@router.message(F.text == "Связь с менеджером компании")
async def request_contact_data(message: Message, state: FSMContext):
    await state.clear() 
    await state.set_state(SaleState.contact_name)
    await message.answer("Как к вам можно обращаться? Введите ваше имя:")

# Обработчик для получения имени клиента
@router.message(SaleState.contact_name)
async def get_contact_name(message: Message, state: FSMContext):
    if message.text.strip() == "стоп":
        await message.answer("Спасибо за ваше время. Если возникнут вопросы, обращайтесь!")
        await state.clear()
        return
    contact_name = message.text.strip()
    if not contact_name.isalpha():
        await message.answer("Имя должно содержать только буквы. Пожалуйста, введите ваше имя корректно.")
        return
    # Сохраняем имя клиента
    await state.update_data(contact_name_user=contact_name)
    await message.answer(f"Спасибо, {contact_name}! Теперь введите ваш номер телефона:")
    await state.set_state(SaleState.contact)

# Обработчик для получения контактных данных(номера телефона)
@router.message(SaleState.contact)
async def get_contact_info(message: Message, state: FSMContext):
    if message.text == "Выбрать услугу":
        await state.clear()  # Очищаем текущее состояние
        await get_service(message, state)  # Переходим в состояние выбора услуги
        return
    if message.text.strip() == "стоп":
        await message.answer("Спасибо за ваше время. Если возникнут вопросы, обращайтесь!")
        await state.clear()
        return
    contact_phone = message.text.strip()
    # Валидация номера телефона 
    if not contact_phone.isdigit():
        await message.answer("Пожалуйста, введите корректный номер телефона (только цифры) или 'стоп' для выхода.")
        return
    if len(contact_phone) < 10 or len(contact_phone) > 15:
        await message.answer("Номер телефона должен содержать от 10 до 15 цифр.")
        return
    # Сохраняем номер телефона
    await state.update_data(contact_phone_user=contact_phone)
    await message.answer("Теперь, пожалуйста, введите ваш email:")
    # Устанавливаем состояние для получения email
    await state.set_state(SaleState.contact_mail)  

# Обработчик для получения контактных данных(электронной почты)
@router.message(SaleState.contact_mail)
async def get_email(message: Message, state: FSMContext):
        if message.text == "Выбрать услугу":
          await state.clear()  # Очищаем текущее состояние
          await get_service(message, state)  # Переходим в состояние выбора услуги
          return
        if message.text.strip() == "стоп":
          await message.answer("Спасибо за ваше время. Если возникнут вопросы, обращайтесь!")
          await state.clear()
          return
        # Сохраняем email
        contact_email = message.text.strip()
        await state.update_data(contact_email_user=contact_email)
        contact_email = message.text.strip()
    # Валидация email
        if "@" not in contact_email or "." not in contact_email:
           await message.answer("Пожалуйста, введите корректный email (например, user@example.com).")
           return
         # Извлекаем данные непосредственно из состояния
        data = await state.get_data()  # Получаем данные состояния
        selected_service = data.get("selected_service")
         # Если выбрана услуга — запрашиваем адрес
        if selected_service and selected_service != "Свой индивидуальный вариант":
          await message.answer("Спасибо! Теперь, пожалуйста, укажите ваш адрес для замера:")
          await state.set_state(SaleState.address)
          return
        contact_phone = data.get('contact_phone_user')  # Извлекаем номер телефона
        contact_name = data.get('contact_name_user')  # Извлекаем
        # Получаем сохраненные данные
        await message.answer(
        f"Ваши контактные данные:\n"
        f"Имя: {contact_name}\n"
        f"Телефон: {contact_phone}\n"
        f"Email: {contact_email}\n"
        "В ближайшее время с вами свяжутся. Можете выберать дальнейшее действие по кнопке ниже",  reply_markup=await create_reply_keyboard())
        # Сохраняем данные в таблицу
        await save_to_table(state)
        # Завершение состояния
        await state.clear()


# Обработчик команды выбора услуг
@router.message(F.text == "Выбрать услугу")
async def get_service(message: Message, state: FSMContext):
    await state.clear()  # Очищаем текущее состояние перед переходом
    await message.answer("""Я задам Вам несколько уточняющих вопросов, 
    чтобы более подробно понять ваш запрос и упорядочить информацию.""")
    await message.answer("Для начала выберите номер услуги из списка:", reply_markup = create_inline_keyboard(services))
    await state.set_state(SaleState.select_service)
    

# Обработка выбора услуги
@router.callback_query(F.data.startswith("service_"))
async def handle_service_choice(callback_query: CallbackQuery, state: FSMContext):
    service_index = int(callback_query.data.split("_")[1]) - 1
    service_name = services[service_index]  # Получаем имя услуги
    await state.update_data(selected_service=service_name)
    await callback_query.message.answer(
        f"Отлично, услуга '{service_name}' выбрана. Теперь уточним подробности!")
    data = await state.get_data()
    if data.get('selected_service') == "Свой индивидуальный вариант":
           # Если выбрана "Свой индивидуальный вариант" -> спрашиваем описание
        await state.set_state(SaleState.custom_request)  # Устанавливаем состояние для ввода описания
        await callback_query.message.answer("Опишите ваш запрос подробнее:")
        return
    # Если выбрана другая услуга -> задаём уточняющие вопросы
    await state.set_state(SaleState.gather_details)
    await ask_next_question(callback_query.message, state)


# Обработчик для ввода пользовательского варианта услуги
@router.message(SaleState.custom_request)
async def get_custom_request(message: Message, state: FSMContext):
    user_request = message.text.strip()
    await state.update_data(custom_request=user_request)
    # Теперь запрашиваем контактные данные (переходим в состояние связи с менеджером)
    await state.set_state(SaleState.contact_name)
    await message.answer(
        "Спасибо! Теперь, пожалуйста, ваши контактные данные, чтобы менеджер компании мог связаться с вами. \n Ваше имя: ")


# Список вопросов для уточнения деталей
questions = [
    "Для какого типа помещения вам нужны услуги: квартира, дом, офис, балкон или что-то другое?",
    "В каком городе или районе находится объект, где нужно выполнить работы?",
    "Вы оформляете заказ как частное лицо или от имени компании?", 
    "Есть ли у вас особые пожелания к окнам/конструкциям/работам?"
]
# Функция для последовательного задавания вопросов
async def ask_next_question(message: Message, state: FSMContext):
    data = await state.get_data()
    current = data.get("current_question", 0)
    if current < len(questions):
        await message.answer(questions[current])
    else:
        # Все вопросы заданы, переходим к подтверждению
        await summarize_and_confirm(message, state)


# Обработчик уточняющих вопросов
@router.message(SaleState.gather_details)
async def gather_details(message: Message, state: FSMContext):
    data = await state.get_data()
    current = data.get("current_question", 0)
    answers = data.get("answers", {})
    # Сохраняем ответ
    answers[str(current)] = message.text.strip()
    await state.update_data(answers=answers, current_question=current + 1)
    # Переход к следующему вопросу
    await ask_next_question(message, state)


# Подтверждение собранных данных
async def summarize_and_confirm(message: Message, state: FSMContext):
    data = await state.get_data()
    answers = data.get("answers", {})
    field_names = {
        "0": "Тип объекта",
        "1": "Местоположение",
        "2": "Вы (частное лицо/компания)",
        "3": "Особые пожелания"
    }
    summary_lines = [f"<b>Услуга:</b> {data.get('selected_service', 'не указано')}"]
    for key, field in field_names.items():
        answer = answers.get(key, "не указано")
        summary_lines.append(f"<b>{field}:</b> {answer}")
    summary = "\n".join(summary_lines)
    await state.update_data(summary=summary)
    await message.answer("Вот собранная информация:\n" + summary, parse_mode="HTML")
    await message.answer("Если всё верно — напишите 'да' или 'ок'. "
                         "Если требуется исправить, напишите 'нет' ")
    await state.set_state(SaleState.confirm_info)


# Подтверждение данных пользователем
@router.message(SaleState.confirm_info)
async def confirm_handler(message: Message, state: FSMContext):
    text = message.text.strip().lower()
    if text in ("да", "ок", "ok"):
        await message.answer("Спасибо, ваша информация принята. Переходим дальше.")
        data = await state.get_data()
        selected_scenario = data.get("selected_service", "Не указано")  # Получаем выбранную услугу
        gathered_info = data.get("summary", {})  # Берём подтверждённые ответы
        await message.answer("Теперь давайте уточним детали вашего запроса. Генерирую вопросы: ...")
        # Генерируем уточняющие вопросы на основе сценария и подтверждённых данных
        questions = await user_question(selected_scenario, gathered_info)
        # Сохраняем вопросы в `state`
        await state.update_data(additional_questions=questions, current_question=0)
        # Переходим в состояние уточняющих вопросов
        await state.set_state(SaleState.additional_questions)
        # Начинаем задавать вопросы
        await ask_next_scenario_question(message, state)
    elif text == "нет":
        await message.answer("Выберите, какое поле хотите исправить:", reply_markup=create_correction_keyboard())
        await state.set_state(SaleState.correct_info)
    else:
        await message.answer("Пожалуйста, введите 'да' для подтверждения или 'нет' для корректировки.")


# Обработка выбора пункта для изменения
@router.callback_query(F.data.startswith("edit_"))
async def edit_field_choice(callback_query: CallbackQuery, state: FSMContext):
    field_key = callback_query.data.split("_")[1]  # Получаем номер поля
    field_names = {
        "0": "Тип объекта",
        "1": "Местоположение",
        "2": "Вы (частное лицо/компания)",
        "3": "Особые пожелания"
    }
    field_name = field_names[field_key]  # Получаем название поля
    await state.update_data(editing_field=field_key)  # Сохраняем, какое поле редактируем
    await callback_query.message.answer(f"Введите новое значение для '{field_name}':")
    await state.set_state(SaleState.correction)


# Обработка ввода нового значения и снова показываем клавиатуру
@router.message(SaleState.correction)
async def update_corrected_info(message: Message, state: FSMContext):
    new_value = message.text.strip()
    data = await state.get_data()
    field_key = data.get("editing_field")  # Получаем, какое поле редактируется
    if field_key is None:
        await message.answer("Ошибка! Попробуйте снова выбрать поле для корректировки.")
        return
    # Обновляем данные
    answers = data.get("answers", {})
    answers[field_key] = new_value
    await state.update_data(answers=answers, editing_field=None)  # Очищаем временные данные
    # Показываем обновлённую информацию
    await message.answer(f"Значение обновлено! Выберите следующее поле для исправления или нажмите '✅ Готово' , если все верно.",
                         reply_markup=create_correction_keyboard())
    await state.set_state(SaleState.correct_info)


# Обрабатываем завершение корректировки
@router.callback_query(F.data == "done_editing")
async def finish_editing(callback_query: CallbackQuery, state: FSMContext):
    await callback_query.message.answer("Корректировка завершена. Вот обновлённая информация:")
    await summarize_and_confirm(callback_query.message, state)  # Повторное подтверждение данных


# Функция для последовательного задавания вопросов от gpt
async def ask_next_scenario_question(message: Message, state: FSMContext):
    data = await state.get_data()
    questions = data.get("additional_questions", [])
    current = data.get("current_question", 0)
    if current < len(questions):
        await message.answer(questions[current])  # Отправляем текущий вопрос
    else:
        # Если вопросы закончились — завершаем процесс
        await message.answer("Спасибо! Все уточняющие вопросы заданы! Давайте сверим")
        await summarize_final_info(message, state)
        #await state.clear()  # Очищаем состояние


# Обработчик ответов пользователя на вопросы от gpt      
@router.message(SaleState.additional_questions)
async def handle_scenario_answers(message: Message, state: FSMContext):
    data = await state.get_data()
    current = data.get("current_question", 0)
    additional_answers = data.get("additional_answers", {})
    additional_answers[str(current)] = message.text.strip()  # Сохраняем ответ
    # Обновляем `state`
    await state.update_data(additional_answers=additional_answers, current_question=current + 1)
    # Переход к следующему вопросу
    await ask_next_scenario_question(message, state)


# Генерация отчета потребностей с подтверждением
async def summarize_final_info(message: Message, state: FSMContext):
    """Генерирует отчёт и сохраняет его в состояние."""
    data = await state.get_data()
    additional_answers = data.get("additional_answers", {})
    # Генерируем отчёт
    report = await generate_client_report(additional_answers, gathered_info=None)
    # Сохраняем отчёт в состояние
    await state.update_data(final_report=report)
    # Формируем сообщение
    summary_lines = [
        f"<b>Услуга:</b> {data.get('selected_service', 'не указано')}",
        "<b>Дополнительные уточнения:</b>",
        report
    ]
    summary = "\n".join(summary_lines)
    # Отправляем отчёт пользователю
    await message.answer(f"Вот полная информация о вашем запросе:\n{summary}", parse_mode="HTML")
    await message.answer("Подтвердите информацию или внесите исправления", reply_markup=create_correction_keyboard_final())
    # Устанавливаем состояние подтверждения финальных данных
    await state.set_state(SaleState.final_confirm)


# Обработчик состояния "Финальное подтверждение"
@router.callback_query(F.data == "final_confirm")
async def handle_final_confirmation(callback_query: CallbackQuery, state: FSMContext):
    """Подтверждает финальный отчёт и переходит к презентации."""
    data = await state.get_data()
    final_report = data.get("final_report", {})
    gathered_info = data.get("summary", {}) 
    # Формируем саммари отчета
    summary_report = f"{gathered_info}\n{final_report}\n"
    # Отправляем итоговый отчёт
    await callback_query.message.answer(f"<b>Итоговый отчёт о потребностях:</b>\n{summary_report}", parse_mode="HTML")
    # Завершаем процесс
    await callback_query.message.answer("Спасибо! Теперь я готов вам рассказать о своем предложении для вас!")
    presentation = await generate_presentation(db_index, summary_report)
    offer = await generate_offer(db_index, summary_report)
    await state.update_data(offer=offer)
    await callback_query.message.answer(f"{presentation} \n\n {offer} ")
    await state.set_state(SaleState.finalize)
    await callback_query.message.answer("Введите да/хорошо/подходит, если все устраивает, или расскажите о сомнениях")


# Обработчик состояния "Финальная корректировка"
@router.callback_query(F.data == "final_edit")
async def ask_for_correction(callback_query: CallbackQuery, state: FSMContext):
    """Запрашивает у пользователя текст с исправлениями."""
    await callback_query.message.answer(
        "Введите, что нужно исправить в отчёте.\n\n"
        "Например: \"Добавьте вывоз старых окон.\""
    )
    await state.set_state(SaleState.final_correction)


# Корректировка ответов на уточняющие вопросы
@router.message(SaleState.final_correction)
async def handle_final_correction(message: Message, state: FSMContext):
    """Обрабатывает исправления и обновляет отчёт."""
    correction = message.text.strip()
    # Достаём текущий отчёт
    data = await state.get_data()
    previous_report = data.get("final_report", {})
    # Генерируем новый отчёт с учётом исправлений
    updated_report = await refine_client_report(previous_report, correction)
    # Сохраняем исправленный отчёт
    await state.update_data(final_report=updated_report)
    # Отправляем исправленный отчёт
    await message.answer(
        f"Обновлённый отчёт с учётом ваших исправлений:\n{updated_report}",
        parse_mode="HTML"
    )
    # Повторно предлагаем подтвердить или внести новые исправления
    await message.answer(
        "Подтвердите новый отчёт или внесите ещё исправления.",
        reply_markup=create_correction_keyboard_final()
    )


# Подтверждение данных пользователем
@router.message(SaleState.finalize)
async def record_measurement_to_excel(message: Message, state: FSMContext):
    data = await state.get_data()
    chat_history = data.get("consultant_history", [])  # Если истории нет, создаем пустой список
    offer = data.get("offer", [])
    text = message.text.strip().lower()
    if text in ("да", "ок", "ok", "готова", "запишите", "хорошо", ):
        await message.answer("""Отлично! Давайте оформим заказ. Понадобятся ваши контактные данные для обратной 
            связи менеджера компании. Пожалуйста, введите имя заказчика: """)
        await state.set_state(SaleState.contact_name)  # Переход к запросу контактов
        return
    # Если клиент выражает сомнения или задает вопрос
    else:
        # Анализируем ответ на наличие возражений
        objection = await user_objection_router(text)
        if objection != "−":  # Если есть возражения
            await message.answer("Спасибо за ваш отзыв! Давайте разберем ваши сомнения.")
            chat_history.append(f"Пользователь: {text}")
            objection_response = await user_objection_close(db_index, offer, objection)  # Отрабатываем возражение
            await message.answer(objection_response)
            chat_history.append(f"Бот: {objection_response}")
            await state.set_state(SaleState.finalize)  # Остаемся в состоянии финализации
        else:  # Если это вопрос
            await message.answer("Спасибо за ваш вопрос! Сейчас я передам ваш вопрос нашему онлайн-консультанту.")
            # Добавляем новый вопрос пользователя в историю
            chat_history.append(f"Бот: {offer}")  # Добавляем оффер в историю
            chat_history.append(f"Пользователь: {text}")
            response = await online_consultant(objection, chat_history, db_index)
            # Добавляем ответ бота в историю
            chat_history.append(f"Бот: {response}")
            # Обновляем историю в состоянии
            await state.update_data(consultant_history=chat_history)
            await message.answer(response)
            await message.answer(f'Итак, вернемся к нашему предложению: {offer} \n\n Введите да/хорошо/подходит, если все устраивает, или расскажите о сомнениях')
            await state.set_state(SaleState.finalize)  # Остаемся в состоянии финализации  



# Обработчик для получения контактных данных(адреса)
@router.message(SaleState.address)
async def get_address(message: Message, state: FSMContext):
    if message.text == "Выбрать услугу":
        await state.clear()  # Очищаем текущее состояние
        await get_service(message, state)  # Переходим в состояние выбора услуги
        return
    if message.text.strip() == "стоп":
        await message.answer("Спасибо за ваше время. Если возникнут вопросы, обращайтесь!")
        await state.clear()
        return
    address = message.text.strip()
    # Сохраняем адрес
    await state.update_data(address=address)
    # Запрашиваем дату замера
    await message.answer("Спасибо! Теперь, пожалуйста, укажите желаемую дату замера в формате 'DD-MM-YYYY':")
    await state.set_state(SaleState.measurement_date)  # Переходим в состояние запроса даты


# Обработчик для получения даты замера
@router.message(SaleState.measurement_date)
async def get_measurement_date(message: Message, state: FSMContext):
    if message.text == "Выбрать услугу":
        await state.clear()  # Очищаем текущее состояние
        await get_service(message, state)  # Переходим в состояние выбора услуги
        return
    if message.text.strip() == "стоп":
        await message.answer("Спасибо за ваше время. Если возникнут вопросы, обращайтесь!")
        await state.clear()
        return
    measurement_date = message.text.strip()
    # Проверим формат даты (например, 'YYYY-MM-DD')
    try:
       datetime.strptime(measurement_date, '%d-%m-%Y')
    except ValueError:
       await message.answer("Неверный формат даты. Пожалуйста, введите дату в формате 'DD-MM-YYYY'.")
       return
    # Сохраняем дату замера
    await state.update_data(measurement_date=measurement_date)
    # Все данные собраны, запрашиваем контактные данные
    data = await state.get_data()
    # Финальное подтверждение данных перед записью в таблицу
    await message.answer(
        f"Ваш заказ оформлен!\n\n"
        f"Услуга: {data.get('selected_service')} "
        f"Имя: {data.get('contact_name_user')}\n"
        f"Телефон: {data.get('contact_phone_user')}\n"
        f"Email: {data.get('contact_email_user')}\n"
        f"Адрес: {data.get('address')}\n"
        f"Дата замера: {data.get('measurement_date')}\n\n"
        "В ближайшее время с вами свяжутся. Спасибо за заявку!",
        reply_markup=await create_reply_keyboard()
    )
    # Сохраняем данные в таблицу
    await save_to_table(state)
    # Очистка состояния
    await state.clear()
            


