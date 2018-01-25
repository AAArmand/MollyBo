import threading
import telebot
import time
from multiprocessing import Queue


token = '453625128:AAF_pVhEuaA4cNoXBTT7ogIhcMs1OsIAhwc'
bot = telebot.TeleBot(token)
queue = Queue()


class App():
    __worker = None

    @staticmethod
    def worker():
        if App.__worker is None:
            App.__worker = Worker(queue)
        return App.__worker


class Worker(threading.Thread):
    """
    Класс потока который будет брать задачи из очереди и выполнять их до успешного
    окончания или до исчерпания лимита попыток
    """

    def __init__(self, queue):
        # Обязательно инициализируем супер класс (класс родитель)
        super(Worker, self).__init__()
        # Устанавливаем поток в роли демона, это необходимо что бы по окончании выполнения
        # метода run() поток корректно завершил работу,а не остался висеть в ожидании
        self.setDaemon(True)
        # экземпляр класса содержит в себе очередь что бы при выполнении потока иметь к ней доступ
        self.queue = queue

    def run(self):
        """
        Основной код выполнения потока должен находиться здесь
        """
        while True:
            if not self.queue.empty():
                # запрашиваем из очереди объект
                target = self.queue.get(block=False)
                file_info = bot.get_file(target.photo[len(target.photo) - 1].file_id)
                src = file_info.file_path
                photo(src, target.chat.id)


@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):

    try:

        queue.put(message)
        worker = App.worker()
        if not worker.is_alive():
            worker.start()

    except Exception as e:
        bot.reply_to(message, e)


def photo(photo_path, chat_id):
    downloaded_file = bot.download_file(photo_path)
    # pass downloaded_file
    bot.send_photo(chat_id, downloaded_file)

