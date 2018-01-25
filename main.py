import crop
from face_landmark import face
import datetime
from random import choice
from string import digits
import threading
import telebot
import time
from multiprocessing import Queue
from PIL import Image

token = '453625128:AAF_pVhEuaA4cNoXBTT7ogIhcMs1OsIAhwc'
bot = telebot.TeleBot(token)
queue = Queue()
style_image_path = "neural/style/simpson.jpg"


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


def detect_style_crop_save(image_path):
    face_image, face_points, face_size = face.detect_face_image_face_borders_points_face_size(image_path)
    face_image_path = None
    if len(face_image) and len(face_points) and face_size is not 0:
        style_image = face.image_as_torch_variable(style_image_path, face_size)
        content_image = face.numpy_ndarray_as_torch_variable(face_image)
        output_image = face.style_transfer(content_image, style_image)

        new_image = face.torch_variable_as_pil(output_image, face_size)
        new_image = crop.crop(new_image, face_points)

        folder_save = "faces/"
        time_now = time.ctime()
        for x in time_now:
            if (x == " ") or (x == ":"):
                x = "_"
            folder_save = folder_save + x

        face_image_path = (folder_save + "-out.png")
        new_image.save(face_image_path)

    return face_image_path


def photo(photo_path, chat_id):
    downloaded_file = bot.download_file(photo_path)
    with open(photo_path, 'wb') as new_file:
        new_file.write(downloaded_file)
    output_image_path = detect_style_crop_save(photo_path)
    if output_image_path is not None:
        output_image = open(output_image_path, 'rb')
        bot.send_photo(chat_id, output_image)
    else:
        bot.send_message(chat_id, "Не найдено лиц")


if __name__ == '__main__':
    bot.polling(none_stop=True)

