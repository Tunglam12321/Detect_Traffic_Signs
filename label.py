import pyttsx3
import pygame
from gtts import gTTS
import time
import io

labels_module=['Biển đường cấm','Biển cấm ngược chiều','Biển cấm xe khách','Biển báo kết thúc đường đôi',
        'Biển báo có cáp điện ở trên','Biển báo chiều cao tĩnh không thực tế','Biển báo đường hầm',
        'Biển báo đường sắt giao vuông góc','Biển báo đoạn đường hay xảy ra tai nạn','Biển báo đi chậm',
        'Biển báo chú ý chướng ngại vật','Biển báo chú ý đỗ xe','Biển báo dừng lại',
        'Biển báo hướng phải đi theo','Biển báo hướng phải vòng chướng ngại vật',
        'Biển báo nơi giao nhau chạy theo vòng xuyến','Đường cho xe thô sơ','Đường cho người đi bộ',
        'Biển báo tốc độ tối thiểu','Đường cho ô tô','Đường cho ô tô xe máy','Đường cho xe máy xe đạp',
        'Hướng đi trên mỗi làn đường','Làn đường cho ô tô con','Cấm xe đạp','Làn cho xe buýt','Làn cho ô tô',
        'Biển gộp làn đường theo phương tiện','Khu đông dân cư','Hết khu đông dân cư','Cấm xe gắn máy',
        'Kết thúc đường hầm','Đường ưu tiên','Cấm xe ba bánh','Đường một chiều','Nơi đỗ xe','Chỗ quay xe',
        'Chỉ hướng đường','Cấm người đi bộ','Lối đi ở những chỗ cấm rẽ','Vị trí người đi bộ sang đường',
        'Cầu vượt qua đường','Hầm chui qua đường','Cấm xe người kéo','Bệnh viện','Cửa hàng xăng dầu',
        'Cấm ô tô','Cấm xe súc vật kéo','Bến xe buýt','Hạn chế trải trọng xe','Hạn chế tải trọng trục xe',
        'Phía trước có công trường','Chợ','Hạn chế chiều cao','Mô tả tình trạng đường',
        'Nơi đỗ xe cho người khuyết tật','Đường cao tốc','Hết biển gộp làn đường theo phương tiện',
        'Hạn chế chiều dài xe','Cự ly tối thiểu giữa hai xe','Cấm rẽ trái','Cấm rẽ phải','Cấm ô tô rẽ trái',
        'Cấm quay đầu xe','Cấm ô tô quay đầu','Cấm rẽ trái quay đầu','Cấm rẽ phải quay đầu',
        'Cấm ô tô rẽ trái quay đầu','Cấm vượt','Tốc độ tối đa','Cấm ô tô rẽ phải',
        'Biển tốc độ tối đa theo phương tiện','Cấm còi','Cấm dừng đỗ xe','Cấm đỗ xe','Hết tốc độ tối đa',
        'Cấm xe máy','Hết tất cả cấm','Cấm đi thẳng','Cấm đi thẳng rẽ trái','Ngoặt nguy hiểm',
        'Ngoặt dễ lật xe','Nhiều ngoặt liên tiếp','Đường thu hẹp','Cấm ô tô xe máy','Đường hai chiều',
        'Đường giao nhau','Giao nhau với đường không ưu tiên','Giao nhau với đường ưu tiên',
        'Giao nhau có đèn tín hiệu','Giao nhau với đường sắt có rào','Giao nhau với đường sắt không rào',
        'Cầu hẹp','Cấm xe tải','Kè vực phía trước','Kè vực bên trái','Kè vực bên phải','Dốc xuống nguy hiểm',
        'Dốc lên nguy hiểm','Đường lồi lõm','Đường có gồ giảm tốc','Đường trơn','Vách núi nguy hiểm',
        'Đường người đi bộ cắt ngang','Trẻ em','Công trường','Cấm xe khách xe tải','Đá lở','Dải máy bay',
        'Thú rừng qua đường','Nguy hiểm khác','Đường đôi']

# Hàm phát âm thanh cảnh báo cho đối tượng
def speak_text(number):
    # Khởi tạo pyttsx3
    pygame.mixer.init()

    tts=gTTS(text=labels_module[number],lang='vi')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    # Phát giọng nói từ bộ nhớ tạm
    pygame.mixer.music.load(mp3_fp, 'mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    