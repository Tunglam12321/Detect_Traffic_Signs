import streamlit as st
import base64
import cv2
import os 
import torch
from PIL import Image,ImageDraw,ImageFont,ImageOps
import numpy as np
import pathlib  
import zipfile
import io 
from label import speak_text
from color import get_label_color,random_color,hex_to_bgr,get_label_color1
from object_tracker import ObjectTracker
from pathlib import Path
temp = pathlib.PosixPath   
pathlib.PosixPath = pathlib.WindowsPath

# Tải mô hình YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

#cho phép chống xoay ngang ảnh
def remove_exif(image):
    # Remove EXIF data from the image
    return ImageOps.exif_transpose(image)

#nén thư mục để download
def zip_folder_in_memory(folder_path):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, folder_path))
    zip_buffer.seek(0)
    return zip_buffer

       
def main():
    # Tạo thư mục data để lưu dữ liệu
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/images"):
        os.makedirs("data/images")
    if not os.path.exists("data/videos"):
        os.makedirs("data/videos")
    
    # Thiết lập layout của ứng dụng Streamlit
    distance=70
    st.set_page_config(layout="wide")

    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.markdown("<div class='sidebar-title'>HỆ THỐNG NHẬN DẠNG BIỂN BÁO GIAO THÔNG ĐƯỜNG BỘ</div>", unsafe_allow_html=True)

    # Sidebar nội dung
    st.sidebar.markdown("<div class='sidebar-title'>BẢNG ĐIỀU KHIỂN</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-text'><i>Nhận dạng biển báo giao thông</i></div>", unsafe_allow_html=True)
    
    st.sidebar.markdown("<div class='sidebar-title'>TẢI DỮ LIỆU</div>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Chọn ảnh/video", type=["jpg", "png", "jpeg","mp4"], key="right_file_uploader")

    # Tạo nút để bật/tắt camera
    camera_button = st.sidebar.checkbox("Bật/Tắt Camera",key="camera_checkbox")
    custom_button  = st.sidebar.button('NHẬN DIỆN', key='custom_button')
     # Thêm thanh điều chỉnh xác suất vào sidebar
    confidence_threshold = st.sidebar.slider(
        'Độ tin cậy', min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )
    #Tạo biến kiểm soát dòng chảy
    flow_camera_button=False
    flow_upload_file=False
    if camera_button :
        flow_camera_button=True
        flow_upload_file=False
        if flow_camera_button:
            # Khởi tạo VideoCapture object để truy cập camera
            # Khởi tạo pygame
            cap = cv2.VideoCapture(0)
            stframe = st.empty()  # Khung trống để hiển thị ảnh từ camera
            
            label_colors = {}

            while camera_button:
                ret, frame = cap.read()  # Đọc khung hình từ camera
                if not ret:
                    st.error("Không thể truy cập camera")
                    break

                # Chuyển đổi khung hình từ BGR (OpenCV mặc định) sang RGB (Streamlit yêu cầu)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Nhận diện đối tượng
                results = model(frame_rgb)

                detected_objects = set()
                for i, (*box, conf, cls) in enumerate(results.xyxy[0]):
                    if conf >= confidence_threshold:
                        x1, y1, x2, y2 = box
                        label = model.names[int(cls)]
                        color_name = get_label_color1(label, label_colors)
                        try:
                            color = hex_to_bgr(color_name)  # Chuyển đổi màu từ hex thành BGR
                        except ValueError as e:
                            st.error(f"Lỗi chuyển đổi màu: {e}")
                            continue

                        # Vẽ hộp và nhãn lên khung hình
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        detected_objects.add(int(cls))
                # Hiển thị khung hình trong giao diện Streamlit
                frame=cv2.resize(frame,(0,0),fx=2,fy=1.5)
                stframe.image(frame,channels="BGR")
                # Phát âm thanh cảnh báo cho từng đối tượng phát hiện được 
                for obj in detected_objects:
                    speak_text(obj)

            cap.release()  # Giải phóng camera khi tắt checkbox
    if uploaded_file is not None:
        flow_camera_button=False
        flow_upload_file=True
        if flow_upload_file:
            distance=0
            # Nội dung của phần chính
            filename = uploaded_file.name.lower()
            if filename.endswith(('jpg', 'png', 'jpeg')):
                folder_name = os.path.splitext(uploaded_file.name)[0]
                save_dir = os.path.join("data/images", folder_name)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, uploaded_file.name)
                with open(save_path,"wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_bytes = uploaded_file.getvalue()
                encoded_image = base64.b64encode(file_bytes).decode("utf-8")
                    
                # Hiển thị ảnh gốc
                if not custom_button:
                    st.markdown("<h3 style='font-size:24px;'>Ảnh/Video đã tải lên</h3>", unsafe_allow_html=True)
                    # Sử dụng một div có kích thước cố định để hiển thị ảnh
                    st.markdown(
                        f"""
                        <div style="width: 400px; height: 400px; overflow: hidden; display: flex; justify-content: center; align-items: center; margin:0 auto;border: 1px solid #ddd;">
                            <img src="data:image/jpeg;base64,{encoded_image}" alt="Ảnh đã tải lên" style="max-width: 100%; max-height: 100%;align-items: center;justify-content: center" />
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                #Hiển thị cả ảnh gốc và ảnh nhận diện
                if custom_button:
                    image=Image.open(uploaded_file)
                    image = remove_exif(image)
                    image_np=np.array(image)
                    results=model(image_np)
                
                    # Lưu thông tin đối tượng vào mảng
                    detected_objects = []
                    object_count = {}
                    label_colors = {}
                    list_cls=set()
                    for i,(*box, conf, cls) in enumerate(results.xyxy[0]):
                        if conf >= confidence_threshold:
                            x1,y1,x2,y2=box
                            label = model.names[int(cls)]
                            if label not in object_count:
                                object_count[label] = 0
                            object_count[label] += 1

                            color = get_label_color(label, label_colors)

                            detected_objects.append({
                                "label_index": int(cls),
                                "label_name": label,
                                "confidence": float(conf),
                                "coordinates":(x1.item(),y1.item(),x2.item(),y2.item()),
                                "object_number":i+1,
                                "color": color  # Gán màu cho đối tượng
                            })
                            list_cls.add(int(cls))
                    
                    # Hiển thị ảnh với các đối tượng đã nhận diện
                    draw = ImageDraw.Draw(image)
                    try:
                        font = ImageFont.truetype("arial.ttf", 40)  # Sử dụng font Arial
                    except IOError:
                        font = ImageFont.load_default()  # Sử dụng font mặc định nếu không tìm thấy font Arial
                    
                    for obj in detected_objects:
                        x1, y1, x2, y2 = obj['coordinates']
                        object_number = obj['object_number']
                        color = obj['color']
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
                        draw.text((x1, y1), str(object_number), fill=color, font=font)
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")
                    detected_image_path = os.path.join(save_dir, f"detected_{uploaded_file.name}")
                    # Đặt lại vị trí con trỏ của buffer về đầu
                    # buffered.seek(0)
                    with open(detected_image_path, "wb") as f:
                        f.write(buffered.getvalue())
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    st.markdown(
                        """
                        <style>
                        .zoom {
                            transition: transform .2s; /* Animation */
                            margin: 0 auto;
                        }

                        .zoom:hover {
                            transform: scale(2); /* (150% zoom) */
                            z-index: 1000;
                            position: absolute;
                            top: 0;
                            left: 0;
                            right: 0;
                            bottom: 0;
                            margin: auto;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    # Lưu thông tin vào tệp nhật ký
                    log_filename = os.path.join(save_dir, 'logfile.txt')
                
                    with open(log_filename, 'w', encoding='utf-8') as logfile:
                        for obj in detected_objects:
                            logfile.write(f"Số đối tượng: {obj['object_number']}, Chỉ số nhãn: {obj['label_index']}, Tên nhãn: {obj['label_name']}, Độ tin cậy: {obj['confidence']:.2f}, Tọa độ: {obj['coordinates']}\n")
                    
                    st.success(f"Thông tin đã được lưu vào tệp {log_filename}")

                    # Tạo tệp ZIP của thư mục
                    zip_buffer = zip_folder_in_memory(save_dir)

                    # Hiển thị ảnh gốc và ảnh đã nhận diện cạnh nhau
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(
                            f"""
                            <div style="width: 400px; height: 400px; overflow: hidden; display: flex; justify-content: center; align-items: center; margin:0 auto;border: 1px solid #ddd;">
                                <img src="data:image/jpeg;base64,{encoded_image}" alt="Ảnh đã tải lên" style="max-width: 100%; max-height: 100%;align-items: center;justify-content: center" />
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown(
                            f"""
                            <div style="width: 400px; height: 400px; overflow: hidden; display: flex; justify-content: center; align-items: center; margin: 0 auto; border: 1px solid #ddd;">
                                <img class="zoom" src="data:image/jpeg;base64,{img_str}" alt="Ảnh đã nhận diện" style="max-width: 100%; max-height: 100%;align-items: center;justify-content: center" />
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.download_button(
                            label="Tải ảnh và logfile",
                            data=zip_buffer,
                            file_name=f"{uploaded_file.name}.zip",
                            mime="application/zip"
                        )
                    
                    for obj in list_cls:
                        speak_text(obj)
                    
                    
                    # Hiển thị thông tin các đối tượng nhận diện
                    st.markdown("<h3>Thông tin đối tượng nhận diện:</h3>", unsafe_allow_html=True)
                    st.markdown("<div style='display: flex; flex-wrap: wrap; justify-content: space-between;'>", unsafe_allow_html=True)
                    for obj in detected_objects:
                        label = obj['label_name']
                        count = object_count[label]
                        obj_ids = [o['object_number'] for o in detected_objects if o['label_name'] == label]
                        color = obj['color']
                        st.markdown(
                            f"""
                            <div style="border: 1px solid #ddd; padding: 10px; margin: 5px; text-align: left; flex: 1;">
                                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                    <div style="width: 50px; height: 50px; background-color: {color}; margin-right: 10px;"></div>
                                    <div>
                                        <div style="margin-bottom: 5px;"><strong>Tên biển:</strong> {label}</div>
                                        <div style="margin-bottom: 5px;"><strong>Số lượng:</strong> {count}</div>
                                        <div><strong>Mã đối tượng:</strong> {', '.join(map(str, obj_ids))}</div>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

            elif filename.endswith('mp4'):
                
                folder_name = os.path.splitext(uploaded_file.name)[0]
                save_dir = os.path.join("data/videos", folder_name)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, uploaded_file.name)
                with open(save_path, "wb") as f4:
                    f4.write(uploaded_file.getbuffer())
                st.markdown("<h3 style='font-size:24px;'>Video đã tải lên</h3>", unsafe_allow_html=True)

                video_bytes = uploaded_file.read()
                encoded_video = base64.b64encode(video_bytes).decode("utf-8")
                video_html = f"""
                    <div style="width: 500px; height: 500px;overflow: hidden; display: flex; justify-content: center; align-items: center; margin: 0 auto; border: 1px solid #ddd;">
                        <video width="500" height="500" controls>
                            <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                """
                st.markdown(video_html, unsafe_allow_html=True)
                object_counts={}
                if custom_button:
                    original_filename = uploaded_file.name
                    output_filename = os.path.join(save_dir, f"{os.path.splitext(original_filename)[0]}_out.mp4")
                    log_filename = os.path.join(save_dir, 'logfile.txt')            
                    cap = cv2.VideoCapture(save_path)               
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_filename, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    # Hiển thị văn bản thông báo
                    st.text("Hệ thống đang xử lý, vui lòng chờ...")
                    progress = st.progress(0)

                    # Tạo một đối tượng theo dõi
                    tracker = ObjectTracker()

                    with open(log_filename, 'w', encoding='utf-8') as logfile:
                        frame_count = 0
                        color_map = {}

                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            frame_count += 1
                            frame_name = f'frame_{frame_count}'
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image=Image.fromarray(frame_rgb)
                            results = model(frame_rgb)
                            detected_objects = []
                            draw = ImageDraw.Draw(image)
                            try:
                                font = ImageFont.truetype("arial.ttf", 40)
                            except IOError:
                                font = ImageFont.load_default()

                        
                            labels = []
                            bboxes = []
                            for i, (*box, conf, cls) in enumerate(results.xyxy[0]):
                                if conf >= confidence_threshold:
                                    x1, y1, x2, y2 = box
                                    label = model.names[int(cls)]
                                    if label not in color_map:
                                        color_map[label] = random_color()
                                    color = color_map[label]
                                    detected_objects.append({
                                        "object_number": i + 1,
                                        "label_name": label,
                                        "coordinates": (x1.item(), y1.item(), x2.item()),
                                        "confidence": float(conf),
                                        "color": color
                                    })
                                    labels.append(label)
                                    bboxes.append((x1.item(), y1.item(), x2.item(), y2.item()))
                            # Cập nhật đối tượng theo dõi
                            tracked_objects =tracker.update(labels,bboxes)
                            logfile.write(f"Frame: {frame_name}, Number of objects: {len(detected_objects)}\n")
                            for obj in detected_objects:
                                logfile.write(f"  Object Number: {obj['object_number']}, "
                                            f"Label Name: {obj['label_name']}, "
                                            f"Coordinates: {obj['coordinates']}, "
                                            f"Confidence: {obj['confidence']:.2f}, "
                                            f"Color: {obj['color']}\n")
                            # Vẽ các đối tượng và ID lên frame
                            for object_id, (label, bbox) in tracked_objects.items():
                                x1, y1, x2, y2 = bbox
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
                                draw.text((x1, y1), f"ID {object_id} {label}", fill=color, font=font)   
                            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                            out.write(frame_bgr)
                            
                            progress.progress(frame_count / total_frames)
                            
                        cap.release()
                        out.release()
                        # Ghi số lượng các đối tượng vào cuối logfile
                        object_counts = tracker.get_object_counts()
                        logfile.write("\nObject counts:\n")
                        for label, count in object_counts.items():
                            logfile.write(f"{label}: {count}\n")
                    # Hiển thị thông tin các đối tượng nhận diện
                    st.markdown("<h3>Thông tin đối tượng nhận diện:</h3>", unsafe_allow_html=True)
                    st.markdown("<div style='display: flex; flex-wrap: wrap; justify-content: space-between;'>", unsafe_allow_html=True)
                    for label, count in object_counts.items():
                        obj_ids = [obj_id for obj_id, (obj_label, _) in tracker.objects.items() if obj_label == label]
                        color = color_map[label]
                        st.markdown(
                            f"""
                            <div style="border: 1px solid #ddd; padding: 10px; margin: 5px; text-align: left; flex: 1;">
                                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                    <div style="width: 50px; height: 50px; background-color: {color}; margin-right: 10px;"></div>
                                    <div>
                                        <div style="margin-bottom: 5px;"><strong>Tên biển:</strong> {label}</div>
                                        <div style="margin-bottom: 5px;"><strong>Số lượng:</strong> {count}</div>
                                        <div><strong>Mã đối tượng:</strong> {', '.join(map(str, obj_ids))}</div>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    # Tạo tệp ZIP của thư mục
                    zip_buffer = zip_folder_in_memory(save_dir)
                    st.download_button(
                                label="Tải video và logfile",
                                data=zip_buffer,
                                file_name=f"{uploaded_file.name}.zip",
                                mime="application/zip"
                            )
                        
                    
                    
            st.markdown(f"""
                <div class="footer" style="padding-top:{distance}vh;padding-bottom:0,margin-bottom:0">
                    <hr>
                    <p style="color: black; font-weight:700;font-size:20px">Copyright © Đỗ Tùng Lâm</p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()