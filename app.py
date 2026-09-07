"""
app.py
โปรแกรมจำแนกโรค Covid จากภาพ X-ray

*** สำคัญ: หลังตรวจสอบไฟล์ .pkcls ทั้ง 3 ไฟล์ที่อัปโหลดมาจริงแล้ว พบว่า ***
- โมเดลถูกฝึกและบันทึกออกมาจากโปรแกรม Orange Data Mining (Orange3) ไม่ใช่
  scikit-learn ตรงๆ (แม้ตัว regressor/classifier ข้างในจะเป็น sklearn ก็ตาม)
  จึงต้องใช้ไลบรารี Orange3 ในการโหลดและทำนายด้วย
- ตัวแปรต้น (features) ที่โมเดลใช้ไม่ใช่ค่าที่กรอกเอง แต่เป็นเวกเตอร์
  "Image Embedding" 2,048 มิติ (คอลัมน์ชื่อ n0-n2047) ที่ Orange สกัดออกมา
  จากภาพ X-ray ด้วยโมเดล deep learning (widget "Image Embedding")
- Class ที่ทำนาย (class_var "category") มี 3 ค่า: covid, normal, pneumonia

ดังนั้นแอปนี้จึงเป็นแบบ "อัปโหลดภาพ X-ray" ไม่ใช่แบบฟอร์มกรอกตัวเลข
"""

import os
import glob
import tempfile

import numpy as np
import joblib
import streamlit as st

import Orange
from Orange.data import Domain, Table
from Orange.base import Model as OrangeModel

# =========================================================================
# 1) ตั้งค่าหน้าเว็บ และหัวข้อแอป
# =========================================================================
st.set_page_config(page_title="จำแนกโรค Covid จากภาพ X-ray", page_icon="🫁")
st.title("โปรแกรมจำแนกโรค Covid จากภาพ X-ray")

# =========================================================================
# 2) ค่ากำหนด (CONFIG)
# =========================================================================
# โฟลเดอร์ที่เก็บไฟล์โมเดล .pkcls (ดาวน์โหลด/คัดลอกไฟล์จากโฟลเดอร์ ModelW10
# บน Google Drive มาวางไว้ที่นี่ก่อนรันแอป)
MODEL_DIR = "models"

# ตัวเลือกโมเดล embedder ของ Orange ("Image Embedding" widget)
# key ที่ใช้เรียก ImageEmbedder ต้องตรงกับใน orangecontrib.imageanalytics
# *** ต้องเลือกให้ตรงกับตอนฝึกโมเดลจริง ไม่งั้นค่าที่ได้จะผิด/ทำนายเพี้ยน ***
EMBEDDER_OPTIONS = {
    "Inception v3 (ค่าเริ่มต้นของ Orange, 2048 มิติ, ผ่านอินเทอร์เน็ต)": "inception-v3",
    "SqueezeNet (ประมวลผลในเครื่อง ไม่ต้องต่อเน็ต)": "squeezenet",
    "Painters (ผ่านอินเทอร์เน็ต)": "painters",
    "VGG-16 (ผ่านอินเทอร์เน็ต)": "vgg16",
    "VGG-19 (ผ่านอินเทอร์เน็ต)": "vgg19",
    "DeepLoc (ผ่านอินเทอร์เน็ต)": "deeploc",
    "openface (ผ่านอินเทอร์เน็ต)": "openface",
}

# แปลผล class จากภาษาอังกฤษ (ตามที่พบในโมเดลจริง) เป็นข้อความอ่านง่าย
CLASS_LABELS_TH = {
    "covid": "🦠 โควิด-19 (COVID-19)",
    "normal": "✅ ปกติ (Normal)",
    "pneumonia": "🫁 ปอดอักเสบจากเชื้ออื่น (Pneumonia)",
}


# =========================================================================
# 3) ฟังก์ชันโหลดโมเดลด้วย joblib (ต้องมี Orange3 ติดตั้งไว้ถึงจะโหลดสำเร็จ)
# =========================================================================
@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(model_path: str):
    """โหลดไฟล์โมเดล .pkcls ด้วย joblib (โมเดลเป็นชนิด Orange.base.Model)"""
    return joblib.load(model_path)


def get_available_models(model_dir: str):
    """หาไฟล์ .pkcls ทั้งหมดในโฟลเดอร์ที่กำหนด"""
    if not os.path.isdir(model_dir):
        return []
    return sorted(glob.glob(os.path.join(model_dir, "*.pkcls")))


# =========================================================================
# 4) ส่วนเลือกโมเดล (Sidebar) - ให้ผู้ใช้เลือกโมเดลเองได้
#    รองรับ 2 วิธี: (ก) เลือกจากไฟล์ในโฟลเดอร์ models/  (ข) อัปโหลดไฟล์เอง
# =========================================================================
st.sidebar.header("⚙️ เลือกโมเดล")

model = None
model_source_label = None

model_files = get_available_models(MODEL_DIR)

choice_mode = st.sidebar.radio(
    "วิธีเลือกโมเดล",
    options=["เลือกจากโฟลเดอร์ models/", "อัปโหลดไฟล์ .pkcls เอง"],
)

if choice_mode == "เลือกจากโฟลเดอร์ models/":
    if model_files:
        selected_path = st.sidebar.selectbox(
            "เลือกไฟล์โมเดล (.pkcls)",
            options=model_files,
            format_func=lambda p: os.path.basename(p),
        )
        model = load_model(selected_path)
        model_source_label = os.path.basename(selected_path)
    else:
        st.sidebar.warning(
            f"ไม่พบไฟล์ .pkcls ในโฟลเดอร์ '{MODEL_DIR}/' "
            f"กรุณาคัดลอกไฟล์โมเดล (เช่น w10_Model_NN.pkcls, "
            f"w10_Model_SVM.pkcls, w10_Model_Tree.pkcls) มาวางไว้ในโฟลเดอร์นี้ "
            f"หรือเลือกวิธี 'อัปโหลดไฟล์ .pkcls เอง' แทน"
        )
else:
    uploaded_model_file = st.sidebar.file_uploader(
        "อัปโหลดไฟล์โมเดล (.pkcls)", type=["pkcls"]
    )
    if uploaded_model_file is not None:
        temp_model_path = "temp_uploaded_model.pkcls"
        with open(temp_model_path, "wb") as f:
            f.write(uploaded_model_file.getbuffer())
        model = load_model(temp_model_path)
        model_source_label = uploaded_model_file.name

if model is not None:
    st.sidebar.success(f"โหลดโมเดล '{model_source_label}' สำเร็จ ✅")
    n_features_expected = len(model.domain.attributes)
    st.sidebar.caption(f"โมเดลนี้ต้องการ embedding {n_features_expected} มิติ")

st.markdown("---")

# =========================================================================
# 5) ส่วนอัปโหลดภาพ X-ray และเลือกโมเดล embedder
# =========================================================================
st.subheader("อัปโหลดภาพ X-ray")

uploaded_image = st.file_uploader(
    "เลือกไฟล์ภาพ X-ray (jpg, jpeg, png)", type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    st.image(uploaded_image, caption="ภาพที่อัปโหลด", use_container_width=True)

embedder_label = st.selectbox(
    "โมเดล Image Embedding (ต้องเลือกให้ตรงกับตอนฝึกโมเดลใน Orange)",
    options=list(EMBEDDER_OPTIONS.keys()),
    index=0,  # ค่าเริ่มต้น = Inception v3 (ตรงกับ 2048 มิติที่ตรวจพบในโมเดล)
)
embedder_key = EMBEDDER_OPTIONS[embedder_label]

st.caption(
    "⚠️ โมเดล embedder ส่วนใหญ่ (ยกเว้น SqueezeNet) ประมวลผลผ่านเซิร์ฟเวอร์ของ "
    "Orange (biolab) บนอินเทอร์เน็ต ซึ่งหมายความว่าภาพที่อัปโหลดจะถูกส่งออกไปนอกเครื่อง "
    "เช่นเดียวกับตอนที่ใช้ widget 'Image Embedding' ใน Orange"
)

st.markdown("---")


# =========================================================================
# 6) ปุ่มทำนายผล
# =========================================================================
if st.button("ทำนายผล", type="primary"):
    if model is None:
        st.error("กรุณาเลือกหรืออัปโหลดไฟล์โมเดล (.pkcls) ก่อนทำการทำนาย")
    elif uploaded_image is None:
        st.error("กรุณาอัปโหลดภาพ X-ray ก่อนทำการทำนาย")
    else:
        tmp_path = None
        try:
            # 6.1 บันทึกภาพที่อัปโหลดลงไฟล์ชั่วคราว เพราะ ImageEmbedder ของ Orange
            #      รับ path ของไฟล์ภาพ ไม่ใช่ bytes โดยตรง
            suffix = os.path.splitext(uploaded_image.name)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_image.getbuffer())
                tmp_path = tmp.name

            # 6.2 สกัด embedding จากภาพด้วยโมเดลเดียวกับที่ใช้ตอนฝึก
            with st.spinner("กำลังสกัด embedding จากภาพ (อาจใช้เวลาสักครู่)..."):
                from orangecontrib.imageanalytics.image_embedder import ImageEmbedder

                with ImageEmbedder(model=embedder_key) as embedder:
                    embeddings = embedder([tmp_path])

            if not embeddings or embeddings[0] is None:
                st.error(
                    "ไม่สามารถสกัด embedding จากภาพนี้ได้ "
                    "(เซิร์ฟเวอร์ embedding อาจไม่ตอบสนอง หรือไฟล์ภาพเสียหาย) "
                    "กรุณาลองใหม่อีกครั้ง"
                )
            else:
                X = np.array(embeddings[0], dtype=float).reshape(1, -1)
                n_expected = len(model.domain.attributes)

                # 6.3 ตรวจสอบจำนวนมิติให้ตรงกับที่โมเดลต้องการ
                if X.shape[1] != n_expected:
                    st.error(
                        f"จำนวนมิติของ embedding ที่ได้ ({X.shape[1]}) "
                        f"ไม่ตรงกับที่โมเดลต้องการ ({n_expected} มิติ) "
                        f"กรุณาลองเปลี่ยนตัวเลือกโมเดล Image Embedding ด้านบน "
                        f"ให้ตรงกับตอนฝึกโมเดลใน Orange แล้วลองใหม่"
                    )
                else:
                    # 6.4 จัดข้อมูลเป็น Orange Table ตาม domain (features) ของโมเดล
                    domain_x = Domain(model.domain.attributes)
                    table = Table.from_numpy(domain_x, X)

                    # 6.5 ทำนายผล พร้อมความน่าจะเป็นของแต่ละ class
                    values, probs = model(table, ret=OrangeModel.ValueProbs)

                    class_var = model.domain.class_var
                    pred_label = class_var.values[int(values[0])]
                    pred_text = CLASS_LABELS_TH.get(pred_label, pred_label)

                    # 6.6 แสดงผลลัพธ์ให้อ่านเข้าใจง่าย
                    if pred_label == "covid":
                        st.error(f"### ผลการทำนาย: {pred_text}")
                    elif pred_label == "normal":
                        st.success(f"### ผลการทำนาย: {pred_text}")
                    else:
                        st.warning(f"### ผลการทำนาย: {pred_text}")

                    st.write("**ความน่าจะเป็นของแต่ละกลุ่ม (Probability):**")
                    for cls_name, p in zip(class_var.values, probs[0]):
                        label = CLASS_LABELS_TH.get(cls_name, cls_name)
                        st.write(f"- {label}: **{p * 100:.2f}%**")
                        st.progress(min(max(float(p), 0.0), 1.0))

        except ModuleNotFoundError as e:
            st.error(
                "ขาดไลบรารีที่จำเป็น: "
                f"{e}. กรุณาติดตั้งตาม requirements.txt (Orange3, "
                "Orange3-ImageAnalytics) ให้ครบก่อนรันแอป"
            )
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่างการทำนาย: {e}")
        finally:
            # ลบไฟล์ภาพชั่วคราวทิ้งเสมอ
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

# =========================================================================
# 7) คำอธิบายท้ายหน้าเว็บ
# =========================================================================
st.markdown("---")
st.caption(
    "โปรแกรมนี้ใช้เพื่อการศึกษาเท่านั้น ไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์ "
    "โดยแพทย์ผู้เชี่ยวชาญได้"
)
