import gradio as gr
import zipfile
import os
import tempfile
from PIL import Image
import uuid

# --- ฟังก์ชันจำลองการเจนรูป (ต้องเอาโค้ดโหลด Model จริงมาใส่ตรงนี้) ---
def generate_single_image(prompt, model_name, ratio):
    # กำหนดขนาดภาพตาม Ratio
    width, height = 1024, 1024 # ค่าเริ่มต้น 1:1
    if ratio == "16:9": width, height = 1280, 720
    elif ratio == "9:16": width, height = 720, 1280
    elif ratio == "4:3": width, height = 1024, 768
    
    # [ใส่โค้ด Inference ของ Flux, Qwen, หรือ Zimage ตรงนี้]
    # ตัวอย่าง: สร้างภาพสีเทาเปล่าๆ เป็น Placeholder
    dummy_img = Image.new('RGB', (width, height), color=(150, 150, 150))
    return dummy_img

# --- ฟังก์ชันหลักที่ทำงานเมื่อกดปุ่ม Generate ---
def process_generation(text_prompt, txt_file, model_choice, num_per_prompt, ratio_choice):
    prompts = []
    
    # 1. ดึงข้อความจาก Textbox
    if text_prompt.strip():
        prompts.append(text_prompt.strip())
        
    # 2. ดึงข้อความจากไฟล์ .txt (ถ้ามีอัปโหลดมา)
    if txt_file is not None:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            prompts.extend([line.strip() for line in lines if line.strip()])
            
    if not prompts:
        return None, "กรุณาใส่ Prompt หรืออัปโหลดไฟล์ .txt"

    # สร้างโฟลเดอร์ชั่วคราวสำหรับเก็บรูปและไฟล์ Zip
    temp_dir = tempfile.mkdtemp()
    zip_filename = os.path.join(temp_dir, f"generated_images_{uuid.uuid4().hex[:6]}.zip")
    
    generated_images = []
    
    # 3. เริ่มลูปเจนรูปตาม Prompt และจำนวนที่ตั้งไว้
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for i, prompt in enumerate(prompts):
            for j in range(num_per_prompt):
                img = generate_single_image(prompt, model_choice, ratio_choice)
                
                # เซฟรูปชั่วคราว
                img_name = f"prompt_{i+1}_img_{j+1}.png"
                img_path = os.path.join(temp_dir, img_name)
                img.save(img_path)
                
                # เอาลง Zip และเก็บไว้แสดงผลบน UI
                zipf.write(img_path, arcname=img_name)
                generated_images.append(img_path)

    return generated_images, zip_filename

# --- สร้าง Gradio UI ---
with gr.Blocks(title="AutoGen Image Studio") as app:
    gr.Markdown("## 🎨 AutoGen Image Studio (RunPod Template)")
    
    with gr.Row():
        with gr.Column():
            text_prompt = gr.Textbox(label="ใส่ Prompt ตรงนี้", placeholder="A cute cat cyberpunk style...", lines=3)
            txt_file = gr.File(label="หรืออัปโหลดไฟล์ .txt (1 บรรทัด = 1 Prompt)", file_types=[".txt"])
            
            with gr.Row():
                model_choice = gr.Dropdown(choices=["Flux", "Qwen", "Zimage"], value="Flux", label="เลือก Model")
                ratio_choice = gr.Radio(choices=["1:1", "16:9", "9:16", "4:3"], value="1:1", label="Aspect Ratio")
            
            num_per_prompt = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="จำนวนรูปต่อ 1 Prompt")
            
            generate_btn = gr.Button("🚀 Generate Images", variant="primary")
            
        with gr.Column():
            gallery_out = gr.Gallery(label="Images Output", columns=2)
            zip_out = gr.File(label="ดาวน์โหลดไฟล์ .zip ที่นี่")
            status_out = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=process_generation,
        inputs=[text_prompt, txt_file, model_choice, num_per_prompt, ratio_choice],
        outputs=[gallery_out, zip_out]
    )

if __name__ == "__main__":
    # รันบน 0.0.0.0 เพื่อให้ RunPod ทำ Port Forwarding ได้ (ปกติใช้ Port 7860 หรือ 8000)
    app.launch(server_name="0.0.0.0", server_port=7860)
