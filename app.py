import gradio as gr
import google.generativeai as genai
import torch
from diffusers import DiffusionPipeline
import os
from PIL import Image

# --- คอนฟิกโมเดล (ตัวอย่างการโหลด Flux แบบเบา) ---
# หมายเหตุ: ใน RunPod ต้องเช่า GPU ที่ VRAM 24GB+ นะครับ
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_name):
    # ฟังก์ชันจำลองการเลือกโมเดล (ในอนาคตคุณสามารถเพิ่มเงื่อนไขโหลดจริงได้ที่นี่)
    # ตัวอย่าง: if model_name == "Flux.1": pipe = DiffusionPipeline.from_pretrained(...)
    return f"โหลดโมเดล {model_name} เรียบร้อยแล้ว (Simulated)"

# --- ฟังก์ชัน Auto Prompt ด้วย Gemini ---
def generate_auto_prompt(api_key, topic):
    if not api_key:
        return "กรุณาใส่ Gemini API Key ก่อนครับ"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash') # ใช้ตัว Flash เพราะเร็วและถูก
        instruction = f"Create a highly detailed, high-quality AI image generation prompt for the topic: '{topic}'. Keep it in English and focus on lighting, texture, and composition."
        response = model.generate_content(instruction)
        return response.text
    except Exception as e:
        return f"เกิดข้อผิดพลาดกับ API: {str(e)}"

# --- ฟังก์ชันการเจนภาพ (Core Logic) ---
def run_gen(model_name, ratio, txt_file, manual_prompt, auto_prompt, api_key):
    # 1. รวบรวม Prompt
    final_prompts = []
    
    # ลำดับความสำคัญ: ไฟล์ .txt > Manual Prompt > Auto Prompt
    if txt_file is not None:
        with open(txt_file.name, 'r', encoding='utf-8') as f:
            final_prompts = [line.strip() for line in f.readlines() if line.strip()]
    elif manual_prompt.strip():
        final_prompts = [manual_prompt]
    elif auto_prompt.strip():
        final_prompts = [auto_prompt]
    
    if not final_prompts:
        return "ไม่มี Prompt ให้ทำงานครับ", None

    # 2. จัดการ Aspect Ratio
    ratios = {
        "1:1": (1024, 1024),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "4:5": (896, 1152)
    }
    width, height = ratios.get(ratio, (1024, 1024))

    # 3. เริ่มการเจน (Simulation) 
    # ในการใช้งานจริงคุณต้องเอาโค้ด Pipe มาใส่ตรงนี้
    output_images = []
    status_msg = f"กำลังประมวลผล {len(final_prompts)} ภาพ ด้วยโมเดล {model_name}..."
    
    # สร้างโฟลเดอร์เก็บผลลัพธ์
    os.makedirs("outputs", exist_ok=True)
    
    # นี่คือตัวอย่าง Loop การทำงาน
    for i, p in enumerate(final_prompts):
        print(f"Generating {i+1}/{len(final_prompts)}: {p}")
        # สมมติว่านี่คือการเจนภาพจริง (Dummy Image)
        dummy_img = Image.new('RGB', (width, height), color = (73, 109, 137))
        img_path = f"outputs/img_{i}.png"
        dummy_img.save(img_path)
        output_images.append(img_path)

    return f"เสร็จสิ้น! เจนทั้งหมด {len(output_images)} ภาพ", output_images

# --- หน้าตา UI (Gradio Layout) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Sian Tui AI Web Gen v.1")
    gr.Markdown("สร้างภาพแบบ Batch พร้อมระบบ Auto Prompt โดยใช้ Gemini")
    
    with gr.Row():
        # ฝั่งซ้าย: ตั้งค่าและ Input
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🔑 API Config")
                api_key = gr.Textbox(label="Gemini API Key", placeholder="Paste your key here", type="password")
                topic = gr.Textbox(label="หัวข้อภาพ (Topic)", placeholder="เช่น: ผู้หญิงไทยในชุดไซเบอร์พังก์")
                gen_prompt_btn = gr.Button("🪄 ปั่น Prompt ออโต้", variant="secondary")
            
            with gr.Group():
                gr.Markdown("### ⚙️ Generation Settings")
                model_choice = gr.Dropdown(["Flux.1-dev", "Wan-2.1", "Z-image", "Qwen-VL-Gen"], label="เลือกโมเดล", value="Flux.1-dev")
                ratio_choice = gr.Radio(["1:1", "16:9", "9:16", "4:5"], label="Aspect Ratio", value="1:1")
                file_input = gr.File(label="อัปโหลดไฟล์ .txt (หนึ่งบรรทัดต่อหนึ่งภาพ)", file_types=[".txt"])

        # ฝั่งขวา: ผลลัพธ์และการควบคุม
        with gr.Column(scale=1):
            prompt_display = gr.TextArea(label="Prompt ที่จะใช้รัน", lines=5)
            manual_input = gr.Textbox(label="หรือพิมพ์ Prompt เอง (ถ้าไม่ได้อัปโหลดไฟล์)")
            
            run_btn = gr.Button("🔥 START GENERATION", variant="primary")
            status_output = gr.Text(label="สถานะ")
            gallery_output = gr.Gallery(label="ผลลัพธ์ภาพ", columns=2, height="auto")

    # --- ระบบเชื่อมต่อปุ่ม (Events) ---
    gen_prompt_btn.click(
        generate_auto_prompt, 
        inputs=[api_key, topic], 
        outputs=prompt_display
    )
    
    run_btn.click(
        run_gen,
        inputs=[model_choice, ratio_choice, file_input, manual_input, prompt_display, api_key],
        outputs=[status_output, gallery_output]
    )

if __name__ == "__main__":
    # เปิด Share=True เพื่อให้ได้ลิงก์ .gradio.live ไว้รันนอก RunPod ได้
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
