import gradio as gr
import google.generativeai as genai
import torch
from diffusers import FluxPipeline
import os
from PIL import Image
import uuid

# --- 🚀 โหลดโมเดล FLUX.1-schnell ---
print("กำลังเตรียมโหลดโมเดล FLUX.1 (อาจใช้เวลาสักครู่ในการรันครั้งแรก)...")
model_id = "black-forest-labs/FLUX.1-schnell"

# ใช้ bfloat16 เพื่อประหยัด VRAM และเพิ่มความเร็ว
pipe = FluxPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16
).to("cuda")

# --- ฟังก์ชัน Auto Prompt ด้วย Gemini ---
def generate_auto_prompt(api_key, topic):
    if not api_key:
        return "กรุณาใส่ Gemini API Key ก่อนครับ"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        instruction = f"Create a highly detailed, professional AI image prompt for: '{topic}'. Keep it in English, descriptive, and focus on artistic style."
        response = model.generate_content(instruction)
        return response.text
    except Exception as e:
        return f"เกิดข้อผิดพลาดกับ API: {str(e)}"

# --- ฟังก์ชันหลักในการเจนภาพ (Real Inference) ---
def run_gen(ratio, txt_file, manual_prompt, auto_prompt):
    # 1. รวบรวม Prompt
    final_prompts = []
    if txt_file is not None:
        with open(txt_file.name, 'r', encoding='utf-8') as f:
            final_prompts = [line.strip() for line in f.readlines() if line.strip()]
    elif manual_prompt.strip():
        final_prompts = [manual_prompt]
    elif auto_prompt.strip():
        final_prompts = [auto_prompt]
    
    if not final_prompts:
        return "ไม่มี Prompt ให้ทำงานครับ", None

    # 2. ตั้งค่า Aspect Ratio
    ratios = {
        "1:1": (1024, 1024),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "4:5": (896, 1152)
    }
    width, height = ratios.get(ratio, (1024, 1024))

    # 3. เริ่มการเจนภาพจริงด้วย FLUX
    output_images = []
    os.makedirs("outputs", exist_ok=True)
    
    print(f"กำลังเริ่มเจนภาพจำนวน {len(final_prompts)} ภาพ...")
    
    for i, p in enumerate(final_prompts):
        try:
            # รันการคำนวณภาพ
            image = pipe(
                prompt=p,
                width=width,
                height=height,
                num_inference_steps=4, # Schnell ใช้ 4 steps ก็เทพแล้ว
                guidance_scale=0.0,    # FLUX Schnell ไม่ต้องใช้ Guidance
                max_sequence_length=256
            ).images[0]
            
            # ตั้งชื่อไฟล์แบบสุ่มกันซ้ำ
            file_name = f"outputs/{uuid.uuid4()}.png"
            image.save(file_name)
            output_images.append(file_name)
        except Exception as e:
            print(f"Error เจนภาพที่ {i}: {str(e)}")

    return f"เจนเสร็จแล้วทั้งหมด {len(output_images)} ภาพ!", output_images

# --- หน้าตา UI ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="orange")) as demo:
    gr.Markdown("# ⚡ FLUX.1 Schnell - Batch Generator")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                api_key = gr.Textbox(label="Gemini API Key", type="password")
                topic = gr.Textbox(label="หัวข้อภาพ (Auto Prompt)")
                gen_prompt_btn = gr.Button("🪄 ปั่น Prompt ออโต้")
            
            ratio_choice = gr.Radio(["1:1", "16:9", "9:16", "4:5"], label="Aspect Ratio", value="1:1")
            file_input = gr.File(label="อัปโหลด .txt สำหรับทำ Batch", file_types=[".txt"])

        with gr.Column(scale=1):
            prompt_display = gr.TextArea(label="Prompt ที่จะใช้รัน", lines=4)
            manual_input = gr.Textbox(label="หรือพิมพ์ Prompt เองที่นี่")
            run_btn = gr.Button("🚀 START GENERATING", variant="primary")
            status_output = gr.Text(label="สถานะ")
            gallery_output = gr.Gallery(label="ผลลัพธ์", columns=2)

    # เชื่อมต่อปุ่ม
    gen_prompt_btn.click(generate_auto_prompt, inputs=[api_key, topic], outputs=prompt_display)
    run_btn.click(run_gen, inputs=[ratio_choice, file_input, manual_input, prompt_display], outputs=[status_output, gallery_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
