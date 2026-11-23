import json
import os
import sys
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 1. Định nghĩa khung JSON cố định khớp với các mục trong SysPulse
class DeviceSpec(BaseModel):
    commercial_name: str | None = Field(description="Tên thương mại chính thức, VD: Xiaomi 17")
    soc_name: str | None = Field(description="Tên đầy đủ chip xử lý SoC")
    process_nm: str | None = Field(description="Tiến trình sản xuất, VD: 3nm")
    cpu_architecture: str | None = Field(description="Cấu trúc nhân và xung nhịp CPU")
    gpu_name: str | None = Field(description="Tên nhân đồ họa GPU và xung nhịp nếu có")
    ram_type: str | None = Field(description="Chuẩn RAM, VD: LPDDR5X")
    storage_type: str | None = Field(description="Chuẩn bộ nhớ trong, VD: UFS 4.0")

DB_FILE = "devices_db.json"

def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def fetch_spec_from_gemini(manufacturer: str, model_code: str, hardware: str):
    # Tự động đọc biến môi trường GEMINI_API_KEY từ GitHub Secrets
    client = genai.Client()

    prompt = (
        f"Tra cứu thông số phần cứng chính xác từ GSMArena, NanoReview hoặc trang chủ hãng cho thiết bị Android sau:\n"
        f"- Hãng sản xuất (Manufacturer): {manufacturer}\n"
        f"- Mã máy hệ thống (Model): {model_code}\n"
        f"- Mã phần cứng/Bo mạch (Hardware): {hardware}\n\n"
        f"Quy tắc bắt buộc:\n"
        f"1. Trước tiên hãy quy đổi mã máy '{model_code}' sang đúng tên thương mại.\n"
        f"2. Chỉ điền thông số có nguồn xác thực trên web. Nếu không chắc chắn hoặc không có thông tin, bắt buộc trả về null, tuyệt đối không tự đoán."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", # Có thể đổi sang "gemini-3.8-flash" nếu muốn độ chi tiết cao hơn
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())], # Bật tìm kiếm Google trực tiếp
            response_mime_type="application/json",
            response_schema=DeviceSpec,
            temperature=0.1, # Giữ nhiệt độ thấp để AI trả lời chuẩn xác, không sáng tạo bừa
        ),
    )
    return json.loads(response.text)

def main():
    if len(sys.argv) < 3:
        print("Thiếu tham số hãng và mã máy.")
        return

    manufacturer = sys.argv[1].strip()
    model_code = sys.argv[2].strip()
    hardware = sys.argv[3].strip() if len(sys.argv) > 3 else ""

    db = load_db()

    # Nếu mã máy đã tồn tại trong kho thì dừng ngay để không tốn API
    if model_code in db:
        print(f"Mã máy {model_code} đã có trong dữ liệu, bỏ qua.")
        return

    print(f"Đang tra cứu máy mới: {manufacturer} - {model_code}...")
    spec = fetch_spec_from_gemini(manufacturer, model_code, hardware)

    # Chỉ lưu nếu AI tìm được tối thiểu tên thương mại và tên chip SoC
    if spec.get("commercial_name") and spec.get("soc_name"):
        db[model_code] = spec
        save_db(db)
        print(f"Đã cập nhật thành công: {spec['commercial_name']}")
    else:
        print("Không đủ dữ liệu xác thực, không lưu vào kho.")

if __name__ == "__main__":
    main()
