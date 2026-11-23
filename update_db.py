import json
import os
import sys
from groq import Groq
from tavily import TavilyClient

DB_FILE = "devices_db.json"


def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def search_device_info(manufacturer: str, model_code: str, hardware: str) -> str:
    """Tìm kiếm thông số thiết bị từ Internet qua Tavily API."""
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    device_name = (
        model_code
        if manufacturer.lower() in model_code.lower()
        else f"{manufacturer} {model_code}"
    )

    queries = [
        f"{device_name} specs chipset cpu gpu ram storage gsmarena nanoreview",
    ]
    if hardware:
        queries.append(f"{device_name} {hardware} specs smartphone")

    snippets = []
    seen_urls = set()

    for query in queries:
        try:
            response = tavily.search(
                query=query,
                search_depth="advanced",
                max_results=5,
            )
            results = response.get("results", [])
            for r in results:
                url = r.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                title = r.get("title", "")
                content = r.get("content", "")
                snippets.append(f"- Nguồn: {title} ({url})\n  Nội dung: {content}")

            if len(snippets) >= 3:
                break
        except Exception as e:
            print(f"Cảnh báo lỗi tìm kiếm Tavily với từ khóa '{query}': {e}")

    return "\n\n".join(snippets)


def get_available_groq_models(client: Groq) -> list[str]:
    """Tự động lấy danh sách các model chat đang hoạt động trên tài khoản Groq."""
    excluded_keywords = (
        "whisper",
        "tts",
        "playai",
        "orpheus",
        "guard",
        "moderation",
        "embed",
        "rerank",
        "compound",
        "allam",
    )
    available = []
    for m in client.models.list().data:
        mid = m.id
        low = mid.lower()
        if any(k in low for k in excluded_keywords):
            continue
        available.append(mid)

    preferred_order = ("llama-4", "qwen", "gpt-oss", "llama-3", "deepseek", "llama")

    def sort_key(mid: str):
        low = mid.lower()
        for idx, kw in enumerate(preferred_order):
            if kw in low:
                return (idx, mid)
        return (len(preferred_order), mid)

    available.sort(key=sort_key)
    return available


def extract_json_object(raw_text: str) -> dict:
    """Tách đối tượng JSON an toàn phòng trường hợp model có kèm ký tự thừa."""
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(raw_text[start : end + 1])
    return json.loads(raw_text)


def fetch_spec_from_groq(manufacturer: str, model_code: str, hardware: str, web_context: str) -> dict:
    """Tự chọn model đang hoạt động trên Groq để trích xuất đúng khung JSON."""
    client = Groq()
    models = get_available_groq_models(client)
    if not models:
        raise RuntimeError("Không tìm thấy model chat nào khả dụng trên tài khoản Groq.")

    print(f"Các model Groq khả dụng tìm thấy: {models[:5]}")

    system_prompt = (
        "Bạn là chuyên gia phân tích phần cứng điện thoại Android. "
        "Nhiệm vụ của bạn là đọc dữ liệu tìm kiếm web được cung cấp và trả về DUY NHẤT một đối tượng JSON hợp lệ "
        "theo đúng các khóa (key) dưới đây, không kèm giải thích ngoài JSON:\n"
        "{\n"
        '  "commercial_name": "Tên thương mại chính thức (VD: Xiaomi 17) hoặc null",\n'
        '  "soc_name": "Tên đầy đủ chip xử lý SoC hoặc null",\n'
        '  "process_nm": "Tiến trình sản xuất (VD: 3nm) hoặc null",\n'
        '  "cpu_architecture": "Cấu trúc nhân và xung nhịp CPU hoặc null",\n'
        '  "gpu_name": "Tên nhân đồ họa GPU (và xung nhịp nếu có) hoặc null",\n'
        '  "ram_type": "Chuẩn RAM (VD: LPDDR5X) hoặc null",\n'
        '  "storage_type": "Chuẩn bộ nhớ trong (VD: UFS 4.0) hoặc null"\n'
        "}\n"
        "Quy tắc bắt buộc: Chỉ lấy thông số có căn cứ từ dữ liệu web cung cấp. "
        "Nếu thông số nào không rõ hoặc không có trong nguồn, bắt buộc để giá trị null, tuyệt đối không tự đoán."
    )

    user_prompt = (
        f"Thông tin hệ thống gửi lên:\n"
        f"- Hãng sản xuất (Manufacturer): {manufacturer}\n"
        f"- Mã máy hệ thống (Model): {model_code}\n"
        f"- Mã phần cứng/Bo mạch (Hardware): {hardware}\n\n"
        f"Kết quả tìm kiếm trực tuyến mới nhất:\n"
        f"{web_context}\n\n"
        f"Hãy quy đổi mã máy '{model_code}' sang tên thương mại chuẩn và xuất ra JSON thông số."
    )

    last_error = None
    for model_id in models[:5]:
        try:
            print(f"Đang thử gọi model: {model_id}...")
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            raw_text = completion.choices[0].message.content
            print(f"Gọi thành công bằng model: {model_id}")
            return extract_json_object(raw_text)
        except Exception as e:
            print(f"Model {model_id} không phù hợp ({e}), chuyển sang model tiếp theo...")
            last_error = e

    raise RuntimeError(f"Tất cả model thử nghiệm đều lỗi. Lỗi cuối: {last_error}")


def main():
    if len(sys.argv) < 3:
        print("Thiếu tham số hãng và mã máy.")
        return

    manufacturer = sys.argv[1].strip()
    model_code = sys.argv[2].strip()
    hardware = sys.argv[3].strip() if len(sys.argv) > 3 else ""

    db = load_db()

    if model_code in db:
        print(f"Mã máy {model_code} đã có trong dữ liệu, bỏ qua.")
        return

    print(f"1. Đang tìm kiếm web qua Tavily cho: {manufacturer} - {model_code}...")
    web_context = search_device_info(manufacturer, model_code, hardware)

    if not web_context:
        print("Không thu thập được dữ liệu tìm kiếm từ Internet.")
        return

    print("2. Đã tìm thấy dữ liệu web. Đang gửi sang Groq trích xuất JSON...")
    spec = fetch_spec_from_groq(manufacturer, model_code, hardware, web_context)

    if spec.get("commercial_name") and spec.get("soc_name"):
        db[model_code] = spec
        save_db(db)
        print(f"Đã cập nhật thành công: {spec['commercial_name']} ({spec['soc_name']})")
    else:
        print("Không đủ dữ liệu xác thực (thiếu tên máy hoặc SoC), không lưu vào kho.")
        print("Dữ liệu AI trả về:", json.dumps(spec, ensure_ascii=False))


if __name__ == "__main__":
    main()
