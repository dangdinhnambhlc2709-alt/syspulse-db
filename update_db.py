import json
import os
import sys
from groq import Groq
from tavily import TavilyClient

DB_FILE = "devices_db.json"
GROQ_MODEL = "llama-3.3-70b-versatile"


def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def search_device_info(manufacturer: str, model_code: str, hardware: str) -> str:
    """Tìm kiếm thông số thiết bị từ Internet qua Tavily API (không bị chặn trên GitHub)."""
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # Tránh lặp tên hãng (VD: tránh tạo thành "Xiaomi Xiaomi 17")
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

            # Nếu lượt tìm đầu đã lấy đủ nguồn chi tiết thì không cần tốn thêm lượt thứ 2
            if len(snippets) >= 3:
                break
        except Exception as e:
            print(f"Cảnh báo lỗi tìm kiếm Tavily với từ khóa '{query}': {e}")

    return "\n\n".join(snippets)


def fetch_spec_from_groq(manufacturer: str, model_code: str, hardware: str, web_context: str) -> dict:
    """Dùng Groq AI đọc kết quả tìm kiếm web và trích xuất đúng khung JSON."""
    client = Groq()

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

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw_text = completion.choices[0].message.content
    return json.loads(raw_text)


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
