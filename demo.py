import math

MAX_SPINE_ANGLE = 10.0  
SD_TABLE_DEMO = {"-3SD": 12.9, "-2SD": 14.4, "+1SD": 20.5, "+2SD": 23.2}

INTENTS = {
    "Masalah Gizi": ["susah makan", "sayur", "nafsu makan", "gizi", "makanan", "pendek"],
    "Pertumbuhan": ["tinggi", "berat", "stunting", "tumbuh", "obesitas"],
}

KNOWLEDGE_BASE = {
    "Masalah Gizi": (
        "Coba sajikan porsi kecil tapi sering, variasikan bentuk dan warna sayur, "
        "libatkan anak saat menyiapkan makanan, dan jaga jadwal makan teratur. "
        "Jika berat badan anak turun atau tidak naik selama beberapa bulan, konsultasikan ke Puskesmas."
    ),
    "Pertumbuhan": (
        "Pantau tinggi dan berat badan anak rutin tiap bulan lalu bandingkan dengan kurva standar Kemenkes. "
        "Bila tren menjauhi kurva normal, segera konsultasi ke tenaga kesehatan."
    ),
}

NEGATIVE_WORDS = ["lelah", "khawatir", "bingung", "stres", "takut", "cemas"]
NEGATIONS = ["tidak", "tak", "nggak", "gak", "bukan", "belum"]

RECOMMENDATIONS = {
    "normal": "Pertahankan pola makan seimbang dan aktivitas fisik rutin.",
    "lebih": "Kurangi minuman manis dan makanan olahan, tambah aktivitas fisik minimal 60 menit/hari, "
             "dan konsultasikan ke tenaga kesehatan.",
    "kurang": "Tingkatkan asupan protein hewani dan kalori padat gizi, lakukan stimulasi motorik, "
              "dan segera periksakan ke Puskesmas.",
}

def pixel_to_cm(body_pixels, reference_pixels, reference_cm):
    if reference_pixels <= 0:
        raise ValueError("Piksel referensi harus lebih dari nol.")
    return (body_pixels / reference_pixels) * reference_cm


def mape(actual, predicted):
    if actual <= 0:
        raise ValueError("Ukuran aktual harus lebih dari nol.")
    return abs(actual - predicted) / actual * 100.0


def spine_angle(shoulder_x, shoulder_y, hip_x, hip_y):
    dx = abs(shoulder_x - hip_x)
    dy = abs(hip_y - shoulder_y)
    return math.degrees(math.atan2(dx, dy))  

def check_posture(angle, max_angle=MAX_SPINE_ANGLE):
    if angle > max_angle:
        return "RISIKO - kemungkinan kelainan tulang belakang"
    return "Postur normal"


def analyze_image_mediapipe(image_path, reference_pixels, reference_cm):
    """Opsional: pose estimation sungguhan. Butuh `pip install mediapipe opencv-python`.
    Belum diuji di lingkungan ini; sesuaikan dengan versi MediaPipe yang Anda pakai."""
    import cv2
    import mediapipe as mp

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Gambar tidak bisa dibaca.")
    h, w = img.shape[:2]
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        raise ValueError("Tubuh tidak terdeteksi.")
    lm = res.pose_landmarks.landmark
    P = mp.solutions.pose.PoseLandmark
    head_y = lm[P.NOSE].y * h
    heel_y = max(lm[P.LEFT_HEEL].y, lm[P.RIGHT_HEEL].y) * h
    body_px = abs(heel_y - head_y)  
    sh_x = (lm[P.LEFT_SHOULDER].x + lm[P.RIGHT_SHOULDER].x) / 2 * w
    sh_y = (lm[P.LEFT_SHOULDER].y + lm[P.RIGHT_SHOULDER].y) / 2 * h
    hip_x = (lm[P.LEFT_HIP].x + lm[P.RIGHT_HIP].x) / 2 * w
    hip_y = (lm[P.LEFT_HIP].y + lm[P.RIGHT_HIP].y) / 2 * h
    return {
        "height_cm": pixel_to_cm(body_px, reference_pixels, reference_cm),
        "angle": spine_angle(sh_x, sh_y, hip_x, hip_y),
    }


def linear_regression(xs, ys):
    n = len(xs)
    if n < 2 or len(ys) != n:
        raise ValueError("Regresi butuh minimal dua data dengan jumlah x dan y sama.")
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        raise ValueError("Variansi x nol; slope tidak dapat dihitung.")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return my - b * mx, b


def predict(a, b, x):
    return a + b * x


def classify_weight(value, sd):
    """Kategori mengikuti pola BB/U Permenkes 2/2020. sd = dict berisi -3SD, -2SD, +1SD, +2SD."""
    if value < sd["-3SD"]:
        return "Berat badan sangat kurang", "kurang"
    if value < sd["-2SD"]:
        return "Berat badan kurang", "kurang"
    if value > sd["+2SD"]:
        return "Di atas +2 SD (risiko obesitas, konfirmasi dengan IMT/U)", "lebih"
    if value > sd["+1SD"]:
        return "Risiko berat badan lebih", "lebih"
    return "Berat badan normal", "normal"


def early_warning(prediction, sd):
    category, level = classify_weight(prediction, sd)
    if level == "normal":
        return "Tren normal", category, RECOMMENDATIONS["normal"]
    return "EARLY WARNING - proyeksi keluar dari rentang normal", category, RECOMMENDATIONS[level]


def forecast(a, b, last_month, sd, horizon=6):
    """Proyeksi 6 bulan ke depan (sesuai laporan); mengembalikan list (bulan, nilai, kategori)."""
    out = []
    for m in range(int(last_month) + 1, int(last_month) + horizon + 1):
        v = predict(a, b, m)
        out.append((m, v, classify_weight(v, sd)[0]))
    return out

def tokenize(text):
    cleaned = "".join(c.lower() if c.isalnum() or c == " " else " " for c in text)
    return cleaned.split()


def classify_intent(text):
    """Mengembalikan (daftar keyword, daftar intent). Pencocokan berbasis kata utuh."""
    padded = " " + " ".join(tokenize(text)) + " "
    found_kw, found_intents = [], []
    for intent, keywords in INTENTS.items():
        hits = [k for k in keywords if f" {k} " in padded]
        if hits:
            found_kw += hits
            found_intents.append(intent)
    return found_kw, (found_intents or ["Unknown"])


def answer_question(text):
    _, intents = classify_intent(text)
    answers = [KNOWLEDGE_BASE[i] for i in intents if i in KNOWLEDGE_BASE]
    return " ".join(answers) if answers else "Maaf, saya belum memahami pertanyaan itu. Silakan hubungi tenaga kesehatan."


def analyze_sentiment(journal):
    tokens = tokenize(journal)
    negatives = []
    for i, w in enumerate(tokens):
        if w in NEGATIVE_WORDS:
            negated = i > 0 and tokens[i - 1] in NEGATIONS
            if not negated:
                negatives.append(w)
    status = "Butuh Dukungan" if negatives else "Stabil"
    return status, negatives


def empathetic_response(status):
    if status == "Butuh Dukungan":
        return ("Terima kasih sudah berbagi. Wajar merasa seperti itu dalam mengasuh anak. "
                "Anda tidak sendirian; luangkan waktu istirahat dan bicarakan dengan keluarga atau tenaga kesehatan.")
    return "Senang mendengar kondisi Anda stabil. Tetap semangat mendampingi tumbuh kembang anak!"


def confusion_matrix(actual, predicted, positive="Butuh Dukungan"):
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Data aktual dan prediksi harus sama panjang dan tidak kosong.")
    tp = sum(1 for a, p in zip(actual, predicted) if a == positive and p == positive)
    fp = sum(1 for a, p in zip(actual, predicted) if a != positive and p == positive)
    fn = sum(1 for a, p in zip(actual, predicted) if a == positive and p != positive)
    tn = len(actual) - tp - fp - fn
    acc = (tp + tn) / len(actual)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def anonymize(record):
    return {
        "height_cm": record.get("height_cm"),
        "weight_kg": record.get("weight_kg"),
        "age_years": record.get("age_years"),
    }

def ask_float(prompt):
    return float(input(prompt).strip().replace(",", "."))


def run_default():
    print("\n--- Simulasi Default Berdasarkan Dokumen Penelitian ---")
    h_est = pixel_to_cm(560, 500, 100)
    print(f"1. CV Tinggi Badan : {h_est:.1f} cm (MAPE vs 110 cm: {mape(110.0, h_est):.2f}%)")
    ang = spine_angle(340, 200, 300, 400)
    print(f"   Sudut tulang belakang: {ang:.1f}° -> {check_posture(ang)}")

    xs, ys = [1, 2, 3], [18.0, 19.0, 20.5]
    a, b = linear_regression(xs, ys)
    pred = predict(a, b, 4)
    status, cat, rec = early_warning(pred, SD_TABLE_DEMO)
    print(f"2. Prediksi        : y = {a:.2f} + {b:.2f}x | Bulan 4 = {pred:.2f} kg")
    print(f"   Status          : {status} | {cat}")
    print(f"   Rekomendasi     : {rec}")
    for m, v, c in forecast(a, b, xs[-1], SD_TABLE_DEMO):
        print(f"     Bulan {m}: {v:.2f} kg -> {c}")

    q = "Anak saya susah makan sayur."
    kw, intents = classify_intent(q)
    print(f"3. NLP             : {intents} (keyword: {kw})")
    print(f"   Jawaban bot     : {answer_question(q)}")
    sent, negs = analyze_sentiment("Hari ini saya lelah dan bingung.")
    print(f"   Sentimen        : {sent} (negatif: {negs})")
    print(f"   Respons         : {empathetic_response(sent)}")

    rec_d = {"name": "Budi", "photo": "child.jpg", "height_cm": 112.0, "weight_kg": 20.5, "age_years": 5}
    print(f"4. Anonimisasi     : {anonymize(rec_d)}")


def main():
    print("=" * 65)
    print("   BAIMAX: Body Analysis & Intelligent Monitor AI Expert")
    print("=" * 65)
    while True:
        print("\nPilih Menu:")
        print("1. Modul 1: Smart Growth & Posture Analysis")
        print("2. Modul 2: Predictive Analytics & Early Warning")
        print("3. Modul 3: Parenting Assistant (QnA & Sentimen)")
        print("4. Modul 4: Anonimisasi Data")
        print("5. Jalankan Semua Skenario Default")
        print("0. Keluar")
        choice = input("\nPilihan (0-5): ").strip()
        try:
            if choice == "1":
                bp = ask_float("Piksel tinggi badan anak: ")
                rp = ask_float("Piksel objek referensi: ")
                rc = ask_float("Tinggi asli objek referensi (cm): ")
                actual = ask_float("Tinggi manual/aktual (cm): ")
                est = pixel_to_cm(bp, rp, rc)
                err = mape(actual, est)
                print(f" -> Estimasi: {est:.2f} cm | MAPE: {err:.2f}% "
                      f"({'memenuhi target <5%' if err < 5 else 'di atas target 5%'})")
                sx, sy = ask_float("Shoulder X: "), ask_float("Shoulder Y: ")
                hx, hy = ask_float("Hip X: "), ask_float("Hip Y: ")
                max_a = ask_float(f"Ambang sudut (default {MAX_SPINE_ANGLE}): ")
                ang = spine_angle(sx, sy, hx, hy)
                print(f" -> Sudut: {ang:.1f}° | {check_posture(ang, max_a)}")

            elif choice == "2":
                xs = [float(x) for x in input("Bulan (pisahkan koma): ").split(",")]
                ys = [float(y) for y in input("Berat badan kg (pisahkan koma): ").split(",")]
                print("Masukkan ambang SD dari tabel Permenkes sesuai usia dan jenis kelamin:")
                sd = {k: ask_float(f"  {k} (kg): ") for k in ("-3SD", "-2SD", "+1SD", "+2SD")}
                a, b = linear_regression(xs, ys)
                print(f" -> Model: y = {a:.2f} + {b:.2f}x")
                for m, v, c in forecast(a, b, max(xs), sd):
                    print(f"    Bulan {m}: {v:.2f} kg -> {c}")
                nxt = predict(a, b, max(xs) + 1)
                st, cat, rec = early_warning(nxt, sd)
                print(f" -> Status bulan berikutnya: {st} | {cat}\n -> Rekomendasi: {rec}")

            elif choice == "3":
                q = input("Pertanyaan orang tua: ")
                kw, intents = classify_intent(q)
                print(f" -> Intent: {intents} | Keyword: {kw}")
                print(f" -> Jawaban: {answer_question(q)}")
                j = input("\nJurnal harian orang tua: ")
                sent, negs = analyze_sentiment(j)
                print(f" -> Status: {sent} | Kata negatif: {negs}")
                print(f" -> {empathetic_response(sent)}")

            elif choice == "4":
                name = input("Nama anak (sensitif): ")
                photo = input("File foto (lokal): ")
                raw = {"name": name, "photo": photo,
                       "height_cm": ask_float("Tinggi (cm): "),
                       "weight_kg": ask_float("Berat (kg): "),
                       "age_years": int(input("Usia (tahun): "))}
                print(f"Data mentah (lokal)  : {raw}")
                print(f"Data anonim ke server: {anonymize(raw)}")

            elif choice == "5":
                run_default()
            elif choice == "0":
                print("Terima kasih!")
                break
            else:
                print("Pilihan tidak valid.")
        except ValueError as e:
            print(f"Input tidak valid: {e}")


if __name__ == "__main__":
    main()
