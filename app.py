# app.py
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 1) โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv("wongnai_reviews_train.csv")

# 2) สร้าง TF-IDF Vectorizer และ fit กับ review_body ทั้งหมด
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["review_body"])

# 3) ฟังก์ชันสำหรับทำ matching
def match_reviews(user_text, user_rating, top_n=5):
    """
    รับข้อความจากผู้ใช้ (user_text) และ star_rating
    แล้ว return แถวใน DataFrame ที่คล้ายคลึงที่สุด (top_n) 
    โดยกรองเฉพาะแถวที่มี star_rating ตรงกับ user_rating
    """
    # แปลง user_text ให้เป็น vector
    user_vec = vectorizer.transform([user_text])
    
    # กรอง DataFrame เฉพาะแถวที่มี star_rating ตรงกับที่ผู้ใช้กำหนด
    filtered_indices = df.index[df["star_rating"] == int(user_rating)].tolist()
    if not filtered_indices:
        return []
    
    # ดึงเฉพาะ tfidf ของแถวที่กรองไว้
    filtered_tfidf = tfidf_matrix[filtered_indices]
    
    # คำนวณ cosine similarity ระหว่าง user_vec กับแถวที่กรอง
    sims = cosine_similarity(user_vec, filtered_tfidf).flatten()
    
    # เลือก index ของแถวที่มีค่าความคล้ายสูงสุด top_n
    top_indices = sims.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        real_index = filtered_indices[idx]
        similarity_score = sims[idx]
        results.append({
            "review_body": df.loc[real_index, "review_body"],
            "star_rating": int(df.loc[real_index, "star_rating"]),
            "similarity": float(similarity_score)
        })
    
    return results

# 4) หน้า Home: แสดงฟอร์ม HTML สำหรับผู้ใช้กรอกข้อมูล
@app.route("/", methods=["GET"])
def home():
    html_form = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Wongnai Reviews Matching</title>
    </head>
    <body>
        <h1>Wongnai Reviews Matching</h1>
        <form action="/match" method="post">
            <label for="review_body">Review Body:</label><br>
            <textarea id="review_body" name="review_body" rows="5" cols="50" placeholder="Enter review text"></textarea><br><br>
            <label for="star_rating">Star Rating (1-5):</label><br>
            <input type="number" id="star_rating" name="star_rating" min="1" max="5" value="3"><br><br>
            <input type="submit" value="Match Reviews">
        </form>
    </body>
    </html>
    """
    return render_template_string(html_form)

# 5) Endpoint /match สำหรับรับข้อมูลจาก JSON หรือ form HTML
@app.route("/match", methods=["POST"])
def match_endpoint():
    # ตรวจสอบว่าข้อมูลถูกส่งมาเป็น JSON หรือ form data
    if request.is_json:
        data = request.get_json()
        user_text = data.get("review_body", "")
        user_rating = data.get("star_rating", None)
    else:
        user_text = request.form.get("review_body", "")
        user_rating = request.form.get("star_rating", None)
    
    if not user_text or user_rating is None:
        return jsonify({"error": "Please provide both review_body and star_rating"}), 400
    
    matched_results = match_reviews(user_text, user_rating, top_n=5)
    
    # หากเป็นการส่งมาจากฟอร์ม HTML ให้แสดงผลในหน้า HTML
    if not request.is_json:
        # สร้าง HTML สำหรับแสดงผล
        html_result = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Matching Results</title>
        </head>
        <body>
            <h1>Matching Results</h1>
            {% if results %}
                <ul>
                {% for item in results %}
                    <li>
                        <strong>Review:</strong> {{ item.review_body }}<br>
                        <strong>Star Rating:</strong> {{ item.star_rating }}<br>
                        <strong>Similarity:</strong> {{ item.similarity | round(4) }}
                    </li>
                {% endfor %}
                </ul>
            {% else %}
                <p>No matching reviews found.</p>
            {% endif %}
            <a href="/">Back to Home</a>
        </body>
        </html>
        """
        # ใช้ render_template_string เพื่อเรนเดอร์ผลลัพธ์
        return render_template_string(html_result, results=matched_results)
    
    # หากเป็น JSON ก็ส่งกลับ JSON
    return jsonify({"results": matched_results})

# 6) รันเซิร์ฟเวอร์ Flask
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
