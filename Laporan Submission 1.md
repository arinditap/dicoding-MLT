# Laporan Proyek Machine Learning Terapan - Submission 1 - Predictive Analytics
Arindita Prihastama - MLT4

## Domain Proyek

Jantung merupakan salah satu organ utama yang dimiliki manusia, yang sangat penting bagi keberlangsungan hidup manusia. Jantung harus bekerja dengan benar karena memiliki fungsi untuk memompa darah ke  seluruh tubuh sehingga oksigen dan zat-zat gizi dapat tersalurkan. Jika jantung tidak bekerja dengan benar, akan mengganggu fungsi organ tubuh lainnya. Berdasarkan data WHO, penyakit jantung merupakan salah satu penyakit yang menyebabkan kematian terbanyak di dunia sebesar 16%[1]. Di Indonesia sendiri menurut Kementerian Kesehatan Indonesia pada tahun 2017 penyakit jantung merupakan penyebab kematian tertinggi[2]. Penyakit jantung dapat disebabkan oleh tekanan darah, stress, bekerja secara berlebihan, gula darah, dan masih banyak penyebab lainnya. Dengan makin berkembangnya teknologi yang ada, dapat membantu memprediksi potensi penyakit jantung berdasarkan data-data yang berkaitan.

## Business Understanding
Penelitian ini dilakukan untuk memahami hubungan masing-masing faktor terhadap adanya kemungkinan penyakit jantung pada pasien, sehingga dapat membantu dokter dalam memberikan tindakan yang tepat serta mengurangi kemungkinan salah diagnosis.

### Problem Statements
- Apa saja faktor yang mempengaruhi adanya potensi penyakit jantung?
- Bagaimana prediksi potensi penyakit jantung berdasarkan faktor yang diberikan?

### Goals
- Mengetahui faktor-faktor yang dapat mempengaruhi adanya potensi penyakit jantung
- Memprediksi apakah seseorang memiliki penyakit jantung berdasarkan 14 faktor yang diberikan

### Solution Statements
- Nilai korelasi antarfaktor dapat digunakan untuk menentukan faktor yang mempengaruhi potensi penyakit jantung
- Melakukan komparasi beberapa model Machine Learning seperti Random Forest, Decision Tree, SVM, dan XGBoost kemudian dilakukan evaluasi model menggunakan metrik seperti akurasi, presisi, recall, dan F1-score

### Data Unserstanding
Dataset yang digunakan diambil dari _kaggle.com_ dapat dilihat pada link [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) yang disederhanakan dari dataset [UCI Heart Disease Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data). Kumpulan data ini berasal dari tahun 1988 dan terdiri dari empat database: Cleveland, Hungaria, Swiss, dan Long Beach V. Berisi 76 atribut, termasuk atribut yang diprediksi, tetapi semua percobaan yang diterbitkan mengacu pada penggunaan subset dari 14 atribut tersebut. Bidang "target" mengacu pada adanya penyakit jantung pada pasien. Bilangan bulat bernilai 0 = tidak ada penyakit dan 1 = penyakit.

### Variabel yang ada pada dataset Heart Disease Dataset adalah sebagai berikut:
- _Age_ : umur pasien dalam tahun
- _sex_ : jenis kelamin pasien
- _cp_ : tipe _chestpain_. 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic
- _trestbps_ : tekanan darah (mmHg)
- _chol_ : kolesterol
- _fbs_ : _fasting blood sugar_ atau gula darah > 120 mg/dl. 1 = True, 0 = False
- _restecg_ : hasil _resting electrocardiographic_. 0 = Normal, 1 = ST-T wave normality, 2 = Left ventricular hypertrophy
- _thalac_ : detak jantung maksimum yang dicapai
- _exang_ : _exercise-induced angina_. 1 = True, 0 = False
- _oldpeak_ : depresi
- _slope_ : slope
- _ca_ : jumlah arteri (0-3)
- _thal_ : tingkat kerusakan/defect. 0 = normal, 1 = fixed defect, 2 = reversible defect
- _target_ : variabel target

