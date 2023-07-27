# Laporan Proyek Machine Learning Terapan - Submission 1 - Predictive Analytics
Arindita Prihastama - MLT4

## "_Prediksi Penyakit Jantung - Heart Attack Prediction_"

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
- Melakukan komparasi beberapa model *Machine Learning* seperti *Random Forest*, *Decision Tree*, *Support Vector Machine*, dan *XGBoost* kemudian dilakukan evaluasi model menggunakan metrik seperti akurasi, presisi, recall, dan F1-score

## Data Understanding
Dataset yang digunakan diambil dari _kaggle.com_ dapat dilihat pada link *[Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)* yang disederhanakan dari dataset *[UCI Heart Disease Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)*. Kumpulan data ini berasal dari tahun 1988 dan terdiri dari empat database: Cleveland, Hungaria, Swiss, dan Long Beach V. Berisi 76 atribut, termasuk atribut yang diprediksi, tetapi semua percobaan yang diterbitkan mengacu pada penggunaan subset dari 14 atribut tersebut. Fitur "target" mengacu pada adanya penyakit jantung pada pasien. Bilangan bulat bernilai 0 = tidak ada penyakit dan 1 = penyakit.

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

Pada tahap ini dilakukan beberapa langkah:
1. Memeriksa informasi terkait dataset
    ```sh
    RangeIndex: 1025 entries, 0 to 1024
    Data columns (total 14 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       1025 non-null   int64  
     1   sex       1025 non-null   int64  
     2   cp        1025 non-null   int64  
     3   trestbps  1025 non-null   int64  
     4   chol      1025 non-null   int64  
     5   fbs       1025 non-null   int64  
     6   restecg   1025 non-null   int64  
     7   thalach   1025 non-null   int64  
     8   exang     1025 non-null   int64  
     9   oldpeak   1025 non-null   float64
     10  slope     1025 non-null   int64  
     11  ca        1025 non-null   int64  
     12  thal      1025 non-null   int64  
     13  target    1025 non-null   int64  
    dtypes: float64(1), int64(13)
    ```
    Dari output yang ada dapat diketahui bahwa:
    - Jumlah data: 1025
    - Jumlah atribut: 14. 1 atribut merupakan variabel target.
    - 13 atribut memiliki tipe data numerik integer dan satu atribut dengan tipe data numerik float

2. Memeriksa data kosong atau *missing values*
   ```sh
   age         0
   sex         0
   cp          0
   trestbps    0
   chol        0
   fbs         0
   restecg     0
   thalach     0
   exang       0
   oldpeak     0
   slope       0
   ca          0
   thal        0
   target      0
    dtype: int64
   ```
    Output menunjukkan bahwa tidak terdapat data yang kosong.

3. Memisahkan categorical features dan continuous features
    - Yang termasuk kolom kategorial adalah :  ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    - Yang termasuk kolom kontinyu adalah :  ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    - Yang termasuk kolom target adalah :  ['target']

## Exploratory Data Analysis (EDA)
Pada tahapan ini dilakukan visualisasi data
```sh 
    	          count	      mean	      std	    min	  25%	  50%	  75%	  max
    age       1025.0	  54.434146	  9.072290	29.0	48.0	56.0	61.0	77.0
    trestbps  1025.0	131.611707	17.516718	94.0	120.0	130.0	140.0	200.0
    chol      1025.0	246.000000	51.592510	126.0	211.0	240.0	275.0	564.0
    thalach   1025.0	149.114146	23.005724	71.0	132.0	152.0	166.0	202.0
    oldpeak    1025.0	1.071512	  1.175053	0.0	  0.0	  0.8	  1.8	  6.2
```

Dari data continuous features diatas, dapat disimpulkan:`age` ada pada rentang 29-77 tahun, dengan rata-rata 54-55 tahun; `trestbps` ada pada rentang 94-200 mmHg dengan rata-rata 131-132 mmHg; `chol` antara 126-564 mmHg, rata-rata 246 mmHg; `thalach` pada rentang 71-202, rata-rata 149-150; `oldpeak` pada rentang 0-6, rata-rata 1.

#### Distribusi Fitur Categorical terhadap Variabel Target
![distribution_cat_ft](https://github.com/arinditap/dicoding-MLT1-heart-attack-pred/assets/48308725/0e8bae9f-853f-4136-a102-ba0aa4b4b0cb)
- Orang dengan Chestpain tipe *Non-Anginal Pain* memiliki potensi terkena penyakit jantung lebih besar
- Orang dengan pembuluh darah besar = 0 berpotensi terkena penyakit jantung lebih besar
- Laki-laki berpotensi terkena penyakit jantung lebih besar
- Orang dengan thalium stress = 2 berpotensi lebih besar terkena penyakit jantung
- Orang dengan *exercise-induced-angina* = 0 berpotensi terkena penyakit jantung lebih besar

#### Distribusi Fitur Continuous terhadap Variabel Target
![distribution_cont_ft](https://github.com/arinditap/dicoding-MLT1-heart-attack-pred/assets/48308725/8e453e99-db7a-4b7e-8324-8bb20521ad6b)

#### Correlation Matrix antarfitur
![correlation_matrix](https://github.com/arinditap/dicoding-MLT1-heart-attack-pred/assets/48308725/bda8f26f-9cf7-4436-94e5-128a80807fa5)
- Faktor yang memiliki korelasi tinggi adalah 'cp', 'thalach', dan 'slope'
- Faktor yang memiliki korelasi rendah adalah 'restecg', dan 'fbs'

## Data Preprocessing
Data yang digunakan dapat dikategorikan sebagai imbalance data. Oleh karena itu akan dilakukan data preprocessing berupa pembagian data menggunakan Train-Test Split dan standarisasi menggunakan StandardScaler. Tahapannya adalah sebagai berikut:
1. Membuang fitur dengan korelasi rendah.
2. Karena sebagian data belum berbentuk numerik maka dilakukan encoding pada fitur-fitur categorical.
3. Melakukan pembagian dataset menjadi data latih (train) dan data uji (test) menggunakan modul `train_test_split` dengan rasio 80:20. Dari total sampel seluruh dataset yang berjumlah 1025 didapatkan sampel pada train dataset sejumlah 820 dan sampel pada test dataset 205.
4. Melakukan standarisasi data menggunakan StandardScaler
   ```sh
            Hasil Standarisasi Fitur Continuous
   	        age	    trestbps	  chol	   thalach	oldpeak
    count	820.0000	820.0000	820.0000	820.0000	820.0000
    mean	-0.0000	  0.0000	  -0.0000	  0.0000	  -0.0000
    std	  1.0006	  1.0006	  1.0006	  1.0006	  1.0006
    min	  -2.7689	  -2.1425	  -2.3984	  -3.4194	  -0.9122
    25%	  -0.8041	  -0.6659	  -0.6860	  -0.7572	  -0.9122
    50%	  0.1782	  -0.0979	  -0.1018	  0.1156	  -0.2107
    75%	  0.7240	  0.4700	  0.5832	  0.7375	  0.5347
    max	  2.4705	  3.8776	  6.4255	  2.2977	  4.5244
   ```
## Modelling
Tahapan ini menerapkan algoritma Machine Learning untuk menjawab permasalahan yang dijabarkan pada problem statements diatas. Algoritma ML yang akan diterapkan antara lain:
1. Random Forest
   Algoritma ini digunakan untuk pengklasifikasian data set yang memiliki jumlah besar, sehingga bisa digunakan untuk banyak dimensi dengan berbagai skala dan performa yang tinggi. Klasifikasi ini               dilakukan melalui penggabungan tree dalam decision tree dengan cara training dataset yang Anda miliki[3].
2. Decision Tree
   Algoritma ini menggunakan seperangkat aturan untuk membuat keputusan dengan struktur seperti pohon yang memodelkan kemungkinan hasil, biaya sumber daya, utilitas dan kemungkinan konsekuensi atau resiko.      Konsepnya dengan menyajikan algoritma dengan pernyataan bersyarat, meliputi cabang untuk mewakili langkah-langkah pengambilan keputusan yang dapat mengarah pada hasil yang menguntungkan[3].
   Random Forest dan Decision Tree cocok untuk melakukan klasifikasi pada data penyakit jantung karena model ini dapat memetakan pola dan hubungan antara variabel independen dengan variabel target, serta        dapat memecah data menjadi subset yang lebih kecil untuk meningkatkan akurasi model.
3. Support Vector Machine (SVM)
   Algoritma ini termasuk dalam kategori Supervised Learning yang digunakan untuk menyelesaikan permasalahan klasifikasi dan regresi. SVM bertujuan untuk menetapkan hyperplane terbaik dalam ruang berdimensi-    N (ruang dengan N-jumlah fitur) yang berfungsi sebagai pemisah yang jelas bagi titik-titik data input. SVM adalah model yang cocok untuk digunakan  karena model ini dapat mengatasi masalah klasifikasi        yang kompleks dan memiliki kemampuan untuk menemukan hyperlane yang lebih baik dalam data yang kompleks.
5. Extreme Gradient Boosting (XGBoost)
   Algoritma ini mengintegrasikan beberapa model pohon untuk membangun model pembelajaran yang lebih kuat. Selain itu, XGBoost dicirikan oleh kemampuannya untuk menggunakan multithreading CPU secara otomatis    untuk komputasi paralel, yang dapat mempercepat penghitungan[5]. Algoritma XGBoost dapat mengatasi data medis yang kompleks dan beragam, dan dapat memenuhi persyaratan ketepatan waktu dan akurasi             diagnosis tambahan dengan lebih baik[]. XGBoost cocok untuk digunakan karena model ini dapat mengatasi masalah overfitting dan memiliki kemampuan untuk meningkatkan performa model dengan menambahkan lebih    banyak estimator pada model.

## Evaluasi Model
Metrik evaluasi yang digunakan adalah:
1. Akurasi (Accuracy) -> nilai keakuratan model dalam melakukan klasifikasi dengan benar.
   ```sh
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```
2. Presisi (Precision) -> nilai sensitifitas atau nilai ketepatan antara data yang diharapkan dengan data hasil prediksi model.
   ```sh
   Precision = TP / (TP + FP)
   ```
3. Recall -> nilai yang menunjukan tingkat keberhasilan atau spesifisitas untuk mengetahui kembali sebuah informasi secara benar.
   ```sh
   Recall = TP / (TP + FN)
   ```
4. f1-Score -> perbandingan rata-rata presisi dan recall yang dibobotkan
   ```sh
   F1-Score = (2 * Precision * Recall) / (Precision + Recall)
   ```
Hasil evaluasi model:
```sh
           Model    Akurasi  Precision    Recall  F1-score
0  Random Forest  87.317073   0.813008  0.970874  0.884956
1  Decision Tree  91.707317   0.947917  0.883495  0.914573
2            SVM  86.829268   0.839286  0.912621  0.874419
3        XGBoost  92.195122   0.906542  0.941748  0.923810
```
Mencari fitur yang paling berpengaruh dalam prediksi menggunakan XGBCLassifier
![feature_importance](https://github.com/arinditap/dicoding-MLT1-heart-attack-pred/assets/48308725/91bc62b1-b9ed-409d-84cd-dc57bb89587a)

Fitur yang paling berpengaruh dalam prediksi penyakit jantung adalah `age` dan `chol`.

## Kesimpulan
- Faktor yang paling berpengaruh dalam prediksi adanya penyakit jantung adalah faktor `age` atau usia dan `chol` atau tingkat kolesterol.
- Berdasarkan evaluasi model, disimpulkan bahwa XGBoost memiliki performa terbaik dengan akurasi 92.19%. Selain itu precision dan recall dari XGBoost juga memiliki nilai terbaik yaitu 0.90 dan 0.94 yang mana ini menunjukkan bahwa model ini mampu memprediksi dengan akurat orang yang memiliki penyakit jantung dengan yang tidak.

## Referensi
1. The top 10 causes of death,” who.int, 2020.
2. "Kementerian Kesehatan Republik Indonesia," 2017. [Online]. Available: http://www.depkes.go.id/article/view/17073100005/penyakit-jantung-penyebab-kematian-tertinggi-kemenkes-ingatkan-cerdik-.html.
3. Rajdhan, A., Agarwal, A., Sai, M., Ravi, D., & Ghuli, P. (2020). Heart disease prediction using machine learning. INTERNATIONAL JOURNAL OF ENGINEERINGRESEARCH & TECHNOLOGY (IJERT), 9(O4).
4. Li, S., & Zhang, X. (2020). Research on orthopedic auxiliary classification and prediction model based on XGBoost algorithm. Neural Computing and Applications, 32, 1971-1979.
5. Ramraj, S., Uzir, N., Sunil, R., & Banerjee, S. (2016). Experimenting XGBoost algorithm for prediction and classification of different datasets. International Journal of Control Theory and Applications, 9(40), 651-662.
6. Li, W., Yin, Y., Quan, X., & Zhang, H. (2019). Gene expression value prediction based on XGBoost algorithm. Frontiers in genetics, 10, 1077.
7. Givari, M. R., Sulaeman, M. R., & Umaidah, Y. (2022). Perbandingan Algoritma SVM, Random Forest Dan XGBoost Untuk Penentuan Persetujuan Pengajuan Kredit. Nuansa Informatika, 16(1), 141-149.
8. Al Azhima, S. A. T., Darmawan, D., Hakim, N. F. A., Kustiawan, I., Al Qibtiya, M., & Syafei, N. S. (2022). Hybrid Machine Learning Model untuk memprediksi Penyakit Jantung dengan Metode Logistic Regression dan Random Forest. Jurnal Teknologi Terpadu, 8(1), 40-46.
9. Azhari, M., Situmorang, Z., & Rosnelly, R. (2021). Perbandingan Akurasi, Recall, dan Presisi Klasifikasi pada Algoritma C4. 5, Random Forest, SVM dan Naive Bayes. Jurnal Media Informatika Budidarma, 5(2), 640-651.
   




 

