# Laporan Proyek Machine Learning - Arkan Wijdan
---
## Domain Proyek
---
Domain proyek yang dipilih dalam proyek _machine learning_ ini adalah mengenai kesehatan dengan judul proyek "Prediksi Penyakit Stroke pada Manusia".

* #####  Latar Belakang 
    ![Image of Data dan Fakta Stroke](https://www.almazia.co/wp-content/uploads/2017/10/Data-dan-Fakta-Tentang-Stroke.jpg)
    Stroke adalah suatu penyakit cerebrovaskuler yang menjadi penyebab utama timbulnya kematian [[1]](http://dx.doi.org/10.20473/jfiki.v5i12018.36-44). Menurut WHO stroke menjadi penyebab utama morbiditas dan sebab kematian nomor dua dengan angka kematian sekitar 5,54 juta. Stroke merupakan penyebab kematian nomor tiga setelah penyakit jantung dan kanker [[2]](http://152.118.24.168/detail?id=20289574&lokasi=lokal). Di Amerika Serikat stroke sebagai penyebab kematian ketiga terbanyak setelah penyakit kardiovaskuler dan kanker. Sekitar 795.000 orang di Amerika Serikat mengalami stroke setiap tahunnya, sekitar 610.000 mengalami serangan stroke yang pertama. Stroke juga merupakan penyebab 134.000 kematian pertahun [[3]](https://www.ahajournals.org/doi/10.1161/STR.0b013e3181fcb238). 

    Penyakit Stroke di Indonesia merupakan terbanyak dan menduduki urutan pertama di Asia. Jumlah kematian yang disebabkan oleh stroke menduduki urutan kedua pada usia diatas 60 tahun dan urutan kelima pada usia 15-59 tahun. Wilayah Kalimantan Timur merupakan wilayah tertinggi pengidap penyakit stroke dengan (14,7%), diikuti Di Yogyakarta (14,3%) Bangka Belitung dan DKI Jakarta masing-masing (11,4%) dan Bali berada pada posisi 17 dengan (10,8%) [[4]](https://scholar.unand.ac.id/3595/). Penyakit stroke sering dianggap sebagai penyakit yang didominasi oleh orang tua. Dulu, stroke hanya terjadi pada usia tua mulai 60 tahun, namun sekarang mulai usia 40 tahun seseorang sudah memiliki risiko stroke, meningkatnya penderita stroke usia muda lebih disebabkan pola hidup, terutama pola makan tinggi kolesterol. Berdasarkan pengamatan di berbagai rumah sakit, justru stroke di usia produktif sering terjadi akibat kesibukan kerja yang menyebabkan seseorang jarang olahraga, kurang tidur, dan stres berat yang juga jadi faktor penyebab [[5]](http://scholar.unand.ac.id/26571/). Diperlukan kesadaran bagi setiap orang, karena masalah ini merupakan kebutuhan yang tidak hanya menyerang diusia tertentu saja. Oleh karena itu maka dibuatlah sebuah model _machine learning_ untuk memprediksi apakah seseorang terkena penyakit stroke atau tidak. Dengan adanya model _machine learning_ ini diharapkan dapat memudahkan pekerjaan dokter dalam mengindetifikasi penyakit stroke lebih awal.  

## Business Understanding
---
#### Problem Statements
berdasarkan latar belakang diatas, berikut ini rumusan masalah yang dapat diselesaikan pada proyek ini:
* Bagaimana cara melakukan pra-pemrosesan pada data penyakit stroke yang akan digunakan untuk membuat model yang baik?
* Bagaimana cara membuat model untuk memprediksi penyakit stroke pada manusia dengan menggunakan _machine learning_?
* Berapa nilai akurasi terbaik yang didapatkan dengan menggunakan _machine learning_?

#### Goals
* Melakukan pra-pemrosesan dengan baik agar dapat digunakan dalam pembuatan model.
* Mengetahui cara membuat model machine learning untuk memprediksi penyakit stroke pada manusia.
* Membuat model _machine learning_ dengan nilai akurasi yang mencapai 90%.

#### Solution Statements
Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :
* Untuk pra-pemrosesan data dapat dilakukan beberapa teknik, diantaranya :
    * Melakukan _drop_ kolom pada kolom ID.
    * Mengatasi masalah data yang kosong dengan nilai rata-rata kolom (_mean substitution_).
    * Melakukan Encoding terhadap kolom yang bertipe _object_.
    * Mengatasi masalah data tidak seimbang dengan _resample_.
    * Melakukan pembagian dataset menjadi dua bagian dengan rasio 80% untuk data latih dan 20% untuk data uji.
    * Melakukan _Standard Scaler_.

* Untuk pembuatan model dipilih penggunaan model dengan algoritma Random Forest dan K-Nearest Neighbor. Algoritma tersebut dipilih karena mudah digunakan dan juga cocok untuk kasus ini. Berikut cara kerja, kelebihan dan kekurangan algoritma Random Forest dan K-Nearest Neighbor:
    * Cara kerja Algoritma Random Forest [[6]](https://repository.usd.ac.id/35513/):
        * Diawali dengan pemilihan k pada sampel dataset yang diambil secara acak dengan pengembalian
        * Gunakan dataset untuk membangun _decision tree_ ke-i
        * Ulangi langkah kedua langkah diatas sebanyak k.
    * Kelebihan dan kekurangan Algoritma Random Forest [[7]](https://eprints.umm.ac.id/39299/):
        * Kelebihannya yaitu dapat mengatasi _noise_ dan _missing value_ serta dapat mengatasi data dalam jumlah yang besar.
        * Kekurangan pada algoritma Random Forest yaitu interpretasi yang sulit dan membutuhkan tuning model yang tepat untuk data. 
    * Cara kerja Algoritma K-Nearest Neighbor [[8]](https://publikasi.dinus.ac.id/index.php/jais/article/view/1189/):
        * Menentukan jumlah tetangga terdekat K
        * Menghitung jarak dokumen _testing_ ke dokumen _training_
        * Urutkan data berdasarkan data yang mempunyai jarak Euclidean terkecil
        * Tentukan kelompok testing berdasarkan label pada K.
    * Kelebihan dan kekurangan Algoritma K-Nearest Neighbor [[9]](https://simdos.unud.ac.id/uploads/file_penelitian_1_dir/721bdb509a6f0bb9ccca6d7374b86759.pdf):
        * KNN memiliki beberapa kelebihan yaitu bahwa algoritmanya tangguh terhadap _training_ data yang _noisy_ dan efektif apabila data latihnya besar.
        * Kekurangan pada algoritma KKN yaitu perlu menentukan nilai dari parameter K (jumlah dari tetangga terdekat), Pembelajaran berdasarkan jarak tidak jelas mengenai jenis jarak apa yang harus digunakan dan atribut mana yang harus digunakan untuk mendapatkan hasil yang terbaik dan Biaya komputasi cukup tinggi karena diperlukan perhitungan dari jarak tiap sample uji pada keseluruhan sample latih.

## Data Understanding
![Image of Dataset](https://i.postimg.cc/X7y94ssJ/Capture.png)
Informasi dataset dapat dilihat pada tabel dibawah ini :
Jenis | Keterangan
--- | ---
Sumber | [Kaggle Dataset : Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)
Lisensi | Data files © Original Authors
Kategori | Kesehatan, Kondisi Kesehatan, Kesehatan Masyarakat
Jenis dan Ukuran Berkas | CSV (316.97 kB)
---
Pada berkas yang diunduh yakni healthcare-dataset-stroke-data.csv berisi 5110 baris dan 12 kolom. Kolom-kolom tersebut terdiri dari 5 buah kolom bertipe objek dan 7 buah kolom bertipe numerik (tipe data int64). Terdapat juga kolom yang memiliki data kosong yaitu pada kolom bmi. Untuk penjelasan mengenai variabel-variable pada dataset stroke ini dapat dilihat sebagai berikut:
* **id** merupakan parameter bernilai unique. Parameter ini tidak penting untuk dimasukkan kedalam model, oleh karena itu parameter ini di _drop_.
* **gender** merupakan parameter untuk mengetahui jenis kelamin. Terdapat 3 nilai yaitu _male_, _female_, dan _other_.
* **age** merupakan parameter untuk mengetahui umur. Terdapat Pada data ini nilainya berada pada rentang 0.080-82 tahun.
* **hypertension** merupakan parameter yang menyatakan apakah pasien memiliki darah tinggi atau tidak. Nilai 0 menyatakan bahwa pasien tidak memiliki darah tinggi dan Nilai 1 menyatakan bahwa pasien memiliki darah tinggi.
* **heart_disease** merupakan parameter yang menyatakan apakah pasien memiliki penyakit jantung atau tidak. Nilai 0 menyatakan bahwa pasien tidak memiliki penyakit jantung dan Nilai 1 menyatakan bahwa pasien memiliki penyakit jantung.
* **ever_married** merupakan parameter yang menyatakan apakah pasien pernah menikah atau tidak. Nilai "_Yes_" menyatakan bahwa pasien pernah menikah dan Nilai "_No_" menyatakan bahwa pasien belum pernah menikah.
* **work_type** merupakan parameter yang menyatakan pekerja pasien. Pada data ini terdapat 5 nilai yaitu "_children_", "_Govt_jov_", "_Never_worked_", "_Private_" dan "_Self-employed_".
* **Residence_type** merupakan parameter yang menyatakan tipe tempat tinggal pasien. Pada data ini terdapat 2 nilai yaitu "_Rural_" dan "_Urban_".
* **avg_glucose_level** merupakan parameter yang menyatakan kadar glukosa rata-rata dalam darah pasien. Pada data ini nilainya berada di rentang 55.12-271.74 mg/dL.
* **bmi** merupakan parameter yang menyatakan kadar glukosa rata-rata dalam darah pasien. Pada data ini nilainya berada di rentang 55.12-271.74.
* **smoking_status** merupakan parameter yang menyatakan status merokok pada pasien. Pada data ini terdapat 4 nilai yaitu "_formerly smoked_", "_never smoked_", "_smokes_" or "_Unknown_".
* **stroke** merupakan parameter yang Menentukan apakah pasien menderita stroke atau tidak. Terdapat 2 nilai yaitu tidak menderita stroke (nilai 0) dan menderita stroke (nilai 1).

Selain itu, terdapat juga visualisasi data pada tiap kolom yang dibagi menjadi 2 tipe seperti berikut:
* Kategorial:
    ![Image of Dataset](https://i.postimg.cc/gLPggRJx/download.png) ![Image of Dataset](https://i.postimg.cc/WdpHRM7r/download-1.png) ![Image of Dataset](https://i.postimg.cc/mcJXFKWN/download-3.png) 
    ![Image of Dataset](https://i.postimg.cc/NKNPQ5Nn/download-2.png) ![Image of Dataset](https://i.postimg.cc/xJmsGg0r/download-4.png)
* Numerik:
    ![Image of Dataset](https://i.postimg.cc/rpcm0Kmr/numerik.png)
    
## Data Preparation
---
Berikut adalah tahapan-tahapan dalam melakukan pra-pemrosesan data:
  * Men-drop kolom ID
  ![Image coloumn ID](https://i.postimg.cc/5290ykBG/id.png)
    Men-drop kolom ID dilakukan karena memiliki nilai korelasi yang kecil dengan kolom target yaitu _stroke_.
  * Mengisi data yang kosong dengan nilai rata rata atau (mean substitution)
  ![Image of Dataset](https://i.postimg.cc/t46yLqm7/missing.png)
    Pada dataset ini terdapat satu kolom yang memiliki nilai _missing value_ yaitu pada kolom bmi. Metode yang digunakan untuk menangani masalah ini adalah dengan mengisi data yang kosong dengan nilai rata-rata kolomnya.Proses yang dilakukan yaitu pertama-tama mengambil nilai rata-rata dari kolom bmi, kemudian memasukannya kepada setiap data kosong sebagai pengganti dari datanya.  
  * Melakukan Encoding terhadap kolom yang bertipe object
  ![Image of Dataset](https://i.postimg.cc/t46yLqm7/missing.png)
    Data bertipe object tidak dapat diproses dalam machine learning, maka dari itu dalam harus diubah dalam bentuk numerik. AAda beberapa cara melakukan encoding categorical data dengan melakukan _label encoding_ dan _one hot encoding_. _Label encoding_ mengubah setiap nilai dalam kolom menjadi angka yang berurutan, contoh pada kolom _ever_married_mapping_, mengubah nilai "_No_" menjadi 0 dan "_Yes_" menjadi 1. Berikut isi dari kolom _ever_married_mapping_ setelah dilakukan _Label encoding_:
   ![Image of Dataset](https://i.postimg.cc/6QnnLmP0/ever.png)
    _One hot encoding_ adalah teknik yang merubah setiap nilai di dalam kolom menjadi kolom baru dan mengisinya dengan nilai biner yaitu 0 dan 1. Contoh pada kolom _gender_ yang memiliki 3 nilai yaitu "_Female_", "_Male_", dan "_Other_". Maka setelah dilakukan teknik _One hot encoding_ akan terbentuk kolom yang baru yaitu seperti pada gambar dibawah ini: 
   ![Image of Dataset](https://i.postimg.cc/nLhGKJ7t/one-hot.png)
  * Mengatasi data target yang tidak seimbang jumlahnya menggunakan teknik resample
    Pada dataset ini mengalami _imbalance_ yang menyebabkan model yang dibuat akan menjadi bias terhadap data target yang memiliki data yang banyak. Oleh karena itu diperlukan teknik untuk manipulasi data, teknik yang digunakan adalah teknik _resample_. Pertama dengan memilih data target yang memiliki data yang dikit. Lalu tentukan _n_samples_ sesuai dengan data target yang memiliki data yang banyak. Lalu, fungsi resample akan menghasilkan data baru dari data yang sudah ada sampai jumlah datanya sama dengan data target yang memiliki data yang banyak. Terakhir gabungkan dengan datasetnya.
  * Melakukan pembagian dataset menjadi dengan 80% untuk data latih dan 20% untuk data uji
  Setelah melakukan pra-pemrosesan ke dataset, selanjutnya adalah membagi dataset untuk data latih dan data uji dengan rasio 80:20. Data latih adalah data yang hanya untuk melatih model, sedangkan data uji adalah data yang hanya sebagai ujicoba model. Pembagian dataset ini menggunakan modul train_test_split dari scikit-learn.
  * Melakukan standardisasi data pada semua fitur data.
    Tahap terakhir yaitu melakukan standarisasi data. Hal ini dilakukan untuk membuat semua fitur berada dalam skala data yang sama yaitu dengan range 0-1. Strandadisasi data ini menggunakan fungsi StandardScaler yang perhitungannya seperti dibawah ini:
    ![Image of Dataset](https://i.stack.imgur.com/obywE.png)

## Modeling
---
Setelah dilakukan pra-pemrosesan pada dataset, langkah selanjutnya adalah _modeling_ terhadap data. Pada tahap ini menggunakan 2 algoritma yaitu Random Forest dan K-Nearest Neighbor dengan tanpa parameter tambahan. Pertama-tama kedua model ini dilatih menggunakan data latih. Setelah itu kedua model akan diuji dengan data uji. Terakhir kedua model akan diukur nilai akurasinya.Perbandingan Hasil dari kedua model sebagai berikut:
    ![Image of perbandingan modelling](https://i.postimg.cc/SRdcx8cC/perbandingan-hasil-model.png)


Pada model dengan algoritma Random Forest memiliki nilai akurasi, _f1-score_, _recall_ dan _precision_ lebih tinggi dibanding dengan algoritma K-Nearest Neighbor. Untuk membuktikannya, kedua model tersebut diuji pada data uji dan di visualisasikan pada confussion matrix seperti berikut. 
* _Confussion Matrix_ algoritma Random Forest:
    ![Image of Dataset](https://i.postimg.cc/YvDx0W5d/cf-random-forest.png)
* _Confussion Matrix_ algoritma K-Nearest Neighbor:
    ![Image of Dataset](https://i.postimg.cc/MnpDzDFc/cf-knn.png)

Dengan hasil diatas, maka model dengan algoritma Random Forest merupakan model yang dipilih untuk digunakan.

## Evaluation
---
Pada proyek ini, model yang dikembangkan adalah kasus klasifikasi dan menggunakan metriks akurasi, _f1-score_, _recall_ dan _precision_. Berikut hasil pengukuran model yang dipilih yaitu model yang menggunakan algoritma Random Forest metriks akurasi, _f1-score_, _recall_ dan _precision_.
![Image of Dataset](https://i.postimg.cc/1tTfCXmD/hasil-random-forest.png)
* Akurasi
    Akurasi merupakan metrik untuk menghitung persentase dari total data yang diidentifikasi dan dinilai benar. Rumus akurasi sebagai berikut:
    ![Image of Dataset](https://i.postimg.cc/NFx1VcgJ/akurasi.png)
    * _True Positive_ (TP) :
    Kasus dimana model memprediksi nilai 0 dan jawaban yang benar juga nilai 0.
    * _True Negative_ (TN):
    Kasus dimana model memprediksi nilai 0 tetapi jawaban yang benar adalah nilai 1.
    * _False Positive_ (FP) :
    Kasus dimana model memprediksi nilai 1 dan jawaban yang benar juga nilai 1.
    * _False Negative_ (FN):
    Kasus dimana model memprediksi nilai 1 tetapi jawaban yang benar adalah nilai 0.
* _Precision_
    _Precision_ merupakan metrik untuk memprediksi benar positif dari keseluruhan hasil yang diprediksi positf. Rumus _precision_ sebagai berikut:
    ![Image of Dataset](https://i.postimg.cc/mzwZLjdM/precision.png)
* _Recall_
    _Recall_ merupakan metrik untuk memprediksi benar positif dibandingkan dengan keseluruhan data yang benar positif. Rumus _precision_ sebagai berikut:
    ![Image of Dataset](https://i.postimg.cc/K38GRTVW/recall.png)
* _f1-score_
    _f1-score_ merupakan metrik untuk perbandingan rata-rata precision dan recall yang dibobotkan. Rumus _f1-score_ sebagai berikut:
    ![Image of Dataset](https://i.postimg.cc/Fzm9ztjQ/f1-score.png)

## Referensi
--- 
[[1]](http://dx.doi.org/10.20473/jfiki.v5i12018.36-44) Handayani, D., & Dominica, D. _GAMBARAN DRUG RELATED PROBLEMS (DRP’S) PADA PENATALAKSANAAN PASIEN STROKE HEMORAGIK DAN STROKE NON HEMORAGIK DI RSUD DR M YUNUS BENGKULU_. Vol 5, No 1 (2018). http://dx.doi.org/10.20473/jfiki.v5i12018.36-44.
[[2]](http://152.118.24.168/detail?id=20289574&lokasi=lokal) Nastiti, D. _GGambaran faktor risiko kejadian stroke pada pasien stroke rawat inap di rumah sakit Krakatau Medika tahun 2011_. Depok: Universitas Indonesia (2012). http://152.118.24.168/detail?id=20289574&lokasi=lokal
[[3]](https://www.ahajournals.org/doi/10.1161/STR.0b013e3181fcb238) Goldstein, L. B., dkk. _Guidelines for the primary prevention of stroke_. Stroke, 42(2), 517–584 (2011). https://doi.org/10.1161/str.0b013e3181fcb238 
[[4]](https://scholar.unand.ac.id/3595/) Roberta, C. _HUBUNGAN HIPERGLIKEMIA DENGAN KELUARAN PASIEN STROKE ISKEMIK DAN HEMORAGIK DI RSUP DR. M. DJAMIL PADANG_. Diploma thesis, Universitas Andalas (2016). https://scholar.unand.ac.id/3595/
[[5]](http://scholar.unand.ac.id/26571/) Yuza, K. _FAKTOR RISIKO YANG BERHUBUNGAN DENGAN KEJADIAN STROKE PADA USIA MUDA DI RUMAH SAKIT STROKE NASIONAL BUKITTINGI_. Diploma thesis, Universitas Andalas (2017). http://scholar.unand.ac.id/26571/
[[6]](https://repository.usd.ac.id/35513/) Haristu, R. A. _PENERAPAN METODE RANDOM FOREST UNTUK PREDIKSI WIN RATIO PEMAIN PLAYER UNKNOWN BATTLEGROUND_. Skripsi thesis, Universitas Sanata Dharma (2019). https://repository.usd.ac.id/35513/ https://repository.usd.ac.id/35513/2/155314090_full.pdf
[[7]](https://repository.usd.ac.id/35513/) Rizqi, M. S. _KLASIFIKASI PENYAKIT DIABETES MELLITUS DENGAN MENGGUNAKAN PERBANDINGAN ALGORITMA J48 DAN RANDOM FOREST (STUDI KASUS : RUMAH SAKIT MUHAMMADIYAH LAMONGAN)_. Undergraduate (S1) thesi, Universitas Muhammadiyah Malang (2018). https://repository.usd.ac.id/35513/
[[8]](https://publikasi.dinus.ac.id/index.php/jais/article/download/1189/893/) Sani, M. S., Zeniarja, J., & Luthfiarta, A. _Penerapan Algoritma K-Nearest Neighbor pada Information Retrieval dalam Penentuan Topik Referensi Tugas Akhir_. UVol 1, No 2 (2016). https://publikasi.dinus.ac.id/index.php/jais/article/download/1189/893/
[[9]](https://simdos.unud.ac.id/uploads/file_penelitian_1_dir/721bdb509a6f0bb9ccca6d7374b86759.pdf) Penyelenggara PS. Teknik Informa ka, Jurusan Ilmu Komputer FMIPA - Universitas Udayana Kampus Bukit Jimbaran. _PROSIDING ISSN : X SEMINAR NASIONAL TEKNOLOGI INFORMASI & APLIKASINYA 2015 INOVASI TEKNOLOGI INFORMASI DAN KOMUNIKASI DALAM MENUNJANG TECHNOPRENEURSHIP_. Universitas Udayana (2015). https://simdos.unud.ac.id/uploads/file_penelitian_1_dir/721bdb509a6f0bb9ccca6d7374b86759.pdf
---Ini adalah bagian akhir laporan---