# Diabetes Classification Project

Nama: Evan Arlen Handy

Username dicoding: warlord194

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Diabetes prediction dataset](https://www.kaggle.com/) |
| Masalah | Diabetes Melitus (DM) adalah penyakit kronis yang ditandai dengan kadar glukosa darah yang melebihi batas normal, yakni gula darah sewaktu lebih dari 200 mg/dl atau gula darah puasa di atas 126 mg/dl. DM dikenal sebagai "silent killer" karena sering kali tidak disadari oleh penderita hingga menyebabkan komplikasi serius. Menurut International Diabetes Federation (IDF), prevalensi diabetes mellitus di dunia adalah 1,9%, dengan 95% di antaranya adalah DM tipe 2. Pada tahun 2013, diperkirakan 382 juta orang di dunia mengidap diabetes. |
| Solusi machine learning | Dengan menggunakan model klasifikasi diabetes berbasis machine learning, diharapkan dapat memberikan informasi yang lebih akurat kepada dokter untuk membantu diagnosis diabetes dan merencanakan pemeriksaan lanjutan terhadap pasien. |
| Metode pengolahan | Data terdiri dari sembilan fitur, dengan delapan fitur digunakan untuk klasifikasi dan satu sebagai target. Terdapat dua fitur kategorikal dan enam numerikal dalam dataset. Data dibagi dengan proporsi 80:20 untuk data latih dan evaluasi. Beberapa transformasi dilakukan seperti mengganti nama fitur yang diubah dan menggunakan one-hot encoding pada target. |
| Arsitektur model | Model ini terdiri dari tiga lapisan Dense dengan ukuran 256, 64, dan 16 neuron serta fungsi aktivasi ReLU. Lapisan terakhir menggunakan Dense 1 dengan fungsi aktivasi sigmoid untuk mengklasifikasikan dua kelas (diabetes dan bukan diabetes). Model dikompilasi dengan optimizer Adam, learning rate 0.001, loss binary_crossentropy, dan metrik BinaryAccuracy. |
| Metrik evaluasi | Model dievaluasi menggunakan metrik AUC, Precision, Recall, ExampleCount, dan BinaryAccuracy untuk menilai kualitas klasifikasi yang dihasilkan. |
| Performa model | Model ini menunjukkan akurasi yang tinggi, yaitu 0.9800 pada data latih dan validasi. Nilai loss yang tercatat pada proses training dan validasi adalah 0.0712, menandakan performa model yang sangat baik dalam klasifikasi. |
| Opsi deployment | Sistem ini akan dideploy melalui platform Railway untuk menyediakan akses mudah kepada pengguna dalam menggunakan model. |
| Web app | [diabetes-classification-project](https://diabetes-ops-production.up.railway.app/v1/models/diabetes-classification-model/metadata) |
| Monitoring | Pemantauan dilakukan menggunakan metric tensorflow:core:saved_model:read:count pada platform Prometheus dan Grafana untuk melacak proses pembacaan model yang disimpan.|
