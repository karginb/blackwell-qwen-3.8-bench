# blackwell-qwen-3.8-bench

8 GB VRAM'li bir Blackwell laptop GPU'sunda **Qwen3.8-27B** çalıştırma denemesi —
tahminle değil, ölçümle.

Amaç: "bu donanımda ne çalışır?" sorusunu forum tahminleriyle değil, tekrarlanabilir
benchmark'larla cevaplamak. Her bulgu ölçüme dayanıyor, her ölçüm burada.

## Test donanımı

| | |
|---|---|
| GPU | RTX 5070 Laptop, 8151 MiB VRAM, sm_120 (Blackwell) |
| CPU | Intel i7-14650HX, 24 thread, AVX2 + AVX-VNNI (AVX-512 **yok**) |
| RAM | 30 GB DDR5 |
| Motor | llama.cpp, kaynaktan derlenmiş (`sm_120a`, `-march=native`) |

## Temel kısıt

27B parametre, hiçbir quantization seviyesinde 8 GB VRAM'e sığmıyor:

```
BF16       54.7 GB      Q4_K_M     17.1 GB
FP8/INT8   29.0 GB      Q3_K_S     12.6 GB
NVFP4      ~14  GB      IQ2_XXS     9.0 GB   <- en kucugu, hala sigmiyor
```

Kullanılabilir VRAM ~7.3 GB. Bu yüzden vLLM ve TensorRT-LLM eleniyor (ikisi de
ağırlıkların tamamen VRAM'de olmasını şart koşuyor). Geriye llama.cpp'nin
hibrit GPU+CPU offload'u kalıyor.

## Ölçülen donanım limitleri

`bench/roofline.py` çıktısı:

```
GPU VRAM (salt oku) :  353.0 GB/s
CPU DRAM            :   43.9 GB/s      <- 8x yavas
PCIe H2D            :   18.2 GB/s
```

Decode bellek-bound olduğu için modelin CPU'da kalan kısmı her şeyi domine ediyor.

## Bulgular

### 1. `-ngl` süpürmesi — tek parametre, 3x hız

Boş bağlam, IQ2_XXS quant:

| `-ngl` | tg (t/s) |
|---:|---:|
| 20 | 5.28 |
| 30 | 6.05 |
| 40 | 7.61 |
| 46 | 9.30 |
| 50 | 10.78 |
| 54 | 12.87 |
| **56** | **15.82** |
| 58 | OOM |

Ölçümlerden türetilen model: `süre = 44.9 ms + 3.29 ms × (CPU'daki katman sayısı)`.
Tüm noktalarda %6'dan iyi tutuyor. Buradaki `3.29 ms/katman`, 36.7 GB/s efektif
CPU bant genişliğine karşılık geliyor — ölçtüğümüz 43.9 GB/s'in %84'ü.

### 2. Quantization tek tip değil

"IQ2_XXS" etiketi ortalama; gerçek dağılım (`bench/gguf_inspect.py`):

```
ORTALAMA: 2.59 bit/agirlik   (2.00 degil)

cikis katmani (lm_head)    3.19 bit   <- korunmus
linear attention (SSM)     3.08 bit   <- korunmus
attention                  2.61 bit
FFN gate/up + down         2.53 bit   <- feda edilmis (modelin %64'u)
norm                      32.00 bit   <- hic dokunulmamis
```

SSM katmanlarının korunması dikkat çekici: recurrent state token token güncellendiği
için quantization hatası birikiyor, klasik attention'da birikmiyor.

### 3. KV cache quantization — kendi başına hızlandırma değil

8K bağlamda:

| KV tipi | max `-ngl` | tg (t/s) |
|---|---:|---:|
| f16 | 50 | 10.25 |
| q8_0 | 52 | **10.48** |

Aynı `ngl=50`'de q8_0 aslında **daha yavaş** (9.82 vs 10.25) — dequantization maliyeti
bant genişliği kazancını aşıyor. Sadece daha fazla katman sığdırdığı için net kârlı,
o da %2.2.

### 4. Bağlam uzunluğu hızı VRAM üzerinden düşürüyor

```
bos baglam  ->  max ngl 56  ->  15.82 t/s
8K baglam   ->  max ngl 52  ->  10.48 t/s   (-%34)
```

Kayıp dikkat hesabından değil (aynı `ngl`'de sadece −%5), ağırlıkların yerini
KV cache'in çalmasından. Hibrit mimari sayesinde KV maliyeti token başına 64 KB —
tam-attention bir 27B'de 4 katı olurdu.

### 5. MTP speculative decoding — kazanç yok (henüz doğrulanmadı)

Model kendi MTP kafasını taşıyor (`blk.64.nextn.*`). Ama:

| `-ngl` | mtp kapalı | mtp açık |
|---:|---:|---:|
| 44 | 9.6 | 8.8 |
| 46 | 9.3 | 8.3 |
| 48 | 8.7 | 9.1 |
| 50 | 10.4 | 10.4 |

Hiçbiri `ngl=54` baseline'ını (12.0 t/s) geçemiyor. MTP ~660 MB ek VRAM istiyor
(bunun `152 MB × tahmin_derinliği` kadarı **recurrent state checkpoint'leri** —
reddedilen tahminde SSM state'ini geri sarabilmek için gerekli).

**Bu sonuç kesin değil:** ölçümde ~%10 gürültü var (monotonluk bozuluyor), termal
throttling şüphesi var, ve kabul oranı (acceptance rate) telemetrisi elde edilemedi.
`llama-server` + JSON `timings` ile tekrar ölçülecek.

## Dosyalar

```
bench/roofline.py       donanim bant genisligi taban olcumu
bench/gguf_inspect.py   GGUF quant tipi / bit dagilimi analizi
bench/results/          olcum ciktilari
```

## Durum

Devam ediyor. Sıradakiler: `llama-server` tabanlı sıkı ölçüm harness'ı,
perplexity ile kalite ekseni, ve VRAM'e tamamen sığan küçük bir modelle
(9B @ 4-bit) karşılaştırma — *"8 GB'da 27B@2bit mi kazanır, 9B@4bit mi?"*

## Lisans

MIT
